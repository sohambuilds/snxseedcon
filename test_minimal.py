"""
Minimal test script for embedding noise injection.

This script demonstrates the core concept with a small-scale test:
- One model (DeepSeek-Coder-6.7B for faster testing)
- A few HumanEval problems
- Comparison between greedy, temperature, and embedding noise

Run: python test_minimal.py
"""
import torch
from tqdm import tqdm

from src.model_loader import load_model
from src.embedding_noise import EmbeddingNoiseInjector
from src.generation import generate_k_solutions
from src.humaneval_loader import load_humaneval, format_prompt_for_model, extract_function_code, check_solution
from src.metrics import distinct_n, self_bleu, mean_edit_distance, compilation_rate, pass_at_k


def run_minimal_test():
    """Run a minimal test comparing generation methods."""
    
    # Configuration
    MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"  # Smaller model for testing
    N_PROBLEMS = 5  # Just a few problems for quick testing
    K_SAMPLES = 10  # Samples per method per problem
    SIGMA_SCALE = 0.1  # Noise scale
    
    print("=" * 60)
    print("Minimal Test: Embedding Noise for Diverse Code Generation")
    print("=" * 60)
    
    # Load model
    print("\n[1/4] Loading model...")
    model, tokenizer = load_model(MODEL_NAME, load_in_8bit=True)
    
    # Load dataset
    print("\n[2/4] Loading HumanEval problems...")
    problems = load_humaneval(n_samples=N_PROBLEMS)
    
    # Setup noise injector
    print("\n[3/4] Setting up embedding noise injector...")
    noise_injector = EmbeddingNoiseInjector(
        sigma_scale=SIGMA_SCALE,
        noise_scope="per_token",
    )
    noise_injector.attach_to_model(model)
    
    # Methods to compare
    methods = {
        "greedy": {},
        "temperature": {"temperature": 0.8},
        "embed_noise": {},
    }
    
    # Results storage
    results = {method: {"solutions": [], "pass_results": []} for method in methods}
    
    print("\n[4/4] Running generation experiments...")
    print(f"    Problems: {N_PROBLEMS}, Samples per method: {K_SAMPLES}")
    print(f"    Total generations: {N_PROBLEMS * K_SAMPLES * len(methods)}")
    
    for prob_idx, problem in enumerate(problems, 1):
        print(f"\n--- Problem {prob_idx}/{N_PROBLEMS}: {problem.task_id} ---")
        prompt = format_prompt_for_model(problem, model_type="deepseek")
        
        for method_name, method_kwargs in methods.items():
            print(f"  {method_name}:")
            
            # Generate K solutions
            injector = noise_injector if method_name == "embed_noise" else None
            
            solutions = generate_k_solutions(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                k=K_SAMPLES,
                method=method_name,
                method_kwargs=method_kwargs,
                noise_injector=injector,
                show_progress=True,
            )
            
            print(f"    Checking correctness... ", end="", flush=True)
            
            # Extract function code and check correctness
            extracted = [extract_function_code(s, problem.entry_point) for s in solutions]
            passed = [check_solution(problem, s) for s in extracted]
            
            n_passed = sum(passed)
            print(f"✓ ({n_passed}/{K_SAMPLES} passed)")
            
            results[method_name]["solutions"].extend(extracted)
            results[method_name]["pass_results"].extend(passed)
    
    # Compute and display metrics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for method_name in methods:
        solutions = results[method_name]["solutions"]
        pass_results = results[method_name]["pass_results"]
        
        n_total = len(pass_results)
        n_correct = sum(pass_results)
        
        print(f"\n{method_name.upper()}")
        print("-" * 40)
        
        # Correctness metrics
        print(f"  Pass@1:     {pass_at_k(n_total, n_correct, 1):.3f}")
        print(f"  Pass@5:     {pass_at_k(n_total, n_correct, 5):.3f}")
        print(f"  Pass@10:    {pass_at_k(n_total, n_correct, 10):.3f}")
        
        # Diversity metrics
        print(f"  Distinct-1: {distinct_n(solutions, 1):.3f}")
        print(f"  Distinct-2: {distinct_n(solutions, 2):.3f}")
        print(f"  Self-BLEU:  {self_bleu(solutions):.3f} (lower = more diverse)")
        print(f"  Edit Dist:  {mean_edit_distance(solutions):.3f} (higher = more diverse)")
        
        # Coherence
        print(f"  Compile %:  {compilation_rate(solutions) * 100:.1f}%")
    
    # Cleanup
    noise_injector.detach()
    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    run_minimal_test()

