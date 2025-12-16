"""
Test embedding noise with a larger model.

Hypothesis: Larger models may have more robust embedding spaces
that tolerate noise better, allowing higher σ values.

Run: CUDA_VISIBLE_DEVICES=0 python test_large_model.py

(Run concurrently with test_minimal.py on a different GPU)
"""
import torch
import time
from src.model_loader import load_model
from src.embedding_noise import EmbeddingNoiseInjector
from src.generation import generate_solution, GenerationConfig
from src.humaneval_loader import load_humaneval, format_prompt_for_model, extract_function_code, check_solution
from src.metrics import distinct_n, compilation_rate, pass_at_k


# ============================================================
# CONFIGURATION - Edit this to test different models
# ============================================================

# Options to try:
# - "deepseek-ai/deepseek-coder-33b-instruct"  # 33B - needs ~20GB VRAM with int8
# - "Qwen/Qwen2.5-Coder-7B-Instruct"           # 7B Qwen
# - "Qwen/Qwen2.5-Coder-14B-Instruct"          # 14B Qwen - needs ~10GB with int8
# - "codellama/CodeLlama-13b-Instruct-hf"      # 13B CodeLlama

MODEL_NAME = "deepseek-ai/deepseek-coder-33b-instruct"
MODEL_TYPE = "deepseek"  # "deepseek", "qwen", or "generic"
LOAD_IN_8BIT = True

# Test parameters
N_PROBLEMS = 3
N_SAMPLES = 5
SIGMA_VALUES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

# ============================================================


def test_model_with_noise():
    """Test embedding noise on a larger model."""
    
    print("=" * 70)
    print(f"Testing Larger Model: {MODEL_NAME}")
    print("=" * 70)
    
    # Load model
    print("\n[1/3] Loading model...")
    start = time.time()
    model, tokenizer = load_model(MODEL_NAME, load_in_8bit=LOAD_IN_8BIT)
    print(f"Model loaded in {time.time()-start:.1f}s")
    
    # Load problems
    print("\n[2/3] Loading HumanEval problems...")
    problems = load_humaneval(n_samples=N_PROBLEMS)
    
    # Results storage
    results = {"greedy": [], "temperature": []}
    for sigma in SIGMA_VALUES:
        results[f"noise_{sigma}"] = []
    
    print(f"\n[3/3] Running experiments...")
    print(f"Problems: {N_PROBLEMS}, Samples: {N_SAMPLES}, Sigmas: {SIGMA_VALUES}")
    
    for prob_idx, problem in enumerate(problems):
        print(f"\n--- Problem {prob_idx+1}/{N_PROBLEMS}: {problem.task_id} ---")
        prompt = format_prompt_for_model(problem, model_type=MODEL_TYPE)
        
        # Greedy baseline (1 sample - deterministic)
        print("  Greedy...", end=" ", flush=True)
        config = GenerationConfig(do_sample=False, max_new_tokens=256)
        sol = generate_solution(model, tokenizer, prompt, config)
        extracted = extract_function_code(sol, problem.entry_point)
        passed = check_solution(problem, extracted)
        results["greedy"].append({"solution": extracted, "passed": passed})
        print(f"{'✓' if passed else '✗'}")
        
        # Temperature baseline
        print("  Temperature 0.8...", end=" ", flush=True)
        temp_solutions = []
        for i in range(N_SAMPLES):
            config = GenerationConfig(do_sample=True, temperature=0.8, max_new_tokens=256)
            sol = generate_solution(model, tokenizer, prompt, config)
            extracted = extract_function_code(sol, problem.entry_point)
            passed = check_solution(problem, extracted)
            temp_solutions.append({"solution": extracted, "passed": passed})
        n_passed = sum(1 for s in temp_solutions if s["passed"])
        results["temperature"].extend(temp_solutions)
        print(f"{n_passed}/{N_SAMPLES} passed")
        
        # Test each sigma value
        for sigma in SIGMA_VALUES:
            print(f"  Noise σ={sigma}...", end=" ", flush=True)
            
            noise_injector = EmbeddingNoiseInjector(
                sigma_scale=sigma,
                noise_scope="per_token",
            )
            noise_injector.attach_to_model(model)
            
            noise_solutions = []
            for seed in range(N_SAMPLES):
                noise_injector.set_seed(seed)
                noise_injector.activate()
                
                config = GenerationConfig(do_sample=False, max_new_tokens=256)
                sol = generate_solution(model, tokenizer, prompt, config)
                
                noise_injector.deactivate()
                
                extracted = extract_function_code(sol, problem.entry_point)
                passed = check_solution(problem, extracted)
                noise_solutions.append({"solution": extracted, "passed": passed})
            
            noise_injector.detach()
            
            n_passed = sum(1 for s in noise_solutions if s["passed"])
            results[f"noise_{sigma}"].extend(noise_solutions)
            print(f"{n_passed}/{N_SAMPLES} passed")
    
    # Compute and display summary
    print("\n" + "=" * 70)
    print(f"RESULTS: {MODEL_NAME}")
    print("=" * 70)
    print(f"{'Method':<20} {'Pass@1':<10} {'Compile%':<12} {'Distinct-2':<12}")
    print("-" * 70)
    
    for method, data in results.items():
        if not data:
            continue
        
        solutions = [d["solution"] for d in data]
        passed_list = [d["passed"] for d in data]
        
        n_total = len(passed_list)
        n_correct = sum(passed_list)
        
        p1 = pass_at_k(n_total, n_correct, 1) if n_total > 0 else 0
        comp = compilation_rate(solutions)
        d2 = distinct_n(solutions, 2) if len(solutions) > 1 else 0
        
        print(f"{method:<20} {p1:<10.3f} {comp*100:<12.1f} {d2:<12.3f}")
    
    # Show example outputs at different sigmas
    print("\n" + "=" * 70)
    print("EXAMPLE OUTPUTS (Problem 1)")
    print("=" * 70)
    
    print("\nGreedy:")
    print(f"  {results['greedy'][0]['solution'][:150]}...")
    
    print("\nTemperature 0.8:")
    print(f"  {results['temperature'][0]['solution'][:150]}...")
    
    for sigma in [0.002, 0.01, 0.05]:
        key = f"noise_{sigma}"
        if results[key]:
            print(f"\nNoise σ={sigma}:")
            print(f"  {results[key][0]['solution'][:150]}...")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    
    return results


if __name__ == "__main__":
    test_model_with_noise()

