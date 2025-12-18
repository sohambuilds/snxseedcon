"""
Sigma sweep experiment: Find optimal noise magnitude.

The initial test showed σ=0.1 completely destroys coherence (0% compile).
This script sweeps over smaller σ values to find the sweet spot.

Run: python test_sigma_sweep.py
"""
import torch
import time
from tqdm import tqdm

from src.model_loader import load_model
from src.embedding_noise import EmbeddingNoiseInjector
from src.generation import generate_k_solutions
from src.humaneval_loader import load_humaneval, format_prompt_for_model, extract_function_code, check_solution
from src.metrics import distinct_n, self_bleu, compilation_rate, pass_at_k
import config


def run_sigma_sweep():
    """Sweep over sigma values to find optimal noise magnitude."""
    
    # Configuration
    MODEL_NAME = config.MODEL_NAME
    N_PROBLEMS = 3  # Fewer problems for faster sweep
    K_SAMPLES = 5   # Fewer samples for faster sweep
    SIGMA_VALUES = config.SIGMA_SWEEP_VALUES
    
    print("=" * 70)
    print("Sigma Sweep: Finding Optimal Embedding Noise Magnitude")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Problems: {N_PROBLEMS}, Samples per sigma: {K_SAMPLES}")
    print(f"Sigma values: {SIGMA_VALUES}")
    
    # Load model
    print("\n[1/3] Loading model...")
    model, tokenizer = load_model(MODEL_NAME, load_in_8bit=config.LOAD_IN_8BIT)
    
    # Load dataset
    print("\n[2/3] Loading HumanEval problems...")
    problems = load_humaneval(n_samples=N_PROBLEMS)
    
    # Results storage
    results = {}
    
    print("\n[3/3] Running sigma sweep...")
    
    for sigma in SIGMA_VALUES:
        print(f"\n{'='*60}")
        print(f"Testing σ = {sigma}")
        print('='*60)
        
        # Setup noise injector with this sigma
        noise_injector = EmbeddingNoiseInjector(
            sigma_scale=sigma,
            noise_scope=config.NOISE_SCOPE,
        )
        noise_injector.attach_to_model(model)
        
        all_solutions = []
        all_passed = []
        
        for prob_idx, problem in enumerate(problems, 1):
            print(f"  Problem {prob_idx}/{N_PROBLEMS}: {problem.task_id}")
            prompt = format_prompt_for_model(problem, model_type=config.MODEL_TYPE)
            
            solutions = generate_k_solutions(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                k=K_SAMPLES,
                method="embed_noise",
                method_kwargs={"max_new_tokens": config.MAX_NEW_TOKENS},
                noise_injector=noise_injector,
                show_progress=True,
            )
            
            extracted = [extract_function_code(s, problem.entry_point) for s in solutions]
            passed = [check_solution(problem, s) for s in extracted]
            
            all_solutions.extend(extracted)
            all_passed.extend(passed)
            
            print(f"    → {sum(passed)}/{K_SAMPLES} passed")
        
        # Detach before creating new injector
        noise_injector.detach()
        
        # Compute metrics
        n_total = len(all_passed)
        n_correct = sum(all_passed)
        
        results[sigma] = {
            "pass_at_1": pass_at_k(n_total, n_correct, 1),
            "pass_at_5": pass_at_k(n_total, n_correct, 5) if n_total >= 5 else None,
            "distinct_1": distinct_n(all_solutions, 1),
            "distinct_2": distinct_n(all_solutions, 2),
            "self_bleu": self_bleu(all_solutions),
            "compile_rate": compilation_rate(all_solutions),
            "solutions": all_solutions[:3],  # Save a few examples
        }
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SIGMA SWEEP RESULTS")
    print("=" * 70)
    print(f"{'Sigma':<10} {'Pass@1':<10} {'Compile%':<12} {'Distinct-2':<12} {'Self-BLEU':<12}")
    print("-" * 70)
    
    for sigma, r in results.items():
        print(f"{sigma:<10.4f} {r['pass_at_1']:<10.3f} {r['compile_rate']*100:<12.1f} {r['distinct_2']:<12.3f} {r['self_bleu']:<12.3f}")
    
    # Find best sigma (highest Pass@1 with compile > 50%)
    viable = {s: r for s, r in results.items() if r['compile_rate'] > 0.5}
    if viable:
        best_sigma = max(viable.keys(), key=lambda s: viable[s]['pass_at_1'])
        print(f"\n→ Best sigma (Pass@1 with >50% compile): {best_sigma}")
    
    # Show example outputs for each sigma
    print("\n" + "=" * 70)
    print("EXAMPLE OUTPUTS (first solution per sigma)")
    print("=" * 70)
    for sigma, r in results.items():
        print(f"\n--- σ = {sigma} ---")
        if r['solutions']:
            example = r['solutions'][0][:200]  # First 200 chars
            print(f"{example}...")
    
    print("\n" + "=" * 70)
    print("Sweep complete!")
    
    return results


if __name__ == "__main__":
    run_sigma_sweep()



