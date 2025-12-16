"""
Full-scale comparison: Greedy vs Temperature vs EmbedNoise

Based on preliminary results:
- 6.7B: EmbedNoise (σ=0.002) achieved Pass@1=0.44 vs Temperature's 0.32 (+37%!)
- 33B: EmbedNoise (σ=0.001) achieved 67% compile rate vs 33% for greedy

This script runs a larger-scale test to validate the findings.
"""
import torch
import time
import json
from datetime import datetime
from tqdm import tqdm

from src.model_loader import load_model
from src.embedding_noise import EmbeddingNoiseInjector
from src.generation import generate_solution, GenerationConfig
from src.humaneval_loader import load_humaneval, format_prompt_for_model, extract_function_code, check_solution
from src.metrics import distinct_n, self_bleu, compilation_rate, pass_at_k, mean_edit_distance


# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"
MODEL_TYPE = "deepseek"
LOAD_IN_8BIT = True

N_PROBLEMS = 50  # More problems for statistical significance
K_SAMPLES = 20   # Samples per method per problem

# Methods to compare (based on preliminary findings)
METHODS = {
    "greedy": {"do_sample": False},
    "temp_0.8": {"do_sample": True, "temperature": 0.8},
    "temp_0.6": {"do_sample": True, "temperature": 0.6},  # Lower temp might help
    "noise_0.001": {"sigma": 0.001},
    "noise_0.002": {"sigma": 0.002},  # Best for 6.7B
    "noise_0.005": {"sigma": 0.005},
}
# ============================================================


def run_full_comparison():
    print("=" * 70)
    print("Full-Scale Comparison: Temperature vs Embedding Noise")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Problems: {N_PROBLEMS}, Samples/method: {K_SAMPLES}")
    print(f"Methods: {list(METHODS.keys())}")
    
    # Load model
    print("\n[1/3] Loading model...")
    model, tokenizer = load_model(MODEL_NAME, load_in_8bit=LOAD_IN_8BIT)
    
    # Load problems
    print("\n[2/3] Loading HumanEval problems...")
    problems = load_humaneval(n_samples=N_PROBLEMS)
    
    # Results storage
    results = {method: [] for method in METHODS}
    
    print(f"\n[3/3] Running experiments...")
    total_gens = N_PROBLEMS * K_SAMPLES * len(METHODS)
    print(f"Total generations: {total_gens}")
    
    for prob_idx, problem in enumerate(tqdm(problems, desc="Problems")):
        prompt = format_prompt_for_model(problem, model_type=MODEL_TYPE)
        
        for method_name, method_config in METHODS.items():
            solutions = []
            
            if method_name.startswith("noise_"):
                # Embedding noise method
                sigma = method_config["sigma"]
                noise_injector = EmbeddingNoiseInjector(sigma_scale=sigma, noise_scope="per_token")
                noise_injector.attach_to_model(model)
                
                for seed in range(K_SAMPLES):
                    noise_injector.set_seed(seed)
                    noise_injector.activate()
                    
                    config = GenerationConfig(do_sample=False, max_new_tokens=256)
                    sol = generate_solution(model, tokenizer, prompt, config)
                    
                    noise_injector.deactivate()
                    
                    extracted = extract_function_code(sol, problem.entry_point)
                    passed = check_solution(problem, extracted)
                    solutions.append({"solution": extracted, "passed": passed})
                
                noise_injector.detach()
            
            elif method_name == "greedy":
                # Greedy - only need 1 sample (deterministic)
                config = GenerationConfig(do_sample=False, max_new_tokens=256)
                sol = generate_solution(model, tokenizer, prompt, config)
                extracted = extract_function_code(sol, problem.entry_point)
                passed = check_solution(problem, extracted)
                # Duplicate for fair comparison
                for _ in range(K_SAMPLES):
                    solutions.append({"solution": extracted, "passed": passed})
            
            else:
                # Temperature sampling
                temp = method_config.get("temperature", 0.8)
                for _ in range(K_SAMPLES):
                    config = GenerationConfig(do_sample=True, temperature=temp, max_new_tokens=256)
                    sol = generate_solution(model, tokenizer, prompt, config)
                    extracted = extract_function_code(sol, problem.entry_point)
                    passed = check_solution(problem, extracted)
                    solutions.append({"solution": extracted, "passed": passed})
            
            results[method_name].extend(solutions)
    
    # Compute metrics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    summary = {}
    for method_name, data in results.items():
        solutions = [d["solution"] for d in data]
        passed_list = [d["passed"] for d in data]
        
        n_total = len(passed_list)
        n_correct = sum(passed_list)
        
        metrics = {
            "pass_at_1": pass_at_k(n_total, n_correct, 1),
            "pass_at_5": pass_at_k(n_total, n_correct, 5),
            "pass_at_10": pass_at_k(n_total, n_correct, 10),
            "compile_rate": compilation_rate(solutions),
            "distinct_1": distinct_n(solutions, 1),
            "distinct_2": distinct_n(solutions, 2),
            "self_bleu": self_bleu(solutions),
            "n_total": n_total,
            "n_correct": n_correct,
        }
        summary[method_name] = metrics
    
    # Print table
    print(f"\n{'Method':<15} {'Pass@1':<8} {'Pass@5':<8} {'Pass@10':<8} {'Compile%':<10} {'Distinct-2':<10}")
    print("-" * 70)
    for method, m in summary.items():
        print(f"{method:<15} {m['pass_at_1']:<8.3f} {m['pass_at_5']:<8.3f} {m['pass_at_10']:<8.3f} {m['compile_rate']*100:<10.1f} {m['distinct_2']:<10.3f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump({
            "config": {
                "model": MODEL_NAME,
                "n_problems": N_PROBLEMS,
                "k_samples": K_SAMPLES,
            },
            "summary": summary,
        }, f, indent=2)
    print(f"\nResults saved to {filename}")
    
    # Highlight best method
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    best_pass1 = max(summary.items(), key=lambda x: x[1]["pass_at_1"])
    best_compile = max(summary.items(), key=lambda x: x[1]["compile_rate"])
    
    print(f"Best Pass@1: {best_pass1[0]} ({best_pass1[1]['pass_at_1']:.3f})")
    print(f"Best Compile%: {best_compile[0]} ({best_compile[1]['compile_rate']*100:.1f}%)")
    
    # Compare noise vs temperature
    noise_methods = [k for k in summary if k.startswith("noise_")]
    temp_methods = [k for k in summary if k.startswith("temp_")]
    
    if noise_methods and temp_methods:
        best_noise = max(noise_methods, key=lambda x: summary[x]["pass_at_1"])
        best_temp = max(temp_methods, key=lambda x: summary[x]["pass_at_1"])
        
        n = summary[best_noise]
        t = summary[best_temp]
        
        print(f"\nBest Noise ({best_noise}) vs Best Temp ({best_temp}):")
        print(f"  Pass@1:     {n['pass_at_1']:.3f} vs {t['pass_at_1']:.3f} ({'+' if n['pass_at_1'] > t['pass_at_1'] else ''}{(n['pass_at_1']-t['pass_at_1'])*100:.1f}%)")
        print(f"  Compile%:   {n['compile_rate']*100:.1f}% vs {t['compile_rate']*100:.1f}%")
        print(f"  Distinct-2: {n['distinct_2']:.3f} vs {t['distinct_2']:.3f}")


if __name__ == "__main__":
    run_full_comparison()

