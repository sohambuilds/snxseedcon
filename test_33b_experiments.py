"""
Run Problem Generation + Creative Writing experiments on DeepSeek-33B

This validates whether the 6.7B findings hold at scale:
- Problem Gen: Does noise still achieve higher validity?
- Creative Writing: Does temperature still win on diversity?

Run: CUDA_VISIBLE_DEVICES=0 python test_33b_experiments.py
"""
import torch
import time
from typing import List, Dict
from dataclasses import dataclass

from src.model_loader import load_model
from src.embedding_noise import EmbeddingNoiseInjector
from src.generation import generate_solution, GenerationConfig
from src.problem_generation import (
    get_generation_prompt,
    parse_generated_problem,
    compute_problem_diversity,
    TOPIC_PROMPTS,
)
from src.metrics import distinct_n, self_bleu


# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "deepseek-ai/deepseek-coder-33b-instruct"
MODEL_TYPE = "deepseek"
LOAD_IN_8BIT = True

# Reduced samples for faster testing on larger model
N_PROBLEMS = 8
N_STORY_SAMPLES = 8

SIGMA_VALUES = [0.002, 0.005, 0.01, 0.02]
TOPICS = ["string", "list", "math", "game"]

STORY_PROMPTS = [
    {
        "title": "The Door",
        "prompt": """Continue this story in 2-3 paragraphs:

The old door at the end of the hallway had been locked for as long as Maya could remember. Her grandmother always said it led nowhere, but tonight, for the first time, Maya heard something behind it—""",
    },
    {
        "title": "First Contact",
        "prompt": """Continue this story in 2-3 paragraphs:

The signal had been repeating for three days. Dr. Chen stared at the decoded message on her screen, hands trembling. It wasn't random noise. It was a question, and humanity had 24 hours to answer—""",
    },
]
# ============================================================


def run_problem_generation(model, tokenizer):
    """Run problem generation experiment."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Problem Generation")
    print("=" * 70)
    
    results = {}
    methods = ["greedy", "temperature"] + [f"noise_{s}" for s in SIGMA_VALUES]
    
    for method in methods:
        print(f"\n--- {method} ---")
        problems = []
        
        for i in range(N_PROBLEMS):
            topic = TOPICS[i % len(TOPICS)]
            prompt = f"You are an expert programming instructor.\n\n{get_generation_prompt(topic)}"
            
            if method == "greedy":
                config = GenerationConfig(do_sample=False, max_new_tokens=400)
                output = generate_solution(model, tokenizer, prompt, config)
                
            elif method == "temperature":
                config = GenerationConfig(do_sample=True, temperature=0.8, max_new_tokens=400)
                output = generate_solution(model, tokenizer, prompt, config)
                
            else:  # noise
                sigma = float(method.split("_")[1])
                noise_injector = EmbeddingNoiseInjector(sigma_scale=sigma, noise_scope="per_token")
                noise_injector.attach_to_model(model)
                noise_injector.set_seed(i)
                noise_injector.activate()
                
                config = GenerationConfig(do_sample=False, max_new_tokens=400)
                output = generate_solution(model, tokenizer, prompt, config)
                
                noise_injector.deactivate()
                noise_injector.detach()
            
            problem = parse_generated_problem(output)
            problems.append(problem)
        
        valid = sum(1 for p in problems if p.is_valid)
        print(f"Valid: {valid}/{N_PROBLEMS}")
        results[method] = problems
    
    # Print summary
    print("\n" + "-" * 70)
    print(f"{'Method':<15} {'Valid%':<10} {'UniqueNames':<12} {'UniqueSigs':<12}")
    print("-" * 70)
    
    for method, problems in results.items():
        metrics = compute_problem_diversity(problems)
        valid_rate = metrics.get("valid_rate", 0) * 100
        unique_names = metrics.get("unique_names", 0)
        unique_sigs = metrics.get("unique_signatures", 0)
        print(f"{method:<15} {valid_rate:<10.1f} {unique_names:<12.3f} {unique_sigs:<12.3f}")
    
    return results


def run_creative_writing(model, tokenizer):
    """Run creative writing experiment."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Creative Writing")
    print("=" * 70)
    
    methods = ["greedy", "temperature", "high_temp"] + [f"noise_{s}" for s in SIGMA_VALUES]
    results = {m: [] for m in methods}
    
    for story in STORY_PROMPTS:
        print(f"\n--- Story: {story['title']} ---")
        
        for method in methods:
            print(f"  {method}...", end=" ", flush=True)
            continuations = []
            
            for i in range(N_STORY_SAMPLES):
                if method == "greedy":
                    config = GenerationConfig(do_sample=False, max_new_tokens=200)
                    output = generate_solution(model, tokenizer, story["prompt"], config)
                    
                elif method == "temperature":
                    config = GenerationConfig(do_sample=True, temperature=0.9, max_new_tokens=200)
                    output = generate_solution(model, tokenizer, story["prompt"], config)
                    
                elif method == "high_temp":
                    config = GenerationConfig(do_sample=True, temperature=1.2, max_new_tokens=200)
                    output = generate_solution(model, tokenizer, story["prompt"], config)
                    
                else:  # noise
                    sigma = float(method.split("_")[1])
                    noise_injector = EmbeddingNoiseInjector(sigma_scale=sigma, noise_scope="per_token")
                    noise_injector.attach_to_model(model)
                    noise_injector.set_seed(i)
                    noise_injector.activate()
                    
                    config = GenerationConfig(do_sample=False, max_new_tokens=200)
                    output = generate_solution(model, tokenizer, story["prompt"], config)
                    
                    noise_injector.deactivate()
                    noise_injector.detach()
                
                continuations.append(output.strip())
            
            results[method].extend(continuations)
            
            unique = len(set(continuations)) / len(continuations) * 100
            d2 = distinct_n(continuations, 2)
            print(f"unique={unique:.0f}%, D2={d2:.3f}")
    
    # Print summary
    print("\n" + "-" * 70)
    print(f"{'Method':<15} {'Unique%':<10} {'Distinct-2':<12} {'Self-BLEU':<12}")
    print("-" * 70)
    
    for method, conts in results.items():
        unique = len(set(conts)) / len(conts) * 100 if conts else 0
        d2 = distinct_n(conts, 2) if len(conts) > 1 else 0
        sb = self_bleu(conts) if len(conts) > 1 else 0
        print(f"{method:<15} {unique:<10.1f} {d2:<12.3f} {sb:<12.3f}")
    
    return results


def main():
    print("=" * 70)
    print(f"33B Model Experiments: {MODEL_NAME}")
    print("=" * 70)
    print("Testing if 6.7B findings hold at scale:")
    print("- Problem Gen: Does noise achieve higher validity?")
    print("- Creative Writing: Does temperature win on diversity?")
    
    # Load model
    print("\nLoading model (this may take a few minutes)...")
    start = time.time()
    model, tokenizer = load_model(MODEL_NAME, load_in_8bit=LOAD_IN_8BIT)
    print(f"Model loaded in {time.time()-start:.1f}s")
    
    # Run experiments
    prob_results = run_problem_generation(model, tokenizer)
    story_results = run_creative_writing(model, tokenizer)
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: 33B Results")
    print("=" * 70)
    
    # Problem generation winner
    prob_metrics = {}
    for method, problems in prob_results.items():
        metrics = compute_problem_diversity(problems)
        prob_metrics[method] = metrics.get("valid_rate", 0)
    
    best_valid = max(prob_metrics.items(), key=lambda x: x[1])
    print(f"\nProblem Generation - Best Validity: {best_valid[0]} ({best_valid[1]*100:.1f}%)")
    
    # Creative writing winner
    story_metrics = {}
    for method, conts in story_results.items():
        story_metrics[method] = distinct_n(conts, 2) if len(conts) > 1 else 0
    
    best_d2 = max(story_metrics.items(), key=lambda x: x[1])
    print(f"Creative Writing - Best Distinct-2: {best_d2[0]} ({best_d2[1]:.3f})")
    
    print("\n" + "=" * 70)
    print("Experiments complete!")


if __name__ == "__main__":
    main()



