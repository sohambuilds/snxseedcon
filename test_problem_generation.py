"""
Experiment: Problem Generation with Embedding Noise

Instead of generating solutions, we generate PROBLEMS.
This is a better test of creativity because:
1. No single correct answer - many valid problems exist
2. Diversity is meaningful - different topics, difficulties, styles
3. We can still measure quality (valid syntax, has tests, solvable)

Run: python test_problem_generation.py
"""
import torch
import time
from tqdm import tqdm
from typing import List

from src.model_loader import load_model
from src.embedding_noise import EmbeddingNoiseInjector
from src.generation import generate_solution, GenerationConfig
from src.problem_generation import (
    get_generation_prompt,
    parse_generated_problem,
    compute_problem_diversity,
    GeneratedProblem,
    TOPIC_PROMPTS,
)
from src.metrics import distinct_n, self_bleu


# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"
MODEL_TYPE = "deepseek"
LOAD_IN_8BIT = True

N_PROBLEMS = 10  # Problems to generate per method
TOPICS = ["string", "list", "math", "game", "text"]  # Topics to test

# Methods to compare
SIGMA_VALUES = [0.002, 0.005, 0.01, 0.02]
# ============================================================


def format_prompt(topic: str = None, model_type: str = "deepseek") -> str:
    """Format the problem generation prompt for the model."""
    base_prompt = get_generation_prompt(topic)
    
    if model_type == "deepseek":
        return f"You are an expert programming instructor.\n\n{base_prompt}"
    else:
        return base_prompt


def generate_problems(
    model, 
    tokenizer, 
    method: str,
    n_problems: int,
    noise_injector=None,
    sigma: float = None,
) -> List[GeneratedProblem]:
    """Generate problems using the specified method."""
    
    problems = []
    
    for i in range(n_problems):
        # Rotate through topics for variety
        topic = TOPICS[i % len(TOPICS)]
        prompt = format_prompt(topic, MODEL_TYPE)
        
        if method == "greedy":
            config = GenerationConfig(do_sample=False, max_new_tokens=400)
            output = generate_solution(model, tokenizer, prompt, config)
            
        elif method == "temperature":
            config = GenerationConfig(do_sample=True, temperature=0.8, max_new_tokens=400)
            output = generate_solution(model, tokenizer, prompt, config)
            
        elif method == "embed_noise":
            noise_injector.set_seed(i)
            noise_injector.activate()
            config = GenerationConfig(do_sample=False, max_new_tokens=400)
            output = generate_solution(model, tokenizer, prompt, config)
            noise_injector.deactivate()
        
        problem = parse_generated_problem(output)
        problems.append(problem)
    
    return problems


def run_problem_generation_experiment():
    """Main experiment comparing methods for problem generation."""
    
    print("=" * 70)
    print("Problem Generation Experiment")
    print("=" * 70)
    print("Hypothesis: Embedding noise should produce more diverse")
    print("coding problems than temperature sampling, while maintaining validity.")
    print("=" * 70)
    
    # Load model
    print("\n[1/3] Loading model...")
    model, tokenizer = load_model(MODEL_NAME, load_in_8bit=LOAD_IN_8BIT)
    
    # Results storage
    results = {}
    
    print(f"\n[2/3] Generating problems...")
    print(f"Problems per method: {N_PROBLEMS}")
    
    # Greedy baseline
    print("\n--- Greedy (baseline) ---")
    results["greedy"] = generate_problems(model, tokenizer, "greedy", N_PROBLEMS)
    valid = sum(1 for p in results["greedy"] if p.is_valid)
    print(f"Valid: {valid}/{N_PROBLEMS}")
    
    # Temperature sampling
    print("\n--- Temperature (0.8) ---")
    results["temperature"] = generate_problems(model, tokenizer, "temperature", N_PROBLEMS)
    valid = sum(1 for p in results["temperature"] if p.is_valid)
    print(f"Valid: {valid}/{N_PROBLEMS}")
    
    # Embedding noise at different sigmas
    for sigma in SIGMA_VALUES:
        print(f"\n--- EmbedNoise (σ={sigma}) ---")
        
        noise_injector = EmbeddingNoiseInjector(
            sigma_scale=sigma,
            noise_scope="per_token",
        )
        noise_injector.attach_to_model(model)
        
        results[f"noise_{sigma}"] = generate_problems(
            model, tokenizer, "embed_noise", N_PROBLEMS,
            noise_injector=noise_injector, sigma=sigma
        )
        
        noise_injector.detach()
        
        valid = sum(1 for p in results[f"noise_{sigma}"] if p.is_valid)
        print(f"Valid: {valid}/{N_PROBLEMS}")
    
    # Compute and display metrics
    print("\n" + "=" * 70)
    print("[3/3] RESULTS")
    print("=" * 70)
    
    print(f"\n{'Method':<20} {'Valid%':<10} {'UniqueNames':<12} {'UniqueSigs':<12} {'AvgTests':<10}")
    print("-" * 70)
    
    for method, problems in results.items():
        metrics = compute_problem_diversity(problems)
        
        valid_rate = metrics.get("valid_rate", 0) * 100
        unique_names = metrics.get("unique_names", 0)
        unique_sigs = metrics.get("unique_signatures", 0)
        avg_tests = metrics.get("avg_test_cases", 0)
        
        print(f"{method:<20} {valid_rate:<10.1f} {unique_names:<12.3f} {unique_sigs:<12.3f} {avg_tests:<10.1f}")
    
    # Show example problems
    print("\n" + "=" * 70)
    print("EXAMPLE PROBLEMS")
    print("=" * 70)
    
    for method in ["greedy", "temperature", f"noise_{SIGMA_VALUES[0]}"]:
        print(f"\n--- {method} ---")
        problems = results[method]
        valid_problems = [p for p in problems if p.is_valid]
        
        if valid_problems:
            p = valid_problems[0]
            print(f"Function: {p.function_name}")
            print(f"Signature: {p.signature}")
            print(f"Tests: {len(p.test_cases)}")
            print(f"Preview: {p.docstring[:200]}..." if len(p.docstring) > 200 else f"Preview: {p.docstring}")
        else:
            print("No valid problems generated")
            # Show first invalid one for debugging
            if problems:
                print(f"Errors: {problems[0].validation_errors}")
                print(f"Raw output preview: {problems[0].raw_output[:200]}...")
    
    # Diversity analysis
    print("\n" + "=" * 70)
    print("DIVERSITY ANALYSIS")
    print("=" * 70)
    
    for method in ["greedy", "temperature", f"noise_{SIGMA_VALUES[0]}"]:
        problems = results[method]
        valid_problems = [p for p in problems if p.is_valid]
        
        if len(valid_problems) >= 2:
            # Text diversity of docstrings
            docstrings = [p.docstring for p in valid_problems]
            d2 = distinct_n(docstrings, 2)
            sb = self_bleu(docstrings)
            
            print(f"\n{method}:")
            print(f"  Docstring Distinct-2: {d2:.3f}")
            print(f"  Docstring Self-BLEU: {sb:.3f} (lower = more diverse)")
            
            # Show all function names
            names = [p.function_name for p in valid_problems]
            print(f"  Function names: {names}")
    
    print("\n" + "=" * 70)
    print("Experiment complete!")
    
    return results


if __name__ == "__main__":
    run_problem_generation_experiment()

