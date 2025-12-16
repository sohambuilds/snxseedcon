"""
Check if small sigma values produce meaningful diversity across seeds.

The key question: Does σ=0.001 with different seeds produce different outputs,
or is it essentially deterministic like greedy?

Run: python test_diversity_check.py
"""
import torch
from src.model_loader import load_model
from src.embedding_noise import EmbeddingNoiseInjector
from src.generation import generate_solution, GenerationConfig
from src.humaneval_loader import load_humaneval, format_prompt_for_model
from src.metrics import distinct_n, self_bleu, compilation_rate
import config


def check_diversity():
    """Generate multiple samples at low sigma and check if they're actually diverse."""
    
    print("=" * 70)
    print("Diversity Check: Do different seeds produce different outputs?")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(config.MODEL_NAME, load_in_8bit=config.LOAD_IN_8BIT)
    
    # Load one problem
    problems = load_humaneval(n_samples=1)
    problem = problems[0]
    prompt = format_prompt_for_model(problem, model_type=config.MODEL_TYPE)
    
    print(f"\nProblem: {problem.task_id}")
    
    # Test sigma values in the "promising" range
    sigma_values = [0.001, 0.002, 0.005, 0.008, 0.01]
    n_samples = 5
    
    for sigma in sigma_values:
        print(f"\n{'='*70}")
        print(f"σ = {sigma} — Generating {n_samples} samples with different seeds")
        print('='*70)
        
        noise_injector = EmbeddingNoiseInjector(
            sigma_scale=sigma,
            noise_scope="per_token",
        )
        noise_injector.attach_to_model(model)
        
        samples = []
        for seed in range(n_samples):
            noise_injector.set_seed(seed)
            noise_injector.activate()
            
            config_gen = GenerationConfig(do_sample=False, max_new_tokens=150)
            solution = generate_solution(model, tokenizer, prompt, config_gen)
            samples.append(solution)
            
            noise_injector.deactivate()
        
        noise_injector.detach()
        
        # Check diversity
        unique_samples = len(set(samples))
        compile_rate = compilation_rate([problem.prompt + s for s in samples])
        d1 = distinct_n(samples, 1)
        d2 = distinct_n(samples, 2)
        
        print(f"\nResults:")
        print(f"  Unique outputs: {unique_samples}/{n_samples}")
        print(f"  Compile rate:   {compile_rate*100:.0f}%")
        print(f"  Distinct-1:     {d1:.3f}")
        print(f"  Distinct-2:     {d2:.3f}")
        
        # Show first 100 chars of each sample
        print(f"\nSample previews (first 100 chars):")
        for i, s in enumerate(samples):
            preview = s[:100].replace('\n', '↵')
            print(f"  [{i}] {preview}")
        
        # Are they all identical?
        if unique_samples == 1:
            print("\n  ⚠️  ALL SAMPLES IDENTICAL — no diversity at this σ")
        elif unique_samples == n_samples:
            print("\n  ✓ All samples unique!")
        else:
            print(f"\n  ~ Some diversity: {unique_samples} unique out of {n_samples}")
    
    # Also test per-sequence noise (might allow higher sigma)
    print("\n" + "=" * 70)
    print("BONUS: Testing per-sequence noise (same noise vector for all tokens)")
    print("=" * 70)
    
    for sigma in [0.01, 0.05, 0.1]:
        print(f"\n--- σ = {sigma}, noise_scope = per_sequence ---")
        
        noise_injector = EmbeddingNoiseInjector(
            sigma_scale=sigma,
            noise_scope="per_sequence",  # Less aggressive!
        )
        noise_injector.attach_to_model(model)
        
        samples = []
        for seed in range(3):
            noise_injector.set_seed(seed)
            noise_injector.activate()
            
            config_gen = GenerationConfig(do_sample=False, max_new_tokens=150)
            solution = generate_solution(model, tokenizer, prompt, config_gen)
            samples.append(solution)
            
            noise_injector.deactivate()
        
        noise_injector.detach()
        
        compile_rate = compilation_rate([problem.prompt + s for s in samples])
        unique_samples = len(set(samples))
        
        print(f"  Unique: {unique_samples}/3, Compile: {compile_rate*100:.0f}%")
        preview = samples[0][:80].replace('\n', '↵')
        print(f"  Sample: {preview}...")
    
    print("\n" + "=" * 70)
    print("Check complete!")


if __name__ == "__main__":
    check_diversity()

