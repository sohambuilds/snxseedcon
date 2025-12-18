"""
Quick inspection of generated outputs.

Use this to visually inspect what each method produces.
Helps debug issues like gibberish output from too-high sigma.

Run: python test_inspect_outputs.py
"""
import torch
from src.model_loader import load_model
from src.embedding_noise import EmbeddingNoiseInjector
from src.generation import generate_k_solutions
from src.humaneval_loader import load_humaneval, format_prompt_for_model
import config


def inspect_outputs():
    """Generate and display outputs from different methods for visual inspection."""
    
    print("=" * 70)
    print("Output Inspection")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(config.MODEL_NAME, load_in_8bit=config.LOAD_IN_8BIT)
    
    # Load one problem
    problems = load_humaneval(n_samples=1)
    problem = problems[0]
    prompt = format_prompt_for_model(problem, model_type=config.MODEL_TYPE)
    
    print(f"\nProblem: {problem.task_id}")
    print(f"\nPrompt:\n{problem.prompt[:300]}...\n")
    
    # Test different sigma values
    sigma_values = [0.001, 0.01, 0.05, 0.1]
    
    # First, show greedy baseline
    print("=" * 70)
    print("GREEDY (baseline)")
    print("=" * 70)
    solutions = generate_k_solutions(
        model, tokenizer, prompt, k=1, method="greedy",
        method_kwargs={"max_new_tokens": 256}
    )
    print(solutions[0][:500])
    
    # Then, show temperature
    print("\n" + "=" * 70)
    print("TEMPERATURE (0.8)")
    print("=" * 70)
    solutions = generate_k_solutions(
        model, tokenizer, prompt, k=1, method="temperature",
        method_kwargs={"temperature": 0.8, "max_new_tokens": 256}
    )
    print(solutions[0][:500])
    
    # Now test each sigma
    for sigma in sigma_values:
        print("\n" + "=" * 70)
        print(f"EMBED_NOISE (σ = {sigma})")
        print("=" * 70)
        
        noise_injector = EmbeddingNoiseInjector(
            sigma_scale=sigma,
            noise_scope=config.NOISE_SCOPE,
        )
        noise_injector.attach_to_model(model)
        
        solutions = generate_k_solutions(
            model, tokenizer, prompt, k=1, method="embed_noise",
            method_kwargs={"max_new_tokens": 256},
            noise_injector=noise_injector
        )
        
        output = solutions[0][:500]
        print(output)
        
        # Check if it compiles
        try:
            compile(problem.prompt + output, '<string>', 'exec')
            print("\n✓ COMPILES")
        except SyntaxError as e:
            print(f"\n✗ SYNTAX ERROR: {e.msg}")
        
        noise_injector.detach()
    
    print("\n" + "=" * 70)
    print("Inspection complete!")


if __name__ == "__main__":
    inspect_outputs()


