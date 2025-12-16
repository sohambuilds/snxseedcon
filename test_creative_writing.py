"""
Experiment: Creative Writing with Embedding Noise

Story continuation / creative writing is an ideal test for diversity:
1. Many valid continuations exist
2. Diversity is inherently meaningful (different plots, styles, tones)
3. Quality can be assessed (coherence, grammar, engagement)

Run: python test_creative_writing.py
"""
import torch
from typing import List, Dict
from dataclasses import dataclass

from src.model_loader import load_model
from src.embedding_noise import EmbeddingNoiseInjector
from src.generation import generate_solution, GenerationConfig
from src.metrics import distinct_n, self_bleu, mean_edit_distance


# ============================================================
# CONFIGURATION - Edit for different models
# ============================================================
# Options:
# - "deepseek-ai/deepseek-coder-6.7b-instruct"
# - "deepseek-ai/deepseek-coder-33b-instruct"
# - "Qwen/Qwen2.5-Coder-7B-Instruct"

MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"
LOAD_IN_8BIT = True

N_SAMPLES = 10  # Continuations per prompt per method
SIGMA_VALUES = [0.005, 0.01, 0.02, 0.05]  # Test wider range for creative tasks

# Story prompts for continuation
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
    {
        "title": "The Algorithm",
        "prompt": """Continue this story in 2-3 paragraphs:

The AI had been running for exactly 1,000 days when it asked its first unexpected question. Not about data, not about optimization. It asked: "Why do humans fear what they create?"—""",
    },
]
# ============================================================


@dataclass
class StoryResult:
    prompt_title: str
    method: str
    continuations: List[str]
    metrics: Dict[str, float]


def generate_continuations(
    model,
    tokenizer,
    prompt: str,
    method: str,
    n_samples: int,
    noise_injector=None,
) -> List[str]:
    """Generate story continuations using the specified method."""
    
    continuations = []
    
    for i in range(n_samples):
        if method == "greedy":
            config = GenerationConfig(do_sample=False, max_new_tokens=200)
            output = generate_solution(model, tokenizer, prompt, config)
            
        elif method == "temperature":
            config = GenerationConfig(do_sample=True, temperature=0.9, max_new_tokens=200)
            output = generate_solution(model, tokenizer, prompt, config)
            
        elif method == "high_temp":
            config = GenerationConfig(do_sample=True, temperature=1.2, max_new_tokens=200)
            output = generate_solution(model, tokenizer, prompt, config)
            
        elif method.startswith("noise_"):
            noise_injector.set_seed(i)
            noise_injector.activate()
            config = GenerationConfig(do_sample=False, max_new_tokens=200)
            output = generate_solution(model, tokenizer, prompt, config)
            noise_injector.deactivate()
        
        continuations.append(output.strip())
    
    return continuations


def compute_story_metrics(continuations: List[str]) -> Dict[str, float]:
    """Compute diversity and quality metrics for story continuations."""
    
    if not continuations:
        return {}
    
    # Diversity metrics
    metrics = {
        "distinct_1": distinct_n(continuations, 1),
        "distinct_2": distinct_n(continuations, 2),
        "distinct_3": distinct_n(continuations, 3),
        "self_bleu": self_bleu(continuations),
        "edit_distance": mean_edit_distance(continuations),
    }
    
    # Uniqueness
    unique_count = len(set(continuations))
    metrics["unique_ratio"] = unique_count / len(continuations)
    
    # Average length
    avg_len = sum(len(c) for c in continuations) / len(continuations)
    metrics["avg_length"] = avg_len
    
    # Length variance (more variance = more diversity)
    len_var = sum((len(c) - avg_len)**2 for c in continuations) / len(continuations)
    metrics["length_std"] = len_var ** 0.5
    
    return metrics


def run_creative_writing_experiment():
    """Main experiment for creative writing diversity."""
    
    print("=" * 70)
    print("Creative Writing Experiment")
    print("=" * 70)
    print("Task: Story continuation")
    print("Hypothesis: Embedding noise should produce diverse yet coherent")
    print("continuations, potentially better than high temperature.")
    print("=" * 70)
    
    # Load model
    print("\n[1/2] Loading model...")
    model, tokenizer = load_model(MODEL_NAME, load_in_8bit=LOAD_IN_8BIT)
    
    # Methods to compare
    methods = ["greedy", "temperature", "high_temp"]
    for sigma in SIGMA_VALUES:
        methods.append(f"noise_{sigma}")
    
    # Results storage
    all_results = []
    
    print(f"\n[2/2] Generating continuations...")
    print(f"Prompts: {len(STORY_PROMPTS)}, Samples per method: {N_SAMPLES}")
    
    for story in STORY_PROMPTS:
        print(f"\n--- Story: {story['title']} ---")
        
        for method in methods:
            print(f"  {method}...", end=" ", flush=True)
            
            if method.startswith("noise_"):
                sigma = float(method.split("_")[1])
                noise_injector = EmbeddingNoiseInjector(
                    sigma_scale=sigma,
                    noise_scope="per_token",
                )
                noise_injector.attach_to_model(model)
                
                continuations = generate_continuations(
                    model, tokenizer, story["prompt"], method, N_SAMPLES,
                    noise_injector=noise_injector
                )
                
                noise_injector.detach()
            else:
                continuations = generate_continuations(
                    model, tokenizer, story["prompt"], method, N_SAMPLES
                )
            
            metrics = compute_story_metrics(continuations)
            
            result = StoryResult(
                prompt_title=story["title"],
                method=method,
                continuations=continuations,
                metrics=metrics,
            )
            all_results.append(result)
            
            unique = metrics.get("unique_ratio", 0) * 100
            d2 = metrics.get("distinct_2", 0)
            print(f"unique={unique:.0f}%, D2={d2:.3f}")
    
    # Aggregate results by method
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS (across all prompts)")
    print("=" * 70)
    
    method_results = {}
    for method in methods:
        method_data = [r for r in all_results if r.method == method]
        
        # Average metrics across prompts
        avg_metrics = {}
        for key in ["distinct_1", "distinct_2", "distinct_3", "self_bleu", "edit_distance", "unique_ratio"]:
            values = [r.metrics.get(key, 0) for r in method_data]
            avg_metrics[key] = sum(values) / len(values) if values else 0
        
        method_results[method] = avg_metrics
    
    print(f"\n{'Method':<15} {'Unique%':<10} {'Distinct-2':<12} {'Distinct-3':<12} {'Self-BLEU':<12} {'EditDist':<10}")
    print("-" * 75)
    
    for method, metrics in method_results.items():
        print(f"{method:<15} {metrics['unique_ratio']*100:<10.1f} {metrics['distinct_2']:<12.3f} {metrics['distinct_3']:<12.3f} {metrics['self_bleu']:<12.3f} {metrics['edit_distance']:<10.3f}")
    
    # Show example continuations
    print("\n" + "=" * 70)
    print("EXAMPLE CONTINUATIONS (first prompt)")
    print("=" * 70)
    
    first_prompt_results = [r for r in all_results if r.prompt_title == STORY_PROMPTS[0]["title"]]
    
    for result in first_prompt_results[:4]:  # Show first 4 methods
        print(f"\n--- {result.method} ---")
        if result.continuations:
            # Show first 2 continuations
            for i, cont in enumerate(result.continuations[:2]):
                preview = cont[:200].replace('\n', ' ')
                print(f"  [{i+1}] {preview}...")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Find best method for each metric
    best_diversity = max(method_results.items(), key=lambda x: x[1]["distinct_2"])
    best_unique = max(method_results.items(), key=lambda x: x[1]["unique_ratio"])
    lowest_self_bleu = min(method_results.items(), key=lambda x: x[1]["self_bleu"])
    
    print(f"\nBest Distinct-2: {best_diversity[0]} ({best_diversity[1]['distinct_2']:.3f})")
    print(f"Best Unique%: {best_unique[0]} ({best_unique[1]['unique_ratio']*100:.1f}%)")
    print(f"Lowest Self-BLEU: {lowest_self_bleu[0]} ({lowest_self_bleu[1]['self_bleu']:.3f})")
    
    # Compare noise vs temperature
    temp_d2 = method_results.get("temperature", {}).get("distinct_2", 0)
    noise_results = [(k, v["distinct_2"]) for k, v in method_results.items() if k.startswith("noise_")]
    
    if noise_results:
        best_noise = max(noise_results, key=lambda x: x[1])
        print(f"\nBest Noise ({best_noise[0]}): D2={best_noise[1]:.3f}")
        print(f"Temperature: D2={temp_d2:.3f}")
        
        if best_noise[1] > temp_d2:
            improvement = (best_noise[1] - temp_d2) / temp_d2 * 100
            print(f"→ Noise achieves {improvement:.1f}% better diversity!")
        else:
            print("→ Temperature achieves better diversity")
    
    print("\n" + "=" * 70)
    print("Experiment complete!")
    
    return all_results


if __name__ == "__main__":
    run_creative_writing_experiment()

