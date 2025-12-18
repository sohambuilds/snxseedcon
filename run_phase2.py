"""
Phase 2 Experiment Runner

Main script for running rigorous code generation experiments with:
- Multiple models (DeepSeek-Coder-6.7B, Llama 3.1 8B)
- Multiple methods (greedy, temperature, nucleus, embedding noise)
- Full HumanEval dataset (164 problems)
- Checkpointing for resume capability
- JSON output for human review
- Comprehensive metrics (surface + semantic diversity)

Usage:
    # Full experiment
    python run_phase2.py
    
    # Resume from checkpoint
    python run_phase2.py --resume
    
    # Quick pilot (subset)
    python run_phase2.py --pilot
    
    # Specific model only
    python run_phase2.py --model deepseek
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict
from tqdm import tqdm

import torch

# Local imports
from phase2_config import (
    MODELS, MODEL_ORDER, METHODS, METHODS_TO_RUN,
    DATASET_CONFIG, METRICS_CONFIG, OUTPUT_CONFIG,
    ModelConfig, MethodConfig, get_pilot_config, estimate_compute,
)
from src.model_loader import load_model
from src.embedding_noise import EmbeddingNoiseInjector
from src.humaneval_loader import (
    load_humaneval, format_prompt_for_model, 
    extract_function_code, check_solution, HumanEvalProblem
)
from src.generation import generate_k_solutions
from src.metrics import distinct_n, self_bleu, mean_edit_distance, pass_at_k


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

class CheckpointManager:
    """Manages experiment checkpoints for resume capability."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "checkpoint.json"
        
    def save(self, state: Dict) -> None:
        """Save checkpoint state."""
        state["timestamp"] = datetime.now().isoformat()
        with open(self.checkpoint_file, "w") as f:
            json.dump(state, f, indent=2)
        print(f"  💾 Checkpoint saved: {self.checkpoint_file}")
        
    def load(self) -> Optional[Dict]:
        """Load checkpoint if exists."""
        if not self.checkpoint_file.exists():
            return None
        with open(self.checkpoint_file, "r") as f:
            return json.load(f)
    
    def get_completed(self) -> set:
        """Get set of completed (model, method, problem_id) tuples."""
        state = self.load()
        if state is None:
            return set()
        return set(tuple(x) for x in state.get("completed", []))
    
    def mark_completed(self, model: str, method: str, problem_id: str) -> None:
        """Mark a specific combination as completed."""
        state = self.load() or {"completed": []}
        state["completed"].append([model, method, problem_id])
        self.save(state)
        
    def clear(self) -> None:
        """Clear checkpoint (start fresh)."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            print("Checkpoint cleared.")


# =============================================================================
# RESULTS STORAGE
# =============================================================================

class ResultsStorage:
    """Manages storage of generated samples and results."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.samples_file = self.output_dir / "generated_samples.json"
        self.results_file = self.output_dir / "experiment_results.json"
        
        # Initialize or load existing data
        self.samples = self._load_or_init(self.samples_file, {"samples": {}})
        self.results = self._load_or_init(self.results_file, {"results": {}})
        
    def _load_or_init(self, path: Path, default: Dict) -> Dict:
        """Load existing file or initialize with default."""
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return default
    
    def add_samples(
        self, 
        model: str, 
        method: str, 
        problem_id: str,
        prompt: str,
        generated: List[str],
        results: List[bool],
        metadata: Optional[Dict] = None,
    ) -> None:
        """Add generated samples for a problem."""
        key = f"{model}|{method}|{problem_id}"
        
        self.samples["samples"][key] = {
            "model": model,
            "method": method,
            "problem_id": problem_id,
            "prompt": prompt,
            "generated": generated,
            "passed": results,
            "n_passed": sum(results),
            "n_total": len(results),
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        
    def add_metrics(
        self,
        model: str,
        method: str, 
        problem_id: str,
        metrics: Dict,
    ) -> None:
        """Add computed metrics for a problem."""
        if model not in self.results["results"]:
            self.results["results"][model] = {}
        if method not in self.results["results"][model]:
            self.results["results"][model][method] = {}
            
        self.results["results"][model][method][problem_id] = metrics
        
    def save(self) -> None:
        """Save all data to disk."""
        with open(self.samples_file, "w") as f:
            json.dump(self.samples, f, indent=2)
        with open(self.results_file, "w") as f:
            json.dump(self.results, f, indent=2)
            
    def get_samples_for_review(self, max_per_method: int = 50) -> Dict:
        """Get subset of samples for human review."""
        review_samples = {}
        
        for key, data in self.samples["samples"].items():
            model = data["model"]
            method = data["method"]
            
            review_key = f"{model}|{method}"
            if review_key not in review_samples:
                review_samples[review_key] = []
            
            if len(review_samples[review_key]) < max_per_method:
                # Include a mix of passed and failed samples
                for i, (gen, passed) in enumerate(zip(data["generated"][:5], data["passed"][:5])):
                    review_samples[review_key].append({
                        "problem_id": data["problem_id"],
                        "sample_idx": i,
                        "generated": gen[:1000],  # Truncate for readability
                        "passed": passed,
                    })
                    
        return review_samples


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class Phase2ExperimentRunner:
    """Main experiment runner with checkpointing."""
    
    def __init__(
        self,
        models: List[str],
        methods: List[str],
        n_problems: Optional[int] = None,
        n_samples: int = 10,
        output_dir: Optional[Path] = None,
        resume: bool = False,
    ):
        self.model_names = models
        self.method_names = methods
        self.n_problems = n_problems
        self.n_samples = n_samples
        
        # Setup output and checkpointing
        output_dir = output_dir or OUTPUT_CONFIG.output_dir
        self.storage = ResultsStorage(output_dir)
        self.checkpoint = CheckpointManager(output_dir / "checkpoints")
        
        if not resume:
            self.checkpoint.clear()
            
        self.completed = self.checkpoint.get_completed()
        
        # Will be loaded on demand
        self.model = None
        self.tokenizer = None
        self.noise_injector = None
        self.current_model_name = None
        
    def load_model(self, model_key: str) -> None:
        """Load a model (or skip if already loaded)."""
        if self.current_model_name == model_key:
            return
            
        # Unload previous model
        if self.model is not None:
            del self.model
            del self.tokenizer
            if self.noise_injector:
                self.noise_injector.detach()
                del self.noise_injector
            torch.cuda.empty_cache()
            
        # Load new model
        model_config = MODELS[model_key]
        print(f"\n{'='*60}")
        print(f"Loading model: {model_config.name}")
        print(f"{'='*60}")
        
        self.model, self.tokenizer = load_model(
            model_name=model_config.name,
            load_in_8bit=model_config.load_in_8bit,
        )
        
        # Setup noise injector
        self.noise_injector = EmbeddingNoiseInjector()
        self.noise_injector.attach_to_model(self.model)
        
        self.current_model_name = model_key
        self.current_model_type = model_config.model_type
        
    def generate_for_problem(
        self,
        problem: HumanEvalProblem,
        method_key: str,
    ) -> tuple[List[str], List[bool]]:
        """Generate samples for a single problem with a single method."""
        method_config = METHODS[method_key]
        
        # Format prompt for current model
        prompt = format_prompt_for_model(problem, self.current_model_type)
        
        # Configure noise injector if needed
        if method_config.method_type == "embed_noise":
            sigma = method_config.params.get("sigma_scale", 0.002)
            self.noise_injector.sigma_scale = sigma
        
        # Build method kwargs
        method_kwargs = {"max_new_tokens": DATASET_CONFIG.max_new_tokens}
        method_kwargs.update(method_config.params)
        
        # Generate samples
        raw_outputs = generate_k_solutions(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            k=self.n_samples,
            method=method_config.method_type,
            method_kwargs=method_kwargs,
            noise_injector=self.noise_injector if method_config.method_type == "embed_noise" else None,
            show_progress=False,
        )
        
        # Extract function code and check correctness
        solutions = []
        results = []
        
        for raw in raw_outputs:
            code = extract_function_code(raw, problem.entry_point)
            solutions.append(code)
            passed = check_solution(problem, code, timeout=DATASET_CONFIG.timeout)
            results.append(passed)
            
        return solutions, results
    
    def compute_metrics(
        self,
        solutions: List[str],
        results: List[bool],
    ) -> Dict:
        """Compute metrics for a set of solutions."""
        metrics = {
            "n_samples": len(solutions),
            "n_passed": sum(results),
            "pass_rate": sum(results) / len(results) if results else 0,
        }
        
        # Pass@K estimates
        for k in METRICS_CONFIG.pass_at_k:
            if k <= len(results):
                metrics[f"pass@{k}"] = pass_at_k(len(results), sum(results), k)
        
        # Surface diversity metrics
        if len(solutions) > 1:
            for n in METRICS_CONFIG.distinct_n:
                metrics[f"distinct_{n}"] = distinct_n(solutions, n)
                
            if METRICS_CONFIG.compute_self_bleu:
                metrics["self_bleu"] = self_bleu(solutions)
                
            if METRICS_CONFIG.compute_edit_distance:
                metrics["mean_edit_distance"] = mean_edit_distance(solutions)
        
        return metrics
    
    def run(self) -> None:
        """Run the full experiment."""
        # Load problems
        problems = load_humaneval(self.n_problems)
        n_problems = len(problems)
        
        # Print experiment summary
        print("\n" + "="*60)
        print("PHASE 2 EXPERIMENT")
        print("="*60)
        print(f"Models: {self.model_names}")
        print(f"Methods: {self.method_names}")
        print(f"Problems: {n_problems}")
        print(f"Samples per problem: {self.n_samples}")
        print(f"Already completed: {len(self.completed)} combinations")
        
        estimates = estimate_compute(
            n_models=len(self.model_names),
            n_methods=len(self.method_names),
            n_problems=n_problems,
            n_samples=self.n_samples,
        )
        print(f"Estimated time: {estimates['estimated_hours']} hours")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        # Iterate through models
        for model_key in self.model_names:
            self.load_model(model_key)
            model_short = MODELS[model_key].short_name
            
            # Iterate through methods
            for method_key in self.method_names:
                method_name = METHODS[method_key].name
                
                print(f"\n📊 {model_short} | {method_name}")
                print("-" * 40)
                
                # Iterate through problems
                pbar = tqdm(problems, desc=f"  {method_name}", leave=True)
                
                problems_completed = 0
                for problem in pbar:
                    # Check if already completed
                    combo = (model_key, method_key, problem.task_id)
                    if combo in self.completed:
                        problems_completed += 1
                        continue
                    
                    # Generate solutions
                    try:
                        solutions, results = self.generate_for_problem(problem, method_key)
                        
                        # Compute metrics
                        metrics = self.compute_metrics(solutions, results)
                        
                        # Store results
                        prompt = format_prompt_for_model(problem, self.current_model_type)
                        self.storage.add_samples(
                            model=model_key,
                            method=method_key,
                            problem_id=problem.task_id,
                            prompt=prompt,
                            generated=solutions,
                            results=results,
                            metadata={"method_params": asdict(METHODS[method_key])},
                        )
                        self.storage.add_metrics(model_key, method_key, problem.task_id, metrics)
                        
                        # Mark completed
                        self.completed.add(combo)
                        problems_completed += 1
                        
                        # Update progress bar
                        pbar.set_postfix({
                            "pass": f"{metrics.get('pass_rate', 0):.0%}",
                            "done": problems_completed,
                        })
                        
                    except Exception as e:
                        print(f"\n  ⚠️ Error on {problem.task_id}: {e}")
                        continue
                    
                    # Checkpoint periodically
                    if problems_completed % OUTPUT_CONFIG.checkpoint_every_n_problems == 0:
                        self._save_checkpoint()
                
                # Save after each method
                self._save_checkpoint()
                
        # Final save
        self._save_checkpoint()
        
        elapsed = time.time() - start_time
        print(f"\n✅ Experiment completed in {elapsed/3600:.1f} hours")
        print(f"   Results saved to: {self.storage.output_dir}")
        
    def _save_checkpoint(self) -> None:
        """Save current progress."""
        self.storage.save()
        self.checkpoint.save({
            "completed": [list(x) for x in self.completed],
            "n_completed": len(self.completed),
        })


# =============================================================================
# POST-EXPERIMENT ANALYSIS
# =============================================================================

def run_semantic_analysis(results_dir: Path) -> None:
    """Run semantic diversity analysis on stored samples."""
    from src.semantic_diversity import SemanticDiversityCalculator, compute_functional_diversity
    
    print("\n" + "="*60)
    print("SEMANTIC DIVERSITY ANALYSIS")
    print("="*60)
    
    samples_file = results_dir / "generated_samples.json"
    if not samples_file.exists():
        print("No samples file found!")
        return
        
    with open(samples_file, "r") as f:
        data = json.load(f)
    
    calculator = SemanticDiversityCalculator()
    semantic_results = {}
    
    # Group samples by model and method
    grouped = {}
    for key, sample_data in data["samples"].items():
        model = sample_data["model"]
        method = sample_data["method"]
        group_key = f"{model}|{method}"
        
        if group_key not in grouped:
            grouped[group_key] = []
        grouped[group_key].extend(sample_data["generated"])
    
    # Compute semantic diversity for each group
    for group_key, samples in tqdm(grouped.items(), desc="Computing semantic diversity"):
        # Sample subset for efficiency (CodeBERT is slow)
        sample_subset = samples[:100] if len(samples) > 100 else samples
        
        semantic = calculator.calculate_diversity(sample_subset)
        functional = compute_functional_diversity(sample_subset)
        
        semantic_results[group_key] = {
            **semantic,
            **functional,
        }
    
    # Save results
    output_file = results_dir / "semantic_diversity.json"
    with open(output_file, "w") as f:
        json.dump(semantic_results, f, indent=2)
    
    print(f"\nSemantic diversity results saved to: {output_file}")
    
    # Print summary
    print("\n📊 Semantic Diversity Summary:")
    print("-" * 60)
    for key, metrics in semantic_results.items():
        model, method = key.split("|")
        print(f"{model} | {method}:")
        print(f"   Mean cosine distance: {metrics.get('mean_cosine_distance', 0):.3f}")
        print(f"   Semantic clusters: {metrics.get('n_semantic_clusters', 0)}")
        print(f"   Pattern diversity: {metrics.get('pattern_diversity', 0):.3f}")


def run_statistical_analysis(results_dir: Path) -> None:
    """Run statistical analysis on results."""
    from src.statistics import StatisticalAnalyzer, format_results_table
    
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    results_file = results_dir / "experiment_results.json"
    if not results_file.exists():
        print("No results file found!")
        return
        
    with open(results_file, "r") as f:
        data = json.load(f)
    
    analyzer = StatisticalAnalyzer(
        n_bootstrap=METRICS_CONFIG.n_bootstrap,
        confidence_level=METRICS_CONFIG.confidence_level,
    )
    
    # Analyze each model
    for model_name, model_results in data["results"].items():
        print(f"\n📊 Model: {model_name}")
        print("-" * 40)
        
        # Collect pass/fail results per method
        method_results = {}
        for method_name, problems in model_results.items():
            method_results[method_name] = []
            for problem_id, metrics in problems.items():
                # Reconstruct pass/fail list from metrics
                n = metrics["n_samples"]
                c = metrics["n_passed"]
                # Create boolean list (approximation)
                results = [True] * c + [False] * (n - c)
                method_results[method_name].append(results)
        
        # Run comparison
        comparison = analyzer.compare_methods(
            method_results,
            k_values=METRICS_CONFIG.pass_at_k,
            baseline_method=METRICS_CONFIG.baseline_method,
        )
        
        # Print table
        print(format_results_table(comparison, METRICS_CONFIG.pass_at_k))
        
        # Save analysis
        output_file = results_dir / f"statistical_analysis_{model_name}.json"
        with open(output_file, "w") as f:
            json.dump(comparison, f, indent=2)


def generate_summary(results_dir: Path) -> None:
    """Generate human-readable summary."""
    summary_lines = [
        "# Phase 2 Experiment Summary",
        f"\nGenerated: {datetime.now().isoformat()}",
        "\n## Configuration",
    ]
    
    # Add configuration details
    summary_lines.append(f"- Models: {MODEL_ORDER}")
    summary_lines.append(f"- Methods: {METHODS_TO_RUN}")
    summary_lines.append(f"- Samples per problem: {DATASET_CONFIG.n_samples}")
    
    # Load and summarize results
    results_file = results_dir / "experiment_results.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            data = json.load(f)
            
        summary_lines.append("\n## Results Overview")
        
        for model_name, model_results in data["results"].items():
            summary_lines.append(f"\n### {model_name}")
            
            for method_name, problems in model_results.items():
                if not problems:
                    continue
                    
                # Aggregate metrics
                pass_rates = [m["pass_rate"] for m in problems.values()]
                avg_pass = sum(pass_rates) / len(pass_rates)
                
                summary_lines.append(f"- **{method_name}**: {avg_pass:.1%} avg pass rate ({len(problems)} problems)")
    
    # Write summary
    summary_file = results_dir / "summary.md"
    with open(summary_file, "w") as f:
        f.write("\n".join(summary_lines))
    
    print(f"\nSummary written to: {summary_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Experiment Runner")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--pilot", action="store_true", help="Run quick pilot experiment")
    parser.add_argument("--model", type=str, help="Run specific model only (deepseek or llama)")
    parser.add_argument("--analyze", action="store_true", help="Run analysis on existing results")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    
    args = parser.parse_args()
    
    # Determine configuration
    if args.pilot:
        config = get_pilot_config()
        models = config["models"]
        methods = config["methods"]
        n_problems = config["n_problems"]
        n_samples = config["n_samples"]
    else:
        models = MODEL_ORDER
        methods = METHODS_TO_RUN
        n_problems = DATASET_CONFIG.n_problems
        n_samples = DATASET_CONFIG.n_samples
    
    # Override model if specified
    if args.model:
        if args.model not in MODELS:
            print(f"Unknown model: {args.model}")
            print(f"Available: {list(MODELS.keys())}")
            return
        models = [args.model]
    
    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_CONFIG.output_dir
    
    # Analysis only mode
    if args.analyze:
        run_semantic_analysis(output_dir)
        run_statistical_analysis(output_dir)
        generate_summary(output_dir)
        return
    
    # Run experiment
    runner = Phase2ExperimentRunner(
        models=models,
        methods=methods,
        n_problems=n_problems,
        n_samples=n_samples,
        output_dir=output_dir,
        resume=args.resume,
    )
    
    runner.run()
    
    # Run analysis
    print("\n" + "="*60)
    print("Running post-experiment analysis...")
    print("="*60)
    
    run_statistical_analysis(output_dir)
    generate_summary(output_dir)
    
    print("\n✅ All done!")


if __name__ == "__main__":
    main()



