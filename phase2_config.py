"""
Phase 2 Experimental Configuration

This configuration defines a rigorous experimental setup addressing:
1. Larger sample sizes (n=10 for pilot, scalable to n=25+)
2. Semantic diversity metrics (CodeBERT)
3. Additional baselines (nucleus sampling)
4. Statistical rigor (bootstrap CIs, Wilcoxon tests)
5. Checkpointing for resume capability
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str                    # HuggingFace model ID
    short_name: str              # Short name for outputs
    model_type: str              # Prompt format type
    load_in_8bit: bool = True    # Quantization
    trust_remote_code: bool = True


MODELS = {
    "deepseek": ModelConfig(
        name="deepseek-ai/deepseek-coder-6.7b-instruct",
        short_name="deepseek-6.7b",
        model_type="deepseek",
        load_in_8bit=True,
    ),
    "llama": ModelConfig(
        name="meta-llama/Llama-3.1-8B-Instruct",
        short_name="llama-3.1-8b",
        model_type="llama3",
        load_in_8bit=True,
    ),
}

# Order of model execution (sequential)
MODEL_ORDER = ["deepseek", "llama"]


# =============================================================================
# GENERATION METHODS
# =============================================================================

@dataclass
class MethodConfig:
    """Configuration for a generation method."""
    name: str
    method_type: str  # "greedy", "temperature", "nucleus", "embed_noise"
    params: Dict = field(default_factory=dict)


METHODS = {
    # Baseline
    "greedy": MethodConfig(
        name="greedy",
        method_type="greedy",
        params={},
    ),
    
    # Temperature sampling (3 values)
    "temp_0.6": MethodConfig(
        name="temp_0.6",
        method_type="temperature",
        params={"temperature": 0.6},
    ),
    "temp_0.8": MethodConfig(
        name="temp_0.8",
        method_type="temperature",
        params={"temperature": 0.8},
    ),
    "temp_1.0": MethodConfig(
        name="temp_1.0",
        method_type="temperature",
        params={"temperature": 1.0},
    ),
    
    # Nucleus sampling (2 values)
    "nucleus_0.9": MethodConfig(
        name="nucleus_0.9",
        method_type="nucleus",
        params={"top_p": 0.9},
    ),
    "nucleus_0.95": MethodConfig(
        name="nucleus_0.95",
        method_type="nucleus",
        params={"top_p": 0.95},
    ),
    
    # Embedding noise (4 sigma values)
    "noise_0.001": MethodConfig(
        name="noise_0.001",
        method_type="embed_noise",
        params={"sigma_scale": 0.001},
    ),
    "noise_0.002": MethodConfig(
        name="noise_0.002",
        method_type="embed_noise",
        params={"sigma_scale": 0.002},
    ),
    "noise_0.005": MethodConfig(
        name="noise_0.005",
        method_type="embed_noise",
        params={"sigma_scale": 0.005},
    ),
    "noise_0.01": MethodConfig(
        name="noise_0.01",
        method_type="embed_noise",
        params={"sigma_scale": 0.01},
    ),
}

# Methods to run (can modify to run subset)
METHODS_TO_RUN = [
    "greedy",
    "temp_0.6", "temp_0.8", "temp_1.0",
    "nucleus_0.9", "nucleus_0.95",
    "noise_0.001", "noise_0.002", "noise_0.005", "noise_0.01",
]


# =============================================================================
# DATASET & SAMPLING CONFIGURATION
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for dataset and sampling."""
    name: str = "humaneval"
    n_problems: Optional[int] = None  # None = all 164 problems
    n_samples: int = 10               # Samples per problem per method
    max_new_tokens: int = 512         # Max generation length
    timeout: float = 5.0              # Execution timeout (seconds)


# Week 1: Pilot run (n=10)
DATASET_CONFIG = DatasetConfig(
    name="humaneval",
    n_problems=None,      # All 164 problems
    n_samples=10,         # 10 samples per problem (pilot)
    max_new_tokens=512,
    timeout=5.0,
)

# Week 2: Main experiment (uncomment to use)
# DATASET_CONFIG = DatasetConfig(
#     name="humaneval",
#     n_problems=None,
#     n_samples=25,        # 25 samples (reliable Pass@10)
#     max_new_tokens=512,
#     timeout=5.0,
# )


# =============================================================================
# METRICS CONFIGURATION
# =============================================================================

@dataclass
class MetricsConfig:
    """Configuration for metrics to compute."""
    # Correctness metrics
    pass_at_k: List[int] = field(default_factory=lambda: [1, 5, 10])
    
    # Surface diversity metrics  
    distinct_n: List[int] = field(default_factory=lambda: [2, 3])
    compute_self_bleu: bool = True
    compute_edit_distance: bool = True
    
    # Semantic diversity metrics
    compute_semantic_diversity: bool = True  # CodeBERT embeddings
    compute_functional_diversity: bool = True  # AST pattern analysis
    
    # Statistical analysis
    n_bootstrap: int = 10000
    confidence_level: float = 0.95
    baseline_method: str = "temp_0.8"  # Compare other methods against this


METRICS_CONFIG = MetricsConfig()


# =============================================================================
# OUTPUT & CHECKPOINTING
# =============================================================================

@dataclass
class OutputConfig:
    """Configuration for outputs and checkpointing."""
    output_dir: Path = field(default_factory=lambda: Path("results/phase2"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("results/phase2/checkpoints"))
    
    # Output files
    samples_file: str = "generated_samples.json"      # All generated samples
    results_file: str = "experiment_results.json"     # Aggregated results
    summary_file: str = "summary.md"                  # Human-readable summary
    
    # Checkpoint frequency
    checkpoint_every_n_problems: int = 10  # Save progress every N problems
    
    # Sample storage for human review
    store_samples: bool = True
    max_samples_per_method: int = 50  # Store up to N samples per method for review


OUTPUT_CONFIG = OutputConfig()


# =============================================================================
# EXPERIMENT PRESETS
# =============================================================================

def get_pilot_config():
    """Get configuration for quick pilot run (Week 1)."""
    return {
        "models": ["deepseek"],  # Start with one model
        "methods": ["greedy", "temp_0.8", "nucleus_0.95", "noise_0.002"],
        "n_problems": 20,        # Subset for quick validation
        "n_samples": 10,
    }


def get_full_config():
    """Get configuration for full experiment (Week 2)."""
    return {
        "models": MODEL_ORDER,
        "methods": METHODS_TO_RUN,
        "n_problems": None,      # All 164
        "n_samples": 25,
    }


# =============================================================================
# COMPUTE ESTIMATES
# =============================================================================

def estimate_compute(
    n_models: int = 2,
    n_methods: int = 10,
    n_problems: int = 164,
    n_samples: int = 10,
    seconds_per_generation: float = 3.0,
) -> Dict:
    """
    Estimate compute requirements.
    
    Returns dict with estimates for time, storage, etc.
    """
    total_generations = n_models * n_methods * n_problems * n_samples
    total_seconds = total_generations * seconds_per_generation
    total_hours = total_seconds / 3600
    
    return {
        "total_generations": total_generations,
        "estimated_hours": round(total_hours, 1),
        "estimated_days": round(total_hours / 24, 1),
        "per_model_hours": round(total_hours / n_models, 1),
        "storage_mb": round(total_generations * 0.002, 1),  # ~2KB per sample
    }


if __name__ == "__main__":
    # Print configuration summary
    print("=" * 60)
    print("PHASE 2 EXPERIMENTAL CONFIGURATION")
    print("=" * 60)
    
    print(f"\n📊 Models: {len(MODELS)}")
    for key, model in MODELS.items():
        print(f"   - {model.short_name}: {model.name}")
    
    print(f"\n🔧 Methods: {len(METHODS_TO_RUN)}")
    for method_name in METHODS_TO_RUN:
        method = METHODS[method_name]
        print(f"   - {method.name}: {method.method_type} {method.params}")
    
    print(f"\n📁 Dataset: {DATASET_CONFIG.name}")
    print(f"   - Problems: {DATASET_CONFIG.n_problems or 'all (164)'}")
    print(f"   - Samples per problem: {DATASET_CONFIG.n_samples}")
    
    print(f"\n📈 Metrics:")
    print(f"   - Pass@K: {METRICS_CONFIG.pass_at_k}")
    print(f"   - Distinct-N: {METRICS_CONFIG.distinct_n}")
    print(f"   - Semantic diversity: {METRICS_CONFIG.compute_semantic_diversity}")
    
    print(f"\n💾 Output: {OUTPUT_CONFIG.output_dir}")
    
    # Compute estimates
    estimates = estimate_compute(
        n_models=len(MODEL_ORDER),
        n_methods=len(METHODS_TO_RUN),
        n_problems=DATASET_CONFIG.n_problems or 164,
        n_samples=DATASET_CONFIG.n_samples,
    )
    
    print(f"\n⏱️ Compute Estimates:")
    print(f"   - Total generations: {estimates['total_generations']:,}")
    print(f"   - Estimated time: {estimates['estimated_hours']} hours ({estimates['estimated_days']} days)")
    print(f"   - Storage: ~{estimates['storage_mb']} MB")


