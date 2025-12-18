"""
Configuration for the sanity check experiment.
Following the protocol in sanitycheckplan.md
"""

# ============================================================================
# Model Configuration
# ============================================================================

# Choose one mid-scale model (7B-13B parameters)
# Options:
#   - "deepseek-ai/deepseek-coder-6.7b-base"  # ~7B, good for structured generation
#   - "deepseek-ai/deepseek-coder-6.7b-instruct"  # Instruction-tuned variant
#   - "codellama/CodeLlama-7b-hf"  # Meta's 7B code model
#   - "codellama/CodeLlama-13b-hf"  # Meta's 13B code model
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-base"

# Memory optimization
LOAD_IN_8BIT = False  # Set to True if VRAM limited
TORCH_DTYPE = "float16"  # "float16" or "bfloat16" or "float32"

# ============================================================================
# Experimental Conditions
# ============================================================================

# Number of outputs per condition
K_SAMPLES = 10  # For conditions B and C

# Condition B: Temperature Sampling
TEMPERATURE = 0.8  # Tuned for fluent but varied outputs
# Note: May need tuning per model. Start with 0.8

# Condition C: Embedding Noise
SIGMA_SCALE = 0.01  # Noise magnitude as fraction of embedding norm
NOISE_SCOPE = "per_sequence"  # Use shared noise vector for global plan shifts

# ============================================================================
# Generation Settings
# ============================================================================

MAX_NEW_TOKENS = 1024  # Enough for full competitive programming problem
MIN_NEW_TOKENS = 64  # Prevent empty / newline-only generations (applied to A/B/C equally)
GENERATION_TIMEOUT = 60  # Seconds per generation (safety)

# ============================================================================
# Prompt Configuration
# ============================================================================

# Number of distinct prompts to use
N_PROMPTS = 10

# ============================================================================
# Output Configuration
# ============================================================================

# Where to save outputs
OUTPUT_DIR = "sanitycheck/outputs"

# Output file naming
# Format: {OUTPUT_DIR}/{condition}_{prompt_idx}_{sample_idx}.txt
SAVE_INDIVIDUAL_FILES = True  # Save each output separately
SAVE_SUMMARY_JSON = True  # Also save consolidated JSON

# ============================================================================
# Runtime Settings
# ============================================================================

SHOW_PROGRESS = True  # Show progress bars
VERBOSE = True  # Print generation info
RANDOM_SEED = 42  # For reproducibility

# ============================================================================
# Hardware
# ============================================================================

# IMPORTANT (Single A6000 protocol requirement):
# Use a single GPU. `device_map="auto"` can silently offload layers to CPU, which
# makes generation look "stuck" / extremely slow on Windows.
# For single-GPU runs, force everything onto GPU 0 via an explicit map.
DEVICE_MAP = {"": 0}

