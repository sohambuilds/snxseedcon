"""
Configuration for minimal test experiments.
Edit this file to adjust parameters without modifying code.
"""

# Model configuration
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"  # Or use 33B for full scale
LOAD_IN_8BIT = True  # Set to False if you have enough VRAM
MODEL_TYPE = "deepseek"  # For prompt formatting

# Dataset configuration
N_PROBLEMS = 5  # Number of HumanEval problems to test (max 164)
K_SAMPLES = 10  # Samples per method per problem

# Embedding noise configuration
# Based on experiments:
#   per_token + σ=0.002 → 80% compile (best)
#   per_sequence + σ=0.01 → 67% compile (allows higher σ)
SIGMA_SCALE = 0.002  # Optimal for per_token noise
NOISE_SCOPE = "per_token"  # "per_token" or "per_sequence"

# Sigma values for sweep experiments (revised based on inspection)
# Original 0.1 was way too high. Sweet spot appears to be 0.001-0.01
SIGMA_SWEEP_VALUES = [0.001, 0.002, 0.005, 0.008, 0.01, 0.02]

# Methods to compare
METHODS = {
    "greedy": {},
    "temperature": {"temperature": 0.8},
    "embed_noise": {},
}

# You can add more methods:
# "temp_high": {"temperature": 1.5},
# "nucleus": {"top_p": 0.95},

# Generation configuration
MAX_NEW_TOKENS = 512  # Maximum tokens to generate per sample
TIMEOUT = 5.0  # Timeout for code execution (seconds)

# Display configuration
SHOW_PROGRESS = True  # Show progress bars during generation
SHOW_EXAMPLES = False  # Print example generated solutions

