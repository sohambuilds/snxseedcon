# Sanity Check Experiment

This directory contains the implementation of the sanity check experiment described in `sanitycheckplan.md`.

## Overview

The experiment tests whether embedding-level noise can induce creative diversity in verifiable generation tasks, specifically competitive programming problem generation.

### Three Experimental Conditions

1. **Condition A**: Deterministic baseline (temperature=0, no noise)
2. **Condition B**: Temperature sampling (temperature>0, no noise)  
3. **Condition C**: Embedding noise (temperature=0, with noise)

## Files

- `sanitycheckplan.md` - Complete experimental protocol (read this first!)
- `config.py` - Experiment configuration (edit this to change settings)
- `prompts.py` - 10 distinct prompts for problem generation
- `inference.py` - Core inference engine implementing the three conditions
- `run_experiment.py` - Main script to run the experiment
- `outputs/` - Generated outputs (created after running)

## Quick Start

### 1. Configure the Experiment

Edit `config.py` to set:
- Model name (default: deepseek-coder-6.7b-base)
- Number of samples per condition (default: K=10)
- Temperature for condition B (default: 0.8)
- Sigma scale for condition C (default: 0.01)
- Other generation settings

### 2. Validate Setup (Recommended)

Run a quick validation before the full experiment:

```bash
# From project root
python sanitycheck/validate_setup.py
```

This will:
- Check GPU availability and memory
- Load the model
- Test all three conditions with 1 sample each
- Verify the single-shot noise behavior
- Estimate total outputs

### 3. Run the Experiment

```bash
# From project root
python sanitycheck/run_experiment.py
```

This will:
1. Load the specified model
2. Run all three conditions on all 10 prompts
3. Save outputs to `sanitycheck/outputs/`

Expected time: ~30-60 minutes depending on model size and GPU.

Expected outputs: 
- Condition A: 10 outputs (1 per prompt)
- Condition B: 100 outputs (10 per prompt)
- Condition C: 100 outputs (10 per prompt)
- **Total: 210 outputs**

### 3. Inspect Outputs

After generation completes:

```bash
# View individual outputs
ls sanitycheck/outputs/

# Example: View Condition A, prompt 0
cat sanitycheck/outputs/A_0_0.txt

# Example: Compare same prompt across conditions
cat sanitycheck/outputs/A_0_0.txt   # Deterministic
cat sanitycheck/outputs/B_0_5.txt   # Temperature sample 5
cat sanitycheck/outputs/C_0_5.txt   # Noise sample 5
```

All outputs are also available in `outputs/all_results.json` for programmatic analysis.

### 4. Plan Evaluation

After inspecting outputs, design evaluation metrics:
- Validity rate (are problems well-formed?)
- Algorithmic diversity (how many distinct algorithm classes?)
- Constraint structure diversity (do constraints vary meaningfully?)

See Section 8 of `sanitycheckplan.md` for metric definitions.

## Minimal Evaluation (Implemented)

This repo now includes a deliberately simple evaluation workflow that matches the protocol:
- **Validity@k** (human-labeled, binary) per output (§8.1)
- **AlgDiversity@k** (human-labeled algorithm class) among *valid* outputs (§8.2)
- **SigDiversity@k** (heuristic constraint signature) among *valid* outputs (§8.3)

### 1) Create an annotation sheet (CSV)

```bash
python sanitycheck/make_annotations.py
```

This reads `sanitycheck/outputs/all_results.json` and writes:
- `sanitycheck/outputs/annotations.csv`

By default it includes **only Conditions B and C** (the plan’s comparison in §9). Use `--include-a` if you want A too.

### 2) Fill the labels (manual)

Open `sanitycheck/outputs/annotations.csv` and fill:
- `valid`: `y` / `n`
- `alg_class`: one of `greedy, dp, graph, math, brute, ds, other`
- optional: `sig_override` if the auto signature is clearly wrong

### 3) Compute metrics + summary table

```bash
python sanitycheck/compute_basic_metrics.py
```

This prints an average-over-prompts summary and writes:
- `sanitycheck/outputs/metrics_summary.json`

## Configuration Options

### Model Selection

Choose a 7B-13B parameter model in `config.py`:

```python
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-base"  # Good default
# Or:
# MODEL_NAME = "codellama/CodeLlama-7b-hf"
# MODEL_NAME = "codellama/CodeLlama-13b-hf"
```

### Sample Size

Adjust `K_SAMPLES` for more/fewer outputs:

```python
K_SAMPLES = 10  # Default - good for initial run
# K_SAMPLES = 20  # More samples for stronger signal
# K_SAMPLES = 5   # Faster testing
```

### Temperature Tuning

If outputs are too similar or too chaotic in Condition B:

```python
TEMPERATURE = 0.8  # Default
# TEMPERATURE = 1.0  # More variation
# TEMPERATURE = 0.6  # Less variation
```

### Noise Strength

If Condition C outputs are invalid or too similar:

```python
SIGMA_SCALE = 0.01  # Default - good starting point
# SIGMA_SCALE = 0.005  # Less noise
# SIGMA_SCALE = 0.02   # More noise
```

## Output Format

Each output file contains:

```
# Condition A/B/C
# Prompt X, Sample Y
# Timestamp: ...
# Seed: ... (Condition C only)

================================================================================
PROMPT:
[Full prompt text]

================================================================================
GENERATED OUTPUT:
[Model generation]
```

## Next Steps

1. **Manual Inspection**: Read through outputs to get intuition
2. **Design Metrics**: Plan evaluation based on what you observe
3. **Implement Evaluation**: Create evaluation scripts
4. **Analyze Results**: Compare conditions systematically

## Troubleshooting

**Out of memory?**
- Set `LOAD_IN_8BIT = True` in config.py
- Use smaller model
- Reduce `MAX_NEW_TOKENS`

**Generations too short/long?**
- Adjust `MAX_NEW_TOKENS` in config.py

**Model downloads fail?**
- Ensure you have HuggingFace access token if needed
- Check internet connection
- Try different model

**Outputs are garbage?**
- Try different `SIGMA_SCALE` (lower = more conservative)
- Try different model (some are more robust to noise)
- Check if base model vs instruct model matters

## Directory Structure

```
sanitycheck/
├── __init__.py
├── README.md              # This file
├── sanitycheckplan.md     # Full experimental protocol
├── config.py              # Configuration
├── prompts.py             # Prompt set
├── inference.py           # Core inference engine
├── run_experiment.py      # Main script
└── outputs/               # Generated outputs
    ├── A_0_0.txt          # Condition A outputs
    ├── B_0_0.txt          # Condition B outputs
    ├── C_0_0.txt          # Condition C outputs
    └── all_results.json   # Consolidated JSON
```

## Citation

If you use this experiment setup, please cite the original protocol document.

