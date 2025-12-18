# Sanity Check Experiment

This directory contains the implementation of the sanity check experiment described in `sanitycheckplan.md`.

## Overview

The experiment tests whether embedding-level noise can induce creative diversity in verifiable generation tasks, specifically competitive programming problem generation.

### Three Experimental Conditions

1. **Condition A**: Deterministic baseline (temperature=0, no noise)
2. **Condition B**: Temperature sampling (temperature>0, no noise)  
3. **Condition C**: Embedding noise (temperature=0, with noise)

## Files

**Core**: `config.py`, `prompts.py`, `inference.py`, `run_experiment.py`  
**Evaluation**: `groq_judge.py`, `groq_judge2.py`, `compute_agreement.py`  
**Utilities**: `validate_setup.py`, `regenerate_empty.py`, `inspect_outputs.py`  
**Protocol**: `sanitycheckplan.md`

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

## Optional: LLM Judge (Groq / Gemini)

Use an LLM as a supplementary signal (not the primary metric per plan, but useful for quick evaluation at scale).

Scores each problem on:
- **Creativity/Uniqueness** (1-10): How novel is the problem idea?
- **Validity/Solvability** (1-10 + PASS/FAIL): Is it well-formed and solvable?

### Option A: Groq API (Recommended - Fast & Free Tier)

```bash
pip install openai
export GROQ_API_KEY="your-api-key"

# Quick test
python sanitycheck/groq_judge.py --n-per-condition 10

# Full run
python sanitycheck/groq_judge.py

# Use different model
python sanitycheck/groq_judge.py --model llama-3.1-8b-instant
```

Available Groq models: `llama-3.3-70b-versatile` (default), `llama-3.1-8b-instant`, `mixtral-8x7b-32768`

### Option B: Gemini API

```bash
pip install google-genai
export GEMINI_API_KEY="your-api-key"

python sanitycheck/gemini_judge.py --n-per-condition 10
```

### Output

- `sanitycheck/outputs/groq_judgments.jsonl` (or `gemini_judgments.jsonl`)
- `sanitycheck/outputs/groq_judgments_summary.json` - aggregated B vs C comparison

---

## Experiment Results (Run 1: deepseek-coder-6.7b-base)

### Model & Settings
- **Model**: `deepseek-ai/deepseek-coder-6.7b-base`
- **Condition B**: Temperature sampling (T=0.8)
- **Condition C**: Embedding noise (σ=0.01, per_sequence)
- **Samples**: 100 per condition (10 prompts × 10 samples)

### Evaluation 1: Scored (1-10 scale)

| Metric | Condition B | Condition C | Delta |
|--------|-------------|-------------|-------|
| Creativity (mean) | 2.58 | 2.65 | **+0.07** |
| Validity (mean) | 2.67 | 3.03 | **+0.36** |
| Validity Pass Rate | 13.0% | 21.0% | **+8.0%** |

### Evaluation 2: Binary (pass/fail, lenient)

| Metric | Condition B | Condition C | Delta |
|--------|-------------|-------------|-------|
| Creative | 57.0% | 68.0% | **+11.0%** |
| Valid | 42.0% | 50.0% | **+8.0%** |
| Both (Creative & Valid) | 39.0% | 50.0% | **+11.0%** |

### Key Findings

1. **Embedding noise (C) consistently outperforms temperature sampling (B)** on both creativity and validity metrics across both evaluation approaches.

2. **Validity improves with embedding noise** — contrary to the concern that noise might degrade output quality, Condition C shows higher validity rates (+8% absolute).

3. **Creativity gains are meaningful** — +11% absolute improvement in binary creative pass rate suggests embedding noise induces genuinely different problem structures, not just surface variation.

4. **Interpretation**: ✓ Embedding noise shows higher creativity with similar or better validity — **positive signal for the hypothesis**.

---

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

## Directory Structure

```
sanitycheck/
├── config.py                 # Experiment configuration
├── prompts.py                # 10 distinct prompts
├── embedding_noise_single.py # Single-shot noise injection
├── inference.py              # Core inference engine
├── run_experiment.py         # Main experiment script
├── validate_setup.py         # Pre-flight checks
├── regenerate_empty.py       # Fix empty outputs
├── groq_judge.py             # LLM judge (scored)
├── groq_judge2.py            # LLM judge (binary)
├── compute_agreement.py      # Inter-rater agreement
├── compute_basic_metrics.py  # Manual annotation metrics
├── make_annotations.py       # Generate annotation CSV
├── inspect_outputs.py        # Output inspection utility
├── signature_utils.py        # Constraint signature extraction
├── sanitycheckplan.md        # Experimental protocol
└── outputs/                  # Generated outputs & results
```

