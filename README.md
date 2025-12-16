# Inference-Time Embedding Noise for Diverse Code Generation

Research implementation testing whether injecting Gaussian noise into LLM embeddings at inference time can match or exceed the diversity benefits of temperature sampling, without any model retraining.

## Core Hypothesis

Training with random string prefixes ("seeds") + greedy decoding achieves diversity comparable to temperature sampling (Nagarajan et al., 2025). We test if this effect can be achieved **without training** by injecting noise directly into embeddings at inference time:

```
h₀ = E(x) + ε, where ε ~ N(0, σ²I)
```

## Quick Start

### Install Dependencies

```bash
uv add torch transformers accelerate bitsandbytes datasets nltk python-Levenshtein tqdm
```

### Run Minimal Test

```bash
python test_minimal.py
```

This will:
- Load DeepSeek-Coder-6.7B with int8 quantization
- Test on 5 HumanEval problems
- Generate 10 solutions per method (greedy, temperature=0.8, embedding noise)
- Report Pass@K, diversity metrics, and compilation rates

### Expected Runtime

- ~3 minutes for model loading
- ~2-3 minutes per problem (30 samples total: 10 × 3 methods)
- Total: ~15-20 minutes for full test

## Project Structure

```
src/
├── model_loader.py       # Load models with quantization
├── embedding_noise.py    # Core noise injection via PyTorch hooks
├── generation.py         # Generation harness (greedy/temp/nucleus/noise)
├── humaneval_loader.py   # Dataset loading + sandboxed execution
└── metrics.py            # Pass@K, Distinct-N, Self-BLEU, edit distance

test_minimal.py           # Quick validation script
```

## How It Works

### Embedding Noise Injection

The `EmbeddingNoiseInjector` class uses PyTorch forward hooks to intercept embeddings and add Gaussian noise:

1. **Compute σ**: Scale noise relative to mean embedding norm
   ```python
   σ = sigma_scale × E[||E(xᵢ)||₂]
   ```

2. **Inject noise**: Add to embeddings before transformer layers
   ```python
   h₀ = E(x) + ε where ε ~ N(0, σ²I)
   ```

3. **Generate**: Use greedy decoding (temperature=0) with noisy embeddings

### Key Parameters

- `sigma_scale`: Controls noise magnitude (default: 0.1)
- `noise_scope`: 
  - `"per_token"`: Independent noise per token
  - `"per_sequence"`: Same noise vector shared across positions
- `seed`: For reproducible diverse outputs

## Evaluation Metrics

### Correctness
- **Pass@K**: Probability ≥1 solution passes in K samples
- Uses unbiased estimator from Codex paper

### Diversity
- **Distinct-N**: Unique n-grams / total n-grams (higher = more diverse)
- **Self-BLEU**: Avg max BLEU between samples (lower = more diverse)
- **Edit Distance**: Mean pairwise Levenshtein distance (higher = more diverse)

### Coherence
- **Compilation Rate**: Fraction of samples that parse correctly
- Tests if noise degrades code quality

## Expected Results

### Success Scenario
EmbedNoise matches Pass@K of temperature sampling while achieving 20%+ higher Distinct-2

### Partial Success
Works for code but not math/reasoning (domain-specific contribution)

### Null Result
Produces garbage or no diversity gain (clarifies necessity of training)

## Next Steps

After validating the minimal test:

1. **Scale up**: Full HumanEval (164 problems), APPS, LiveCodeBench
2. **Model comparison**: Test on Qwen-2.5-Coder, larger DeepSeek variants
3. **Ablations**: Injection depth, noise schedules, structured injection
4. **Analysis**: Attention patterns, embedding geometry, latent trajectories

## Citation

Based on the research question:
> Can injecting Gaussian noise into the embedding layer of frozen LLMs at inference time match or exceed the diversity benefits of temperature sampling, without any model retraining?

Inspired by seed-conditioning (Nagarajan et al., 2025, ICML) and NEFTune (Jain et al., 2024).

