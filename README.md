# Inference-Time Embedding Noise for Diverse Generation

Research implementation testing whether injecting Gaussian noise into LLM embeddings at inference time can provide controlled diversity, without any model retraining.

## 🔬 Research Question

> Can injecting Gaussian noise into the embedding layer of frozen LLMs at inference time provide better diversity-quality tradeoffs than temperature sampling?

Inspired by seed-conditioning (Nagarajan et al., 2025, ICML) and NEFTune (Jain et al., 2024).

---

## 📊 Preliminary Findings (Code Solution Generation)

### Key Result: EmbedNoise Outperforms Temperature on Pass@K!

**DeepSeek-Coder-6.7B on HumanEval (5 problems, 10 samples each):**

| Method | Pass@1 | Pass@5 | Pass@10 | Compile% | Distinct-2 |
|--------|--------|--------|---------|----------|------------|
| Greedy | 0.600 | 0.993 | 1.000 | 80% | 0.092 |
| Temperature (0.8) | 0.320 | 0.869 | 0.987 | 38% | 0.510 |
| **EmbedNoise (σ=0.002)** | **0.440** | **0.954** | 0.999 | 40% | 0.298 |

**EmbedNoise vs Temperature:**
- ✅ **Pass@1: +37% improvement** (0.44 vs 0.32)
- ✅ **Pass@5: +10% improvement** (0.95 vs 0.87)
- ✅ Similar compile rate (~40%)
- ⚠️ Lower diversity (but higher correctness!)

### Optimal Noise Magnitude

Extensive sigma sweep revealed a narrow usable range:

| σ (per-token) | Compile% | Diversity | Assessment |
|---------------|----------|-----------|------------|
| 0.001 | 40% | Low | Too conservative |
| **0.002** | **80%** | Moderate | **Sweet spot** |
| 0.005 | 40% | Higher | Starting to degrade |
| 0.01 | 0% | High | Broken |
| 0.1 | 0% | Gibberish | "!!!!!!!" |

### Interpretation

1. **Embedding noise creates diversity differently than temperature:**
   - Temperature: Random token-level perturbations → diverse but often incorrect
   - EmbedNoise: Systematic embedding shift → stays "in distribution", fewer errors

2. **The diversity is surface-level:**
   - Whitespace variations (`i + 1` vs `i+1`)
   - Different comments/docstrings
   - NOT algorithmic diversity (same solution structure)

3. **For Pass@K, this is fine:**
   - We just need ONE correct solution among K samples
   - More conservative diversity = fewer broken samples

### Limitations Discovered

- Very narrow usable σ range (0.001-0.005)
- Diversity is syntactic, not semantic
- May not generalize to tasks requiring true creativity

---

## 🚀 Quick Start

### Install Dependencies

```bash
uv add torch transformers accelerate bitsandbytes datasets nltk python-Levenshtein tqdm
```

### Run Tests

```bash
# Quick validation (5 problems)
python test_minimal.py

# Inspect outputs visually
python test_inspect_outputs.py

# Find optimal sigma
python test_sigma_sweep.py

# Test larger model
python test_large_model.py
```

---

## 🔧 How It Works

### Embedding Noise Injection

```python
h₀ = E(x) + ε, where ε ~ N(0, σ²I)
σ = sigma_scale × E[||E(xᵢ)||₂]
```

The `EmbeddingNoiseInjector` class uses PyTorch forward hooks to intercept embeddings and add Gaussian noise before the transformer layers process them.

### Key Parameters

- `sigma_scale`: Noise magnitude (optimal: 0.002 for code)
- `noise_scope`: `"per_token"` or `"per_sequence"`
- `seed`: For reproducible diverse outputs

---

## 📁 Project Structure

```
src/
├── model_loader.py       # Load models with quantization
├── embedding_noise.py    # Core noise injection via PyTorch hooks
├── generation.py         # Generation harness (greedy/temp/nucleus/noise)
├── humaneval_loader.py   # Dataset loading + sandboxed execution
└── metrics.py            # Pass@K, Distinct-N, Self-BLEU, edit distance

Test Scripts:
├── test_minimal.py           # Quick validation
├── test_inspect_outputs.py   # Visual inspection of outputs
├── test_sigma_sweep.py       # Find optimal sigma
├── test_diversity_check.py   # Check seed diversity
├── test_large_model.py       # Test larger models
└── test_full_comparison.py   # Full-scale comparison
```

---

## 🔮 Next Experiments

The current findings on code solution generation show promise, but the task is highly constrained. Better domains to test creative diversity:

1. **Problem Generation** - Generate coding/math problems (no single correct answer)
2. **Creative Writing** - Story continuation, poetry generation
3. **Brainstorming** - Idea generation for a given topic

These tasks have multiple valid outputs, making diversity more meaningful.

---

## 📚 References

- Seed-conditioning (Nagarajan et al., 2025, ICML): Training with random prefixes
- NEFTune (Jain et al., 2024): Embedding noise during training
- Temperature sampling critiques (Finlayson et al., 2024)

