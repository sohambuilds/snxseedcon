# Inference-Time Embedding Noise for Diverse Generation

Research implementation testing whether injecting Gaussian noise into LLM embeddings at inference time can provide controlled diversity, without any model retraining.

## 🔬 Research Question

> Can injecting Gaussian noise into the embedding layer of frozen LLMs at inference time provide better diversity-quality tradeoffs than temperature sampling?

Inspired by seed-conditioning (Nagarajan et al., 2025, ICML) and NEFTune (Jain et al., 2024).

---

## 📊 Complete Experimental Results

### Experiment 1: Code Solution Generation (HumanEval)

**Model: DeepSeek-Coder-6.7B | 5 problems × 10 samples**

| Method | Pass@1 | Pass@5 | Pass@10 | Compile% | Distinct-2 |
|--------|--------|--------|---------|----------|------------|
| Greedy | 0.600 | 0.993 | 1.000 | 80% | 0.092 |
| Temperature (0.8) | 0.320 | 0.869 | 0.987 | 38% | 0.510 |
| **EmbedNoise (σ=0.002)** | **0.440** | **0.954** | 0.999 | 40% | 0.298 |

**🏆 Winner: EmbedNoise**
- Pass@1: **+37% improvement** over temperature (0.44 vs 0.32)
- Pass@5: **+10% improvement** (0.95 vs 0.87)
- Same compile rate as temperature (~40%)

---

### Experiment 2: Problem Generation

**Model: DeepSeek-Coder-6.7B | 10 problems per method**

| Method | Valid% | UniqueNames | UniqueSigs | Docstring D2 | Self-BLEU |
|--------|--------|-------------|------------|--------------|-----------|
| Greedy | 60% | 0.500 | 0.500 | 0.458 | 1.000 |
| Temperature (0.8) | 70% | **1.000** | **1.000** | **0.947** | 0.012 |
| EmbedNoise (σ=0.002) | 80% | 0.625 | 0.750 | 0.629 | 0.719 |
| EmbedNoise (σ=0.005) | 70% | 0.857 | 1.000 | - | - |
| **EmbedNoise (σ=0.01)** | **100%** | 0.800 | 0.800 | - | - |
| EmbedNoise (σ=0.02) | 70% | 0.714 | 0.857 | - | - |

**🏆 Winner: Mixed**
- **Validity**: EmbedNoise σ=0.01 (100%) vs Temperature (70%)
- **Diversity**: Temperature (D2=0.947) vs EmbedNoise (D2=0.629)
- Trade-off: Noise = more valid, Temperature = more diverse

**Example Function Names Generated:**
- Greedy: `['reverse_words', 'find_unique_elements', 'find_common_elements', 'reverse_words', ...]` (repeating)
- Temperature: `['scrambled_words', 'flatten_and_filter', 'calculate_distance', 'hangman', 'csv_to_dict']` (all unique)
- EmbedNoise: `['reverse_words', 'reverse_list', 'maximize_sum', 'find_common_elements', ...]` (some repeats)

---

### Experiment 3: Creative Writing (Story Continuation)

**Model: DeepSeek-Coder-6.7B | 3 prompts × 10 samples**

| Method | Unique% | Distinct-2 | Distinct-3 | Self-BLEU | EditDist |
|--------|---------|------------|------------|-----------|----------|
| Greedy | 10% | 0.069 | 0.078 | 1.000 | 0.000 |
| Temperature (0.9) | 100% | 0.848 | 0.971 | 0.034 | 0.747 |
| **High Temp (1.2)** | 100% | **0.914** | **0.987** | **0.017** | 0.766 |
| EmbedNoise (σ=0.005) | 100% | 0.400 | 0.514 | 0.313 | 0.658 |
| EmbedNoise (σ=0.01) | 100% | 0.377 | 0.477 | 0.241 | 0.682 |
| EmbedNoise (σ=0.02) | 100% | 0.264 | 0.317 | 0.089 | 0.833 |
| EmbedNoise (σ=0.05) | 83% | 0.607 | 0.739 | 0.064 | 0.833 |

**🏆 Winner: Temperature**
- Best D2: High Temp (0.914) vs Best Noise (0.607)
- Temperature creates semantically diverse continuations
- Noise creates surface-level variations

---

### Experiment 4: Creative Writing on Larger Model

**Model: DeepSeek-Coder-33B | 2 prompts × 8 samples**

| Method | Unique% | Distinct-2 | Self-BLEU |
|--------|---------|------------|-----------|
| Greedy | 12.5% | 0.066 | 1.000 |
| Temperature (0.9) | 100% | 0.870 | 0.028 |
| **High Temp (1.2)** | 100% | **0.901** | **0.013** |
| EmbedNoise (σ=0.002) | 93.8% | 0.332 | 0.436 |
| EmbedNoise (σ=0.005) | 100% | 0.372 | 0.387 |
| EmbedNoise (σ=0.01) | 100% | 0.377 | 0.253 |
| EmbedNoise (σ=0.02) | 100% | 0.164 | 0.132 |

**🏆 Winner: Temperature (pattern holds at scale)**

**Surprising Finding:** 33B noise produces LESS diversity than 6.7B noise:
- 6.7B best noise D2: 0.607
- 33B best noise D2: 0.377

Interpretation: Larger models have tighter probability distributions, so embedding perturbations have less effect.

---

## 🎯 Key Findings Summary

### Task-Dependent Effectiveness

| Task Type | Winner | Why |
|-----------|--------|-----|
| **Code Solutions** | 🏆 EmbedNoise | +37% Pass@1 — safer variations stay correct |
| **Problem Generation** | 🏆 EmbedNoise (validity) | 100% valid vs 70% — controlled exploration |
| **Problem Generation** | 🏆 Temperature (diversity) | D2=0.95 vs 0.63 — radical creativity |
| **Creative Writing** | 🏆 Temperature | D2=0.91 vs 0.61 — semantic diversity needed |

### Optimal Noise Magnitude (σ)

| Task | Optimal σ | Usable Range | Notes |
|------|-----------|--------------|-------|
| Code Solutions | 0.002 | 0.001-0.005 | Very narrow! |
| Problem Generation | 0.01 | 0.005-0.02 | Wider range |
| Creative Writing | 0.05 | 0.02-0.05 | Still worse than temp |

### The Mechanism Difference

**Temperature Sampling:**
- Perturbs at token selection (output layer)
- Any token with probability mass can be selected
- Creates **semantic diversity** — different plots, ideas, algorithms
- Risk: Incoherent outputs, syntax errors

**Embedding Noise:**
- Perturbs at input embedding (input layer)
- Shifts the entire computation trajectory
- Creates **surface diversity** — formatting, comments, naming
- Benefit: Stays "in distribution", fewer errors

---

## 💡 Implications

### When to Use Embedding Noise
- ✅ Code generation (Pass@K optimization)
- ✅ Constrained generation where validity matters
- ✅ When you need reproducible diverse outputs (seeded noise)
- ✅ Batch inference with controlled variation

### When to Use Temperature
- ✅ Creative writing and brainstorming
- ✅ Open-ended generation with no "correct" answer
- ✅ When semantic diversity is the goal
- ✅ Exploratory tasks

### Paper Thesis
> Embedding noise provides **controlled diversity** ideal for correctness-sensitive tasks, while temperature provides **radical diversity** ideal for open-ended creativity. The mechanisms complement rather than replace each other.

---

## 🚀 Quick Start

### Install Dependencies

```bash
uv add torch transformers accelerate bitsandbytes datasets nltk python-Levenshtein tqdm
```

### Run Experiments

```bash
# Code solutions (original experiment)
python test_minimal.py

# Problem generation
python test_problem_generation.py

# Creative writing
python test_creative_writing.py

# 33B model experiments
python test_33b_experiments.py

# Utility scripts
python test_inspect_outputs.py   # Visual inspection
python test_sigma_sweep.py       # Find optimal σ
python test_diversity_check.py   # Check seed diversity
```

---

## 🔧 How It Works

### Embedding Noise Injection

```python
h₀ = E(x) + ε, where ε ~ N(0, σ²I)
σ = sigma_scale × E[||E(xᵢ)||₂]
```

The `EmbeddingNoiseInjector` class uses PyTorch forward hooks to intercept embeddings and add Gaussian noise before the transformer layers process them.

```python
from src.embedding_noise import EmbeddingNoiseInjector

# Create injector
injector = EmbeddingNoiseInjector(
    sigma_scale=0.002,      # Noise magnitude
    noise_scope="per_token" # or "per_sequence"
)

# Attach to model
injector.attach_to_model(model)

# Generate with different seeds for diversity
for seed in range(10):
    injector.set_seed(seed)
    injector.activate()
    output = model.generate(...)
    injector.deactivate()

# Cleanup
injector.detach()
```

---

## 📁 Project Structure

```
src/
├── model_loader.py         # Load models with quantization
├── embedding_noise.py      # Core noise injection via PyTorch hooks
├── generation.py           # Generation harness (greedy/temp/nucleus/noise)
├── humaneval_loader.py     # HumanEval dataset + sandboxed execution
├── problem_generation.py   # Problem generation utilities
└── metrics.py              # Pass@K, Distinct-N, Self-BLEU, edit distance

Test Scripts:
├── test_minimal.py             # Quick validation (code solutions)
├── test_problem_generation.py  # Problem generation experiment
├── test_creative_writing.py    # Story continuation experiment
├── test_33b_experiments.py     # Large model experiments
├── test_full_comparison.py     # Full-scale comparison
├── test_inspect_outputs.py     # Visual output inspection
├── test_sigma_sweep.py         # Find optimal sigma
├── test_diversity_check.py     # Check seed diversity
└── test_large_model.py         # Test larger models
```

---

## 📚 References

- **Seed-conditioning** (Nagarajan et al., 2025, ICML): Training with random prefixes
- **NEFTune** (Jain et al., 2024): Embedding noise during training
- **Temperature sampling critiques** (Finlayson et al., 2024): Temperature hurts coherence

---

## 🔮 Future Directions

1. **Layer-wise injection**: Test noise at different transformer layers
2. **Noise scheduling**: Anneal σ during generation
3. **Structured noise**: Apply to first k tokens only (like seed-conditioning)
4. **Combination**: Noise + low temperature together
5. **Other modalities**: Test on image/audio generation
