# Sanity Check Results - Run 1

**Date**: December 2024  
**Model**: `deepseek-ai/deepseek-coder-6.7b-base`

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Model | deepseek-coder-6.7b-base |
| Condition A | Deterministic (T=0) |
| Condition B | Temperature sampling (T=0.8) |
| Condition C | Embedding noise (σ=0.01, per_sequence) |
| Prompts | 10 |
| Samples per condition | 10 (B, C), 1 (A) |
| Total outputs | 210 |

## Results

### Evaluation 1: Scored (1-10 scale, Groq judge)

```
Condition B (n=100, errors=0):
  Creativity:  mean=2.58  range=[1, 5]
  Validity:    mean=2.67  range=[1, 9]
  Pass rate:   13.0%

Condition C (n=100, errors=0):
  Creativity:  mean=2.65  range=[1, 5]
  Validity:    mean=3.03  range=[1, 9]
  Pass rate:   21.0%

B vs C Comparison:
  Creativity:  B=2.58  C=2.65  Δ=+0.07
  Validity:    B=2.67  C=3.03  Δ=+0.36
  Pass rate:   B=13.0%  C=21.0%
```

### Evaluation 2: Binary (pass/fail, lenient)

```
Condition B (n=100, empty=0, errors=0):
  Creative: 57/100 = 57.0%
  Valid:    42/100 = 42.0%
  Both:     39/100 = 39.0%

Condition C (n=100, empty=0, errors=0):
  Creative: 68/100 = 68.0%
  Valid:    50/100 = 50.0%
  Both:     50/100 = 50.0%

B vs C Comparison:
  Creative rate:  B=57.0%  C=68.0%  Δ=+11.0%
  Valid rate:     B=42.0%  C=50.0%  Δ=+8.0%
  Both rate:      B=39.0%  C=50.0%  Δ=+11.0%

Interpretation:
  ✓ Embedding noise (C) shows higher creativity with similar validity
```

## Summary Table

| Metric | Condition B | Condition C | Δ (C-B) | Winner |
|--------|-------------|-------------|---------|--------|
| Creativity (scored) | 2.58 | 2.65 | +0.07 | C |
| Validity (scored) | 2.67 | 3.03 | +0.36 | **C** |
| Validity Pass % | 13.0% | 21.0% | +8.0% | **C** |
| Creative % (binary) | 57.0% | 68.0% | +11.0% | **C** |
| Valid % (binary) | 42.0% | 50.0% | +8.0% | **C** |
| Both % (binary) | 39.0% | 50.0% | +11.0% | **C** |

## Conclusion

**Positive signal for the embedding noise hypothesis.**

Condition C (embedding noise with deterministic decoding) consistently outperforms Condition B (temperature sampling) across all metrics:

1. **Higher creativity** — Both scored (+0.07) and binary (+11%) evaluations show C generates more creative/novel problems.

2. **Higher validity** — Contrary to the concern that noise might degrade quality, C produces more valid problems (+8% pass rate).

3. **Practical significance** — The 11% absolute improvement in "both creative AND valid" rate (39% → 50%) represents a meaningful gain.

**Next steps**:
- Scale to larger models (7B-13B instruction-tuned)
- Test on different tasks (code generation, creative writing)
- Sweep σ values to find optimal noise magnitude

