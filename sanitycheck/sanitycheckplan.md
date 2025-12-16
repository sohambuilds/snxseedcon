Below is a **self-contained experimental protocol document**.
It is written as if it could be dropped into a lab notebook or an appendix and followed verbatim.
No code, only *what to do*, *why*, and *what to look for*.

---

# Sanity Check Experiment

**Embedding-Level Noise for Creativity in Verifiable Generation**

---

## 1. Objective (Precisely Defined)

The purpose of this sanity check is **not** to prove a full research claim.

It is to answer a single falsifiable question:

> *Does a single, scale-aware perturbation at the input embedding level induce non-trivial creative diversity—defined as structural or conceptual variation—while preserving correctness in a verifiable generative task?*

If the answer is “no” under controlled conditions, the idea should not be scaled.

---

## 2. Scope and Constraints

* **Single GPU**: NVIDIA A6000
* **Single model**
* **Single task**
* **Minimal metrics**
* **Human-verifiable outcomes**

This experiment deliberately avoids:

* large benchmarks
* multi-task generalization
* fine-tuning
* training-time modifications

---

## 3. Model Selection

### Model Size Requirement

Choose **one mid-scale autoregressive model**, approximately **7B–13B parameters**.

Rationale:

* Models <1B collapse under noise
* Models >30B tend to normalize noise away
* Mid-scale models exhibit unstable latent planning, which is exactly what we want to probe

### Model Characteristics

The model must:

* Be decoder-only
* Use standard token embeddings followed by transformer blocks
* Be capable of structured text generation (e.g., coding tasks)

No instruction tuning tricks, adapters, or RLHF variants are required for this sanity check.

---

## 4. Task Definition

### Task Type

Use **competitive programming problem generation**.

### Task Prompt (Fixed Template)

The same prompt template must be used across all conditions:

> “Generate a competitive programming problem suitable for Codeforces Div-2 B.
> Include:
> • problem statement
> • input format
> • output format
> • constraints
> • short solution outline.”

### Why This Task

This task is ideal because it:

* Requires early planning (problem type, constraints, algorithm)
* Has many valid outputs
* Allows correctness checks without executing code
* Makes “creativity” observable as **algorithmic variation**, not style

---

## 5. Experimental Conditions

You will run **three conditions only**.
Nothing else is allowed to vary.

---

### Condition A: Deterministic Baseline

**Purpose:** Establish the model’s default planning bias.

Settings:

* Temperature = 0
* No sampling randomness
* No noise
* One generation per prompt

Expected behavior:

* Highly consistent outputs
* Same problem archetype repeated
* Minimal structural diversity

This condition defines the *baseline planning manifold*.

---

### Condition B: Temperature Sampling Baseline

**Purpose:** Compare against standard stochastic decoding.

Settings:

* Temperature > 0 (tuned so outputs are fluent but varied)
* No embedding noise
* k generations per prompt

Expected behavior:

* Lexical and stylistic variation
* Similar constraints
* Same algorithmic idea repeated with minor twists

This condition tests whether embedding noise is merely “temperature in disguise.”

---

### Condition C: Embedding-Level Noise Intervention

**Purpose:** Test the hypothesis directly.

Settings:

* Temperature = 0
* Deterministic decoding
* Single noise injection at the embedding layer
* One generation per noise seed
* k noise seeds per prompt

Crucial constraint:

* **Noise is injected exactly once**, at the first forward pass
* No randomness elsewhere in decoding

This isolates the effect of **representation-level stochasticity**.

---

## 6. Noise Design (Conceptual, Not Implementation)

### Noise Characteristics

* Zero-mean
* Isotropic
* Scaled relative to embedding magnitude

Conceptually:

* Noise perturbs *where* the model starts in representation space
* Not *how* it samples tokens

### Scaling Strategy

Noise magnitude must be proportional to the norm of the embedding vectors.

Reason:

* Prevents over-perturbing rare tokens
* Maintains relative semantic importance
* Makes results comparable across models

### Noise Granularity

For the sanity check:

* Use **one shared noise vector per generation**
* Applied uniformly across tokens

This encourages **global plan shifts**, not token-level chaos.

---

## 7. Prompt and Sampling Protocol

### Prompt Set

* Use **10 distinct prompts**
* Prompts should differ slightly in wording but ask for the same task
* Prompts are fixed across all conditions

### Sampling Budget

For each prompt:

* Condition A: 1 output
* Condition B: k outputs (e.g., k = 10)
* Condition C: k outputs (same k)

Total outputs:

* Small enough for manual inspection
* Large enough to see patterns

---

## 8. Evaluation Criteria

This sanity check uses **three metrics only**.

No BLEU, ROUGE, perplexity, or automated “creativity” scores.

---

### Metric 1: Validity Rate

Binary judgment per output:

* Are constraints internally consistent?
* Does the solution outline plausibly solve the problem?
* Is the problem solvable within stated constraints?

This filters out noise-induced garbage.

---

### Metric 2: Algorithmic Diversity

For each **valid** output, classify the dominant algorithmic idea:

* Greedy
* Dynamic Programming
* Graph traversal
* Mathematical insight
* Brute force with pruning
* Data-structure driven

Then compute:

* Number of **distinct algorithm classes** per condition per prompt

This is the **core creativity signal**.

---

### Metric 3: Constraint Structure Diversity

Extract from each problem:

* Upper bounds on N
* Time complexity implied
* Input structure (array, tree, grid, etc.)

Measure:

* How often constraint patterns repeat
* Whether different algorithmic choices correspond to different constraints

This checks whether creativity is *structural*, not cosmetic.

---

## 9. What You Compare

You are **not** comparing absolute quality.

You are comparing:

* Condition B vs Condition C **under similar validity**

Key question:

> *Does embedding noise produce more structurally distinct valid outputs than temperature sampling?*

---

## 10. Expected Outcomes (Interpretation Guide)

### Positive Signal

You observe:

* Validity rate similar to temperature sampling
* More distinct algorithmic classes under embedding noise
* Constraint structures that differ meaningfully
* Outputs that a human immediately recognizes as “different problems”

This validates the hypothesis.

---

### Negative Signal

You observe:

* No difference from deterministic baseline
* Or diversity that is purely stylistic
* Or rapid collapse of validity

This falsifies the hypothesis *for this model and task*.

That is still a valuable result.

---

## 11. Visualization and Reporting

Prepare:

* One table summarizing:

  * Validity %
  * Distinct algorithms@k
  * Constraint diversity
* One qualitative page showing:

  * Two example outputs per condition
  * Highlighted differences in problem structure

If the result cannot be explained clearly to a peer in 5 minutes, it is not strong enough.

---

## 12. Pre-Declared Stop Conditions

You **terminate further experimentation** if:

* Embedding noise behaves identically to temperature
* Creativity gains require unacceptable validity loss
* Effects are inconsistent across prompts

This prevents post-hoc rationalization.

---

## 13. Why This Sanity Check Is Sufficient

Because it:

* Is mechanistically targeted
* Is falsifiable
* Tests representation-level causality
* Costs little compute
* Produces interpretable outcomes

If this works, scaling is justified.
If it fails, the idea should be reframed or abandoned.

---

## 14. Final Note

This experiment is **not about winning benchmarks**.
It is about answering a deep mechanistic question:

> *Does early representational uncertainty alter creative planning in language models?*

This setup is the cleanest way to find out.

If you want, the next logical step would be:

* converting this into a formal “Method” section
* or designing the minimal negative-result paper if the effect is weak
