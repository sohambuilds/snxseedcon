Inference-Time Embedding Noise for Diverse Code Generation
Research Question: Can injecting Gaussian noise into the embedding layer of frozen LLMs at inference time match or exceed the diversity benefits of temperature sampling, without any model retraining?

1. Motivation
The Seed-Conditioning Mystery (Nagarajan et al., 2025, ICML)
Training with random string prefixes ("seeds") + greedy decoding achieves algorithmic creativity comparable to temperature sampling
Seeds are arbitrary (e.g., "XKPQMZ") with no semantic relationship to task
Effect requires training: model learns to map meaningless seeds → diverse outputs
Open Question: Is the learned mapping necessary, or does noise merely perturb hidden states to break mode collapse?
Hypothesis: Inference-time embedding perturbation on frozen models provides equivalent diversity without training.
Impact: If true, this is a zero-cost upgrade for every deployed LLM.

2. Method
2.1 Embedding Noise Injection
Given input tokens $\mathbf{x} = [x_1, ..., x_n]$ and embedding function $E: \mathcal{V} \to \mathbb{R}^d$:
$$\mathbf{h}_0 = E(\mathbf{x}) + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 \mathbf{I}_d)$$
where $\sigma$ is scaled relative to mean embedding norm: $\sigma \in {0.01, 0.05, 0.1, 0.5, 1.0} \times \mathbb{E}[|E(x_i)|_2]$
Ablations:
Noise scope: Per-token vs. per-sequence (one $\boldsymbol{\epsilon}$ shared across all positions)
Injection depth: Embeddings only vs. after layer $k$ for $k \in {0, 1, 5, 10, 20}$
Structured injection: All tokens vs. first $k$ tokens vs. appended dummy tokens
2.2 Baseline Comparisons
Method
Configuration
Greedy
$T=0$, no noise
Temp-0.8
Standard temperature sampling
Temp-1.5
High-temperature baseline
Nucleus-0.95
Top-$p$ sampling (alternative strategy)
EmbedNoise-$\sigma$
$T=0$ + Gaussian noise at embeddings

2.3 Models & Hardware
Models: DeepSeek-Coder-33B, Qwen-2.5-Coder-32B (both fit on 3×A6000 with int8 quantization)
Inference: FP16 with Flash Attention 2, batch size 1 (sequential generation for fair comparison)

3. Experimental Design
3.1 Datasets
Dataset
Size
Metric
Domain
APPS
5000 problems
Pass@K, test cases
Code (primary)
HumanEval
164 problems
Pass@K
Code (standard)
LiveCodeBench
400 problems
Pass@K
Code (contamination-resistant)
GSM8K
1319 problems
Accuracy
Math reasoning

For each problem, generate $K=100$ solutions per method.
3.2 Metrics
Correctness:
Pass@K: Probability $\geq 1$ solution in $K$ samples passes all tests
Success@K: Stricter—binary per problem
Diversity:
Distinct-N: $|\text{unique N-grams}| / |\text{total N-grams}|$ for $N \in {1,2,3}$
Self-BLEU: $\frac{1}{K}\sum_{i=1}^K \max_{j \neq i} \text{BLEU}(s_i, s_j)$ (lower = more diverse)
Edit distance: Mean pairwise Levenshtein distance (normalized by length)
Coherence:
Compilation rate (for code)
Perplexity under original model (no noise)
The Killer Plot: Pass@K vs. Distinct-2 Pareto frontier. Methods dominating this curve win.
3.3 Statistical Protocol
Sample size: 500 problems per dataset (power analysis: detect 5% difference with 80% power)
Repeated measures: Same problems across all methods
Significance: Wilcoxon signed-rank test (paired, non-parametric)
Multiple comparisons: Bonferroni correction for 5 method comparisons

4. Mechanistic Analysis
If EmbedNoise outperforms temperature sampling, investigate why:
4.1 Embedding Space Geometry
Generate 1000 solutions to fixed problem with different noise seeds
Cluster perturbed embeddings: Do clusters correspond to solution strategies?
Measure perturbation magnitude: $|\boldsymbol{\epsilon}|_2$ vs. diversity achieved
4.2 Attention Pattern Perturbation
Extract attention weights at layers 1, 5, 10 for (greedy, temp, noise)
Hypothesis: Noise breaks spurious attention patterns causing repetition
Quantify: KL divergence of attention distributions, entropy over attention weights
4.3 Representational Trajectory Analysis
PCA on hidden states $\mathbf{h}_\ell$ at each layer $\ell$
Plot 2D trajectories for different generation methods
Question: Does noise explore different regions of latent space?
4.4 Optimal Noise Schedule
Test annealing: Start with $\sigma_{\text{high}}$, decay to $\sigma_{\text{low}}$ over generation
Test layer-specific: Different $\sigma_\ell$ for different depths
Hypothesis: Early noise matters most (cf. seed-conditioning appends prefix)

5. Expected Outcomes & Contingencies
5.1 Success Scenario (Primary Hypothesis)
Finding: EmbedNoise-$\sigma^*$ matches Pass@K of Temp-0.8 while achieving 20%+ higher Distinct-2
Interpretation: The seed-conditioning effect doesn't require training; stochastic perturbation of initial hidden states suffices to break mode collapse
Impact: Immediate deployment in inference engines (vLLM, TGI, llama.cpp)
5.2 Partial Success (Secondary Hypothesis)
Finding: EmbedNoise works for code but not math/reasoning
Interpretation: Code has high solution degeneracy + syntactic constraints filter noise; reasoning requires coherent chain-of-thought incompatible with early perturbation
Impact: Domain-specific contribution, still valuable for coding assistants
5.3 Null Result (Backup Paper)
Finding: EmbedNoise produces garbage (low compilation rate) or no diversity gain
Interpretation: Seed-conditioning works because model learns structured latent space during training; inference-time noise lacks this structure
Impact: Negative result clarifying necessity of training; characterize what makes seed-conditioning special

6. Timeline (5 Weeks)
Week
Tasks
1
Infrastructure: Inference harness, metric computation, baseline runs (greedy, temp)
2
Core experiment: $\sigma$ sweep for EmbedNoise on APPS + HumanEval
3
Ablations: Injection depth, per-token vs. per-sequence, structured noise
4
Cross-domain validation (GSM8K), mechanistic analysis (attention, embeddings)
5
Failure case analysis, write-up, ArXiv submission


7. Related Work
Seed-conditioning (Nagarajan et al., 2025): Trains models with random prefixes; we test if training is necessary
NEFTune (Jain et al., 2024): Adds embedding noise during training for instruction tuning; we do inference-only
Parameter noise in RL (Plappert et al., 2017): Weight-space noise beats action-space noise for exploration; analogous to our input-space vs. output-space comparison
Prompt perturbation (Li et al., 2023; Naik et al., 2024): Manual prompt variation induces diversity; we mechanize this via continuous noise
Temperature sampling critiques (Finlayson et al., 2024): Temperature hurts coherence; we offer alternative

8. Deliverables
Minimum (null result):
Empirical characterization of when/why embedding noise fails
Analysis of seed-conditioning's learned structure
ICLR workshop paper
Expected (partial success):
Training-free diversity method for code generation
Mechanistic understanding of noise-induced exploration
NeurIPS/ICLR conference paper
Best case (strong success):
Universal replacement for temperature sampling across domains
Theoretical framework for input vs. output perturbation
Deployed in major inference frameworks within 6 months
Top-tier venue (NeurIPS spotlight / ICLR oral)

Code Release: Full implementation + evaluation suite on GitHub (reproducibility critical for adoption)

