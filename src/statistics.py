"""
Statistical analysis utilities for rigorous experimental comparison.

Provides:
- Bootstrap confidence intervals
- Wilcoxon signed-rank tests (non-parametric)
- Cliff's delta effect sizes
- Pass@K computation with unbiased estimator
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ConfidenceInterval:
    """Confidence interval result."""
    mean: float
    ci_lower: float
    ci_upper: float
    std: float
    n_samples: int


@dataclass  
class StatisticalTestResult:
    """Result of a statistical comparison test."""
    p_value: float
    effect_size: float
    effect_interpretation: str
    significant: bool
    method1_mean: float
    method2_mean: float
    test_name: str


class StatisticalAnalyzer:
    """
    Statistical analysis for comparing generation methods.
    
    Uses non-parametric tests suitable for non-normal distributions
    typical in code generation metrics.
    """
    
    def __init__(self, n_bootstrap: int = 10000, confidence_level: float = 0.95):
        """
        Initialize analyzer.
        
        Args:
            n_bootstrap: Number of bootstrap samples for CI computation
            confidence_level: Confidence level (default 95%)
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        
    def pass_at_k(self, n: int, c: int, k: int) -> float:
        """
        Unbiased estimator of Pass@K from Chen et al. (2021).
        
        Args:
            n: Total number of samples
            c: Number of correct samples
            k: k value for pass@k
            
        Returns:
            Estimated probability that at least one of k samples passes
        """
        if n - c < k:
            return 1.0
        
        # Use product formulation for numerical stability
        result = 1.0
        for i in range(k):
            result *= (n - c - i) / (n - i)
        
        return 1.0 - result
    
    def compute_pass_at_k_per_problem(
        self, 
        results: List[List[bool]], 
        k: int
    ) -> np.ndarray:
        """
        Compute Pass@K for each problem.
        
        Args:
            results: List of lists, where results[i] is a list of bool 
                     (pass/fail for each sample of problem i)
            k: k value for pass@k
            
        Returns:
            Array of Pass@K scores, one per problem
        """
        scores = []
        for problem_results in results:
            n = len(problem_results)
            c = sum(problem_results)
            score = self.pass_at_k(n, c, k)
            scores.append(score)
        return np.array(scores)
    
    def bootstrap_confidence_interval(
        self, 
        scores: np.ndarray,
        confidence: Optional[float] = None,
    ) -> ConfidenceInterval:
        """
        Compute bootstrap confidence interval for mean.
        
        Args:
            scores: Array of scores (one per problem)
            confidence: Override confidence level
            
        Returns:
            ConfidenceInterval with mean, CI bounds, std
        """
        confidence = confidence or self.confidence_level
        
        if len(scores) == 0:
            return ConfidenceInterval(0.0, 0.0, 0.0, 0.0, 0)
        
        # Bootstrap resampling
        bootstrap_means = []
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        
        for _ in range(self.n_bootstrap):
            # Sample with replacement
            sample_idx = rng.randint(0, len(scores), size=len(scores))
            sample = scores[sample_idx]
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Percentile CI
        alpha = (1 - confidence) / 2
        ci_lower = np.percentile(bootstrap_means, alpha * 100)
        ci_upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
        
        return ConfidenceInterval(
            mean=float(np.mean(scores)),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            std=float(np.std(scores)),
            n_samples=len(scores),
        )
    
    def wilcoxon_test(
        self,
        scores1: np.ndarray,
        scores2: np.ndarray,
        alternative: str = 'two-sided',
    ) -> StatisticalTestResult:
        """
        Wilcoxon signed-rank test for paired samples.
        
        Non-parametric test suitable for non-normal distributions.
        
        Args:
            scores1: Scores from method 1 (one per problem)
            scores2: Scores from method 2 (one per problem)
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            StatisticalTestResult with p-value and effect size
        """
        from scipy import stats
        
        # Handle case where scores are identical
        if np.allclose(scores1, scores2):
            return StatisticalTestResult(
                p_value=1.0,
                effect_size=0.0,
                effect_interpretation="negligible",
                significant=False,
                method1_mean=float(np.mean(scores1)),
                method2_mean=float(np.mean(scores2)),
                test_name="wilcoxon",
            )
        
        # Wilcoxon signed-rank test
        try:
            statistic, p_value = stats.wilcoxon(
                scores1, 
                scores2,
                alternative=alternative,
                zero_method='wilcox',
            )
        except ValueError:
            # All differences are zero
            p_value = 1.0
        
        # Cliff's delta effect size
        effect_size = self.cliffs_delta(scores1, scores2)
        effect_interpretation = self._interpret_cliffs_delta(effect_size)
        
        return StatisticalTestResult(
            p_value=float(p_value),
            effect_size=float(effect_size),
            effect_interpretation=effect_interpretation,
            significant=p_value < 0.05,
            method1_mean=float(np.mean(scores1)),
            method2_mean=float(np.mean(scores2)),
            test_name="wilcoxon",
        )
    
    def cliffs_delta(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Cliff's delta effect size.
        
        Non-parametric effect size suitable for ordinal data.
        Range: [-1, 1], where:
        - 1 means all x > all y
        - -1 means all x < all y
        - 0 means no difference
        
        Args:
            x: First group scores
            y: Second group scores
            
        Returns:
            Cliff's delta value
        """
        n1, n2 = len(x), len(y)
        if n1 == 0 or n2 == 0:
            return 0.0
            
        # Count dominance
        greater = sum(1 for xi in x for yi in y if xi > yi)
        less = sum(1 for xi in x for yi in y if xi < yi)
        
        return (greater - less) / (n1 * n2)
    
    def _interpret_cliffs_delta(self, delta: float) -> str:
        """
        Interpret Cliff's delta magnitude.
        
        Based on Romano et al. (2006) thresholds.
        """
        abs_delta = abs(delta)
        if abs_delta < 0.147:
            return "negligible"
        elif abs_delta < 0.330:
            return "small"
        elif abs_delta < 0.474:
            return "medium"
        else:
            return "large"
    
    def compare_methods(
        self,
        method_results: Dict[str, List[List[bool]]],
        k_values: List[int] = [1, 5, 10],
        baseline_method: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """
        Comprehensive comparison of multiple methods.
        
        Args:
            method_results: Dict mapping method name to list of problem results
            k_values: List of k values for Pass@K computation
            baseline_method: Optional baseline method to compare against
            
        Returns:
            Dictionary with Pass@K CIs and pairwise comparisons
        """
        results = {}
        
        # Compute Pass@K with CIs for each method
        pass_at_k_scores = {}
        
        for method_name, problem_results in method_results.items():
            results[method_name] = {}
            pass_at_k_scores[method_name] = {}
            
            for k in k_values:
                scores = self.compute_pass_at_k_per_problem(problem_results, k)
                pass_at_k_scores[method_name][k] = scores
                ci = self.bootstrap_confidence_interval(scores)
                
                results[method_name][f"pass@{k}"] = {
                    "mean": ci.mean,
                    "ci_lower": ci.ci_lower,
                    "ci_upper": ci.ci_upper,
                    "std": ci.std,
                }
        
        # Pairwise comparisons against baseline
        if baseline_method and baseline_method in method_results:
            results["comparisons"] = {}
            baseline_scores = pass_at_k_scores[baseline_method]
            
            for method_name in method_results:
                if method_name == baseline_method:
                    continue
                    
                results["comparisons"][f"{method_name}_vs_{baseline_method}"] = {}
                
                for k in k_values:
                    test_result = self.wilcoxon_test(
                        pass_at_k_scores[method_name][k],
                        baseline_scores[k],
                    )
                    
                    results["comparisons"][f"{method_name}_vs_{baseline_method}"][f"pass@{k}"] = {
                        "p_value": test_result.p_value,
                        "effect_size": test_result.effect_size,
                        "effect_interpretation": test_result.effect_interpretation,
                        "significant": test_result.significant,
                        "method_mean": test_result.method1_mean,
                        "baseline_mean": test_result.method2_mean,
                    }
        
        return results


def format_results_table(
    results: Dict[str, Dict],
    k_values: List[int] = [1, 5, 10],
) -> str:
    """
    Format results as a markdown table.
    
    Args:
        results: Output from StatisticalAnalyzer.compare_methods()
        k_values: K values to include
        
    Returns:
        Markdown formatted table string
    """
    lines = []
    
    # Header
    header = "| Method |"
    for k in k_values:
        header += f" Pass@{k} |"
    lines.append(header)
    
    # Separator
    sep = "|--------|"
    for _ in k_values:
        sep += "---------|"
    lines.append(sep)
    
    # Data rows
    for method_name, method_data in results.items():
        if method_name == "comparisons":
            continue
            
        row = f"| {method_name} |"
        for k in k_values:
            key = f"pass@{k}"
            if key in method_data:
                data = method_data[key]
                # Format: mean [ci_lower, ci_upper]
                row += f" {data['mean']:.3f} [{data['ci_lower']:.3f}, {data['ci_upper']:.3f}] |"
            else:
                row += " - |"
        lines.append(row)
    
    return "\n".join(lines)


def apply_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[float], float]:
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List of p-values from multiple tests
        alpha: Original significance level
        
    Returns:
        Tuple of (adjusted_p_values, corrected_alpha)
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests
    adjusted_p_values = [min(p * n_tests, 1.0) for p in p_values]
    
    return adjusted_p_values, corrected_alpha



