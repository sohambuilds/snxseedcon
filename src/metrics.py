"""
Evaluation metrics for code generation diversity and correctness.
"""
from typing import List
from collections import Counter
import math


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator of pass@k from the Codex paper.
    
    Args:
        n: Total number of samples
        c: Number of correct samples
        k: k value for pass@k
        
    Returns:
        Estimated probability that at least one of k samples passes
    """
    if n - c < k:
        return 1.0
    
    # Use log space for numerical stability
    # pass@k = 1 - C(n-c, k) / C(n, k)
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    
    return 1.0 - result


def distinct_n(texts: List[str], n: int) -> float:
    """
    Compute Distinct-N: ratio of unique n-grams to total n-grams.
    
    Higher values indicate more diversity.
    
    Args:
        texts: List of generated text samples
        n: n-gram size (1, 2, or 3 typically)
        
    Returns:
        Distinct-N score in [0, 1]
    """
    all_ngrams = []
    
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
    
    if len(all_ngrams) == 0:
        return 0.0
    
    unique_ngrams = set(all_ngrams)
    return len(unique_ngrams) / len(all_ngrams)


def self_bleu(texts: List[str], n_gram: int = 4) -> float:
    """
    Compute Self-BLEU: average max BLEU of each sample against others.
    
    Lower values indicate more diversity.
    
    Args:
        texts: List of generated text samples
        n_gram: Maximum n-gram for BLEU computation
        
    Returns:
        Self-BLEU score (lower = more diverse)
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    
    if len(texts) <= 1:
        return 0.0
    
    smoother = SmoothingFunction().method1
    weights = tuple([1.0 / n_gram] * n_gram)
    
    scores = []
    for i, hypothesis in enumerate(texts):
        # References are all other texts
        references = [texts[j].split() for j in range(len(texts)) if j != i]
        hypothesis_tokens = hypothesis.split()
        
        if len(hypothesis_tokens) == 0:
            continue
            
        # Compute BLEU against all references, take max
        bleu_scores = []
        for ref in references:
            if len(ref) == 0:
                continue
            try:
                score = sentence_bleu([ref], hypothesis_tokens, weights=weights, smoothing_function=smoother)
                bleu_scores.append(score)
            except:
                pass
        
        if bleu_scores:
            scores.append(max(bleu_scores))
    
    return sum(scores) / len(scores) if scores else 0.0


def mean_edit_distance(texts: List[str], normalize: bool = True) -> float:
    """
    Compute mean pairwise Levenshtein edit distance.
    
    Higher values indicate more diversity.
    
    Args:
        texts: List of generated text samples
        normalize: Whether to normalize by max length
        
    Returns:
        Mean pairwise edit distance
    """
    import Levenshtein
    
    if len(texts) <= 1:
        return 0.0
    
    distances = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            dist = Levenshtein.distance(texts[i], texts[j])
            if normalize:
                max_len = max(len(texts[i]), len(texts[j]))
                if max_len > 0:
                    dist = dist / max_len
            distances.append(dist)
    
    return sum(distances) / len(distances) if distances else 0.0


def compilation_rate(code_samples: List[str]) -> float:
    """
    Compute the fraction of code samples that parse correctly.
    
    Args:
        code_samples: List of Python code strings
        
    Returns:
        Fraction that successfully parse (0 to 1)
    """
    import ast
    
    valid = 0
    for code in code_samples:
        try:
            ast.parse(code)
            valid += 1
        except SyntaxError:
            pass
    
    return valid / len(code_samples) if code_samples else 0.0



