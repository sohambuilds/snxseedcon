"""
Semantic diversity metrics using CodeBERT embeddings.

This module provides semantic-level diversity analysis beyond surface metrics
like Distinct-N and Self-BLEU. Uses CodeBERT for code-specific embeddings.
"""
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter


class SemanticDiversityCalculator:
    """
    Calculate semantic diversity metrics using CodeBERT embeddings.
    
    Measures:
    - Mean pairwise cosine distance between embeddings
    - Number of semantic clusters (DBSCAN)
    - Cluster entropy (distribution of samples across clusters)
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the calculator.
        
        Args:
            device: Device to run embeddings on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
    def _load_model(self):
        """Lazy load CodeBERT model."""
        if self._loaded:
            return
            
        from transformers import AutoTokenizer, AutoModel
        
        print("Loading CodeBERT for semantic diversity analysis...")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        print(f"CodeBERT loaded on {self.device}")
        
    def encode(self, code_samples: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Encode code samples into CodeBERT embeddings.
        
        Args:
            code_samples: List of code strings
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of shape (n_samples, embedding_dim)
        """
        self._load_model()
        
        all_embeddings = []
        
        for i in range(0, len(code_samples), batch_size):
            batch = code_samples[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding (first token)
                embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def pairwise_cosine_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine distances between embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            
        Returns:
            Condensed distance matrix (upper triangle)
        """
        from scipy.spatial.distance import pdist
        return pdist(embeddings, metric='cosine')
    
    def cluster_embeddings(
        self, 
        embeddings: np.ndarray,
        method: str = "dbscan",
        n_clusters: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Cluster embeddings to identify semantic groups.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            method: Clustering method ('dbscan' or 'kmeans')
            n_clusters: Number of clusters (for kmeans only)
            
        Returns:
            Tuple of (cluster_labels, n_clusters_found)
        """
        if len(embeddings) < 2:
            return np.array([0] * len(embeddings)), 1
            
        if method == "dbscan":
            from sklearn.cluster import DBSCAN
            
            # Use median distance as eps for adaptive threshold
            distances = self.pairwise_cosine_distances(embeddings)
            eps = np.percentile(distances, 30)  # 30th percentile for tighter clusters
            
            clustering = DBSCAN(
                eps=max(eps, 0.1),  # Minimum eps to avoid all singletons
                min_samples=2,
                metric='cosine'
            )
            labels = clustering.fit_predict(embeddings)
            
            # Count clusters (excluding noise labeled as -1)
            n_found = len(set(labels)) - (1 if -1 in labels else 0)
            
        elif method == "kmeans":
            from sklearn.cluster import KMeans
            
            k = n_clusters or min(5, len(embeddings))
            clustering = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = clustering.fit_predict(embeddings)
            n_found = k
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
            
        return labels, max(n_found, 1)
    
    def cluster_entropy(self, labels: np.ndarray) -> float:
        """
        Calculate entropy of cluster distribution.
        
        Higher entropy = more even distribution across clusters = more diversity.
        
        Args:
            labels: Cluster labels for each sample
            
        Returns:
            Entropy value (higher = more diverse)
        """
        counts = Counter(labels)
        total = len(labels)
        
        if total == 0:
            return 0.0
            
        probs = np.array(list(counts.values())) / total
        # Add small epsilon to avoid log(0)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        return entropy
    
    def calculate_diversity(
        self, 
        code_samples: List[str],
        include_clustering: bool = True,
    ) -> Dict[str, float]:
        """
        Calculate comprehensive semantic diversity metrics.
        
        Args:
            code_samples: List of code strings
            include_clustering: Whether to run clustering analysis
            
        Returns:
            Dictionary with diversity metrics
        """
        if len(code_samples) < 2:
            return {
                "mean_cosine_distance": 0.0,
                "std_cosine_distance": 0.0,
                "min_cosine_distance": 0.0,
                "max_cosine_distance": 0.0,
                "n_semantic_clusters": 1,
                "cluster_entropy": 0.0,
            }
        
        # Filter out empty samples
        code_samples = [c for c in code_samples if c.strip()]
        if len(code_samples) < 2:
            return {
                "mean_cosine_distance": 0.0,
                "std_cosine_distance": 0.0,
                "min_cosine_distance": 0.0,
                "max_cosine_distance": 0.0,
                "n_semantic_clusters": 1,
                "cluster_entropy": 0.0,
            }
        
        # Get embeddings
        embeddings = self.encode(code_samples)
        
        # Pairwise distances
        distances = self.pairwise_cosine_distances(embeddings)
        
        results = {
            "mean_cosine_distance": float(np.mean(distances)),
            "std_cosine_distance": float(np.std(distances)),
            "min_cosine_distance": float(np.min(distances)),
            "max_cosine_distance": float(np.max(distances)),
        }
        
        # Clustering analysis
        if include_clustering:
            labels, n_clusters = self.cluster_embeddings(embeddings)
            results["n_semantic_clusters"] = n_clusters
            results["cluster_entropy"] = self.cluster_entropy(labels)
        
        return results
    
    def compare_methods(
        self,
        method_samples: Dict[str, List[str]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare semantic diversity across multiple methods.
        
        Args:
            method_samples: Dict mapping method name to list of code samples
            
        Returns:
            Dict mapping method name to diversity metrics
        """
        results = {}
        for method_name, samples in method_samples.items():
            print(f"  Computing semantic diversity for {method_name}...")
            results[method_name] = self.calculate_diversity(samples)
        return results


def compute_functional_diversity(code_samples: List[str]) -> Dict[str, float]:
    """
    Compute functional diversity metrics for code samples.
    
    Analyzes structural differences in code without using embeddings:
    - Unique function signatures
    - Different control flow patterns (loops, conditions)
    - Different data structures used
    
    Args:
        code_samples: List of code strings
        
    Returns:
        Dictionary with functional diversity metrics
    """
    import ast
    import re
    
    patterns = {
        "for_loops": [],
        "while_loops": [],
        "list_comprehensions": [],
        "recursion": [],
        "try_except": [],
        "with_statements": [],
    }
    
    for code in code_samples:
        try:
            tree = ast.parse(code)
            
            has_for = any(isinstance(node, ast.For) for node in ast.walk(tree))
            has_while = any(isinstance(node, ast.While) for node in ast.walk(tree))
            has_listcomp = any(isinstance(node, ast.ListComp) for node in ast.walk(tree))
            has_try = any(isinstance(node, ast.Try) for node in ast.walk(tree))
            has_with = any(isinstance(node, ast.With) for node in ast.walk(tree))
            
            # Check for recursion (function calls itself)
            func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            call_names = [node.func.id for node in ast.walk(tree) 
                         if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
            has_recursion = any(fn in call_names for fn in func_names)
            
            patterns["for_loops"].append(has_for)
            patterns["while_loops"].append(has_while)
            patterns["list_comprehensions"].append(has_listcomp)
            patterns["recursion"].append(has_recursion)
            patterns["try_except"].append(has_try)
            patterns["with_statements"].append(has_with)
            
        except SyntaxError:
            # If code doesn't parse, mark all as False
            for key in patterns:
                patterns[key].append(False)
    
    n = len(code_samples)
    if n == 0:
        return {"pattern_diversity": 0.0, "n_unique_patterns": 0}
    
    # Count unique pattern combinations
    pattern_tuples = []
    for i in range(n):
        pattern = tuple(patterns[k][i] for k in sorted(patterns.keys()))
        pattern_tuples.append(pattern)
    
    unique_patterns = len(set(pattern_tuples))
    
    return {
        "pattern_diversity": unique_patterns / n,
        "n_unique_patterns": unique_patterns,
        "pct_for_loops": sum(patterns["for_loops"]) / n,
        "pct_while_loops": sum(patterns["while_loops"]) / n,
        "pct_list_comprehensions": sum(patterns["list_comprehensions"]) / n,
        "pct_recursion": sum(patterns["recursion"]) / n,
    }


