"""
Core embedding noise injection for inference-time diversity.
"""
import torch
from typing import Optional, Literal


class EmbeddingNoiseInjector:
    """
    Injects Gaussian noise into embedding layer at inference time.
    
    The key hypothesis: adding noise h_0 = E(x) + ε where ε ~ N(0, σ²I)
    can induce diverse outputs similar to temperature sampling, without
    requiring any model retraining.
    """
    
    def __init__(
        self,
        sigma_scale: float = 0.1,
        noise_scope: Literal["per_token", "per_sequence"] = "per_token",
        seed: Optional[int] = None,
    ):
        """
        Args:
            sigma_scale: Noise std as fraction of mean embedding norm
            noise_scope: Apply noise per-token or share one noise vector per sequence
            seed: Random seed for reproducibility
        """
        self.sigma_scale = sigma_scale
        self.noise_scope = noise_scope
        self.seed = seed
        self._generator = None
        self._hook_handle = None
        self._is_active = False
        
    def _get_generator(self, device: torch.device) -> torch.Generator:
        """Get or create random generator for reproducibility."""
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device)
            if self.seed is not None:
                self._generator.manual_seed(self.seed)
        return self._generator
    
    def compute_sigma(self, embeddings: torch.Tensor) -> float:
        """
        Compute σ based on mean embedding norm.
        σ = sigma_scale × E[||E(x_i)||_2]
        """
        # embeddings shape: (batch, seq_len, embed_dim)
        norms = embeddings.norm(dim=-1)  # (batch, seq_len)
        mean_norm = norms.mean().item()
        return self.sigma_scale * mean_norm
    
    def inject_noise(
        self, 
        embeddings: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Add Gaussian noise to embeddings.
        
        Args:
            embeddings: Input embeddings of shape (batch, seq_len, embed_dim)
            generator: Optional torch generator for reproducibility
            
        Returns:
            Perturbed embeddings of same shape
        """
        if not self._is_active:
            return embeddings
            
        sigma = self.compute_sigma(embeddings)
        
        if generator is None:
            generator = self._get_generator(embeddings.device)
        
        if self.noise_scope == "per_token":
            # Independent noise for each token
            noise = torch.randn(
                embeddings.shape,
                device=embeddings.device,
                dtype=embeddings.dtype,
                generator=generator,
            ) * sigma
        else:  # per_sequence
            # Same noise vector shared across all positions
            batch_size, seq_len, embed_dim = embeddings.shape
            noise_vec = torch.randn(
                (batch_size, 1, embed_dim),
                device=embeddings.device,
                dtype=embeddings.dtype,
                generator=generator,
            ) * sigma
            noise = noise_vec.expand(-1, seq_len, -1)
        
        return embeddings + noise
    
    def _embedding_hook(self, module, input, output):
        """Forward hook to inject noise after embedding layer."""
        return self.inject_noise(output)
    
    def attach_to_model(self, model) -> None:
        """
        Attach noise injection hook to model's embedding layer.
        
        Works with most HuggingFace causal LMs.
        """
        # Find the embedding layer - different models have different structures
        embed_layer = None
        
        # Try common patterns
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            # LLaMA, Mistral, Qwen style
            embed_layer = model.model.embed_tokens
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            # GPT-2 style
            embed_layer = model.transformer.wte
        elif hasattr(model, 'model') and hasattr(model.model, 'embeddings'):
            # Some other architectures
            embed_layer = model.model.embeddings
        else:
            # Try to find it dynamically
            for name, module in model.named_modules():
                if 'embed' in name.lower() and 'token' in name.lower():
                    embed_layer = module
                    break
        
        if embed_layer is None:
            raise ValueError("Could not find embedding layer in model")
        
        self._hook_handle = embed_layer.register_forward_hook(self._embedding_hook)
        print(f"Attached noise hook to embedding layer: {type(embed_layer).__name__}")
    
    def detach(self) -> None:
        """Remove the hook from the model."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
    
    def activate(self) -> None:
        """Enable noise injection."""
        self._is_active = True
        
    def deactivate(self) -> None:
        """Disable noise injection (pass-through mode)."""
        self._is_active = False
    
    def set_seed(self, seed: int) -> None:
        """Update seed for reproducible noise."""
        self.seed = seed
        self._generator = None  # Reset generator
        
    def __enter__(self):
        """Context manager entry - activate noise."""
        self.activate()
        return self
    
    def __exit__(self, *args):
        """Context manager exit - deactivate noise."""
        self.deactivate()

