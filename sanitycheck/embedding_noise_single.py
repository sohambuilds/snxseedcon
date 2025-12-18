"""
Embedding noise injector for sanity check experiment.

CRITICAL DIFFERENCE from src/embedding_noise.py:
This version injects noise EXACTLY ONCE at the first forward pass,
then passes through without modification for all subsequent tokens.

This is required by the protocol (Section 5C, 6):
> "Noise is injected exactly once, at the first forward pass"
> "No randomness elsewhere in decoding"
"""
import torch
from typing import Optional, Literal


class SingleShotEmbeddingNoise:
    """
    Injects Gaussian noise into embedding layer ONCE per generation.
    
    Unlike the standard EmbeddingNoiseInjector, this version:
    1. Injects noise only on the first forward pass (prompt embeddings)
    2. Passes through without modification on subsequent forward passes
    3. Must be reset between generations via reset() or activate()
    
    This isolates the effect of representation-level stochasticity
    to the initial planning phase, not the token-by-token decoding.
    """
    
    def __init__(
        self,
        sigma_scale: float = 0.01,
        noise_scope: Literal["per_token", "per_sequence"] = "per_sequence",
        seed: Optional[int] = None,
    ):
        """
        Args:
            sigma_scale: Noise std as fraction of mean embedding norm
            noise_scope: Apply noise per-token or share one noise vector per sequence
                        For sanity check, use "per_sequence" for global plan shifts
            seed: Random seed for reproducibility
        """
        self.sigma_scale = sigma_scale
        self.noise_scope = noise_scope
        self.seed = seed
        self._generator = None
        self._hook_handle = None
        self._is_active = False
        self._has_injected = False  # Track if we've already injected this generation
        
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
        
        This makes noise magnitude proportional to embedding scale,
        preventing over-perturbation of rare tokens.
        """
        norms = embeddings.norm(dim=-1)  # (batch, seq_len)
        mean_norm = norms.mean().item()
        return self.sigma_scale * mean_norm
    
    def inject_noise(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to embeddings (ONLY on first call per generation).
        
        Args:
            embeddings: Input embeddings of shape (batch, seq_len, embed_dim)
            
        Returns:
            Perturbed embeddings on first call, unchanged embeddings thereafter
        """
        # Not active? Pass through
        if not self._is_active:
            return embeddings
        
        # Already injected this generation? Pass through
        if self._has_injected:
            return embeddings
        
        # First forward pass: inject noise and mark as done
        self._has_injected = True
        
        sigma = self.compute_sigma(embeddings)
        generator = self._get_generator(embeddings.device)
        
        if self.noise_scope == "per_token":
            # Independent noise for each token
            noise = torch.randn(
                embeddings.shape,
                device=embeddings.device,
                dtype=embeddings.dtype,
                generator=generator,
            ) * sigma
        else:  # per_sequence (recommended for sanity check)
            # Same noise vector shared across all positions
            # This encourages global plan shifts, not token-level chaos
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
        embed_layer = None
        
        # Try common patterns
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            # LLaMA, Mistral, DeepSeek, Qwen style
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
        print(f"Attached single-shot noise hook to: {type(embed_layer).__name__}")
    
    def detach(self) -> None:
        """Remove the hook from the model."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
    
    def activate(self) -> None:
        """Enable noise injection and reset injection state."""
        self._is_active = True
        self._has_injected = False  # Reset for new generation
        
    def deactivate(self) -> None:
        """Disable noise injection."""
        self._is_active = False
    
    def reset(self) -> None:
        """Reset injection state for a new generation (keeps active state)."""
        self._has_injected = False
    
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

