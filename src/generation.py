"""
Generation utilities supporting different sampling strategies.
"""
import torch
from typing import List, Optional, Literal
from dataclasses import dataclass

from .embedding_noise import EmbeddingNoiseInjector


@dataclass
class GenerationConfig:
    """Configuration for code generation."""
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = False
    
    @classmethod
    def from_config_module(cls, config_module):
        """Load from config.py module."""
        return cls(max_new_tokens=getattr(config_module, 'MAX_NEW_TOKENS', 512))
    

def generate_solution(
    model,
    tokenizer,
    prompt: str,
    config: GenerationConfig,
    noise_injector: Optional[EmbeddingNoiseInjector] = None,
) -> str:
    """
    Generate a single solution for the given prompt.
    
    Args:
        model: The language model
        tokenizer: Associated tokenizer
        prompt: Input prompt
        config: Generation configuration
        noise_injector: Optional noise injector (must be pre-attached to model)
    
    Returns:
        Generated code string
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature if config.do_sample else None,
            top_p=config.top_p if config.do_sample else None,
            do_sample=config.do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode only the generated part
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text


def generate_k_solutions(
    model,
    tokenizer,
    prompt: str,
    k: int,
    method: Literal["greedy", "temperature", "nucleus", "embed_noise"],
    method_kwargs: Optional[dict] = None,
    noise_injector: Optional[EmbeddingNoiseInjector] = None,
    show_progress: bool = False,
) -> List[str]:
    """
    Generate K solutions using the specified method.
    
    Args:
        model: The language model
        tokenizer: Associated tokenizer  
        prompt: Input prompt
        k: Number of solutions to generate
        method: Sampling method
        method_kwargs: Additional kwargs for the method (e.g., temperature value)
        noise_injector: Required for embed_noise method
        show_progress: Whether to show a progress bar
    
    Returns:
        List of K generated solutions
    """
    method_kwargs = method_kwargs or {}
    solutions = []
    
    iterator = range(k)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"  {method}", leave=False)
    
    max_tokens = method_kwargs.get("max_new_tokens", 512)
    
    for i in iterator:
        if method == "greedy":
            config = GenerationConfig(do_sample=False, max_new_tokens=max_tokens)
            
        elif method == "temperature":
            temp = method_kwargs.get("temperature", 0.8)
            config = GenerationConfig(do_sample=True, temperature=temp, max_new_tokens=max_tokens)
            
        elif method == "nucleus":
            top_p = method_kwargs.get("top_p", 0.95)
            config = GenerationConfig(do_sample=True, top_p=top_p, max_new_tokens=max_tokens)
            
        elif method == "embed_noise":
            if noise_injector is None:
                raise ValueError("noise_injector required for embed_noise method")
            # Use greedy decoding with embedding noise
            config = GenerationConfig(do_sample=False, max_new_tokens=max_tokens)
            # Set different seed for each sample to get diversity
            noise_injector.set_seed(i)
            noise_injector.activate()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        solution = generate_solution(model, tokenizer, prompt, config, noise_injector)
        solutions.append(solution)
        
        # Deactivate noise after generation
        if method == "embed_noise" and noise_injector:
            noise_injector.deactivate()
    
    return solutions

