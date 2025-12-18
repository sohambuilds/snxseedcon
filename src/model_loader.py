"""
Model loading utilities for code generation experiments.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(
    model_name: str = "deepseek-ai/deepseek-coder-6.7b-instruct",
    device_map: str = "auto",
    torch_dtype = torch.float16,
    load_in_8bit: bool = False,
):
    """
    Load a causal LM with optional quantization.
    
    Args:
        model_name: HuggingFace model identifier
        device_map: Device placement strategy
        torch_dtype: Model dtype (float16 for inference)
        load_in_8bit: Whether to use int8 quantization
    
    Returns:
        model, tokenizer tuple
    """
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_kwargs = {
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        # Remove torch_dtype when using 8bit
        model_kwargs.pop("torch_dtype", None)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )
    
    model.eval()
    print(f"Model loaded. Device: {next(model.parameters()).device}")
    
    return model, tokenizer



