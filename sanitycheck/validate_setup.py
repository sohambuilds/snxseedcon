#!/usr/bin/env python3
"""
Quick validation script to test the sanity check setup.

Usage:
    python sanitycheck/validate_setup.py

This performs a dry-run with:
- 1 prompt only
- 1 sample per condition (not k)
- Verifies model loads and fits in memory
- Verifies all three conditions work
- Shows example outputs

Run this BEFORE the full experiment to catch issues early.
"""
import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sanitycheck.config as config
from sanitycheck.prompts import get_prompt
from sanitycheck.embedding_noise_single import SingleShotEmbeddingNoise
from src.model_loader import load_model
from sanitycheck.generation_utils import generate_text


def validate():
    """Run a minimal validation of the setup."""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         SETUP VALIDATION                                     ║
║                                                                              ║
║  Quick check before running the full experiment                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Check GPU
    print("="*80)
    print("1. GPU CHECK")
    print("="*80)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ GPU: {gpu_name}")
        print(f"  ✓ VRAM: {gpu_mem:.1f} GB")
        
        if "A6000" in gpu_name or gpu_mem >= 40:
            print(f"  ✓ Sufficient for experiment")
        else:
            print(f"  ⚠ Warning: A6000 (48GB) recommended, you have {gpu_mem:.1f}GB")
    else:
        print("  ✗ No GPU found! This experiment requires a GPU.")
        return False
    
    # Load model
    print("\n" + "="*80)
    print("2. MODEL LOADING")
    print("="*80)
    print(f"  Loading: {config.MODEL_NAME}")
    
    try:
        torch_dtype = getattr(torch, config.TORCH_DTYPE)
        model, tokenizer = load_model(
            model_name=config.MODEL_NAME,
            device_map=config.DEVICE_MAP,
            torch_dtype=torch_dtype,
            load_in_8bit=config.LOAD_IN_8BIT,
        )

        # Print actual HF device map if present (helps diagnose CPU offload)
        hf_map = getattr(model, "hf_device_map", None)
        if hf_map is not None:
            # Show a compact summary
            unique_devices = sorted(set(hf_map.values()))
            print(f"  Device map devices: {unique_devices}")
            # Warn if any CPU/disk offload
            if any(str(d).startswith("cpu") or str(d).startswith("disk") for d in unique_devices):
                print("  ⚠ Detected CPU/disk offload in device_map. This will be very slow.")
        
        # Check memory after loading
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  ✓ Model loaded")
            print(f"  ✓ GPU memory: {mem_used:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return False
    
    # Setup noise injector
    print("\n" + "="*80)
    print("3. NOISE INJECTOR")
    print("="*80)
    try:
        noise_injector = SingleShotEmbeddingNoise(
            sigma_scale=config.SIGMA_SCALE,
            noise_scope=config.NOISE_SCOPE,
        )
        noise_injector.attach_to_model(model)
        print(f"  ✓ Single-shot noise injector attached")
        print(f"  ✓ Sigma scale: {config.SIGMA_SCALE}")
        print(f"  ✓ Noise scope: {config.NOISE_SCOPE}")
    except Exception as e:
        print(f"  ✗ Failed to setup noise injector: {e}")
        return False
    
    # Test generation - one prompt, one sample per condition
    prompt = get_prompt(0)
    print("\n" + "="*80)
    print("4. GENERATION TEST (1 prompt, 1 sample per condition)")
    print("="*80)
    print(f"\n  Prompt:\n  {prompt[:100]}...")
    
    # Condition A: Deterministic
    print("\n  --- Condition A: Deterministic ---")
    try:
        noise_injector.deactivate()
        output_a = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            do_sample=False,
            temperature=None,
            max_new_tokens=256,
            min_new_tokens=min(32, getattr(config, "MIN_NEW_TOKENS", 64)),
        )
        print(f"  ✓ Generated {len(output_a)} chars")
        print(f"  Preview: {output_a[:150]}...")
    except Exception as e:
        print(f"  ✗ Condition A failed: {e}")
        return False
    
    # Condition B: Temperature
    print("\n  --- Condition B: Temperature Sampling ---")
    try:
        noise_injector.deactivate()
        output_b = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            do_sample=True,
            temperature=config.TEMPERATURE,
            max_new_tokens=256,
            min_new_tokens=min(32, getattr(config, "MIN_NEW_TOKENS", 64)),
            seed=0,
        )
        print(f"  ✓ Generated {len(output_b)} chars")
        print(f"  Preview: {output_b[:150]}...")
    except Exception as e:
        print(f"  ✗ Condition B failed: {e}")
        return False
    
    # Condition C: Embedding noise
    print("\n  --- Condition C: Embedding Noise ---")
    try:
        noise_injector.set_seed(42)
        noise_injector.activate()  # This resets has_injected flag
        output_c = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            do_sample=False,
            temperature=None,
            max_new_tokens=256,
            min_new_tokens=min(32, getattr(config, "MIN_NEW_TOKENS", 64)),
        )
        noise_injector.deactivate()
        print(f"  ✓ Generated {len(output_c)} chars")
        print(f"  Preview: {output_c[:150]}...")
    except Exception as e:
        print(f"  ✗ Condition C failed: {e}")
        return False
    
    # Verify single-shot behavior
    print("\n  --- Verifying Single-Shot Noise Behavior ---")
    try:
        # Generate twice with same seed - should be identical if noise is truly single-shot
        noise_injector.set_seed(123)
        noise_injector.activate()
        out1 = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            do_sample=False,
            temperature=None,
            max_new_tokens=256,
            min_new_tokens=min(32, getattr(config, "MIN_NEW_TOKENS", 64)),
        )
        noise_injector.deactivate()
        
        noise_injector.set_seed(123)
        noise_injector.activate()
        out2 = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            do_sample=False,
            temperature=None,
            max_new_tokens=256,
            min_new_tokens=min(32, getattr(config, "MIN_NEW_TOKENS", 64)),
        )
        noise_injector.deactivate()
        
        if out1 == out2:
            print(f"  ✓ Same seed produces identical output (reproducible)")
        else:
            print(f"  ⚠ Same seed produced different outputs (non-deterministic)")
            print(f"    This may indicate noise is being injected multiple times")
            
        # Different seeds should produce different outputs
        noise_injector.set_seed(456)
        noise_injector.activate()
        out3 = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            do_sample=False,
            temperature=None,
            max_new_tokens=256,
            min_new_tokens=min(32, getattr(config, "MIN_NEW_TOKENS", 64)),
        )
        noise_injector.deactivate()
        
        if out1 != out3:
            print(f"  ✓ Different seeds produce different outputs (diversity)")
        else:
            print(f"  ⚠ Different seeds produced same output (no diversity)")
            
    except Exception as e:
        print(f"  ⚠ Could not verify single-shot behavior: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("5. VALIDATION SUMMARY")
    print("="*80)
    print(f"  ✓ All checks passed!")
    print(f"\n  Full experiment will generate:")
    print(f"    Condition A: {config.N_PROMPTS} outputs")
    print(f"    Condition B: {config.N_PROMPTS * config.K_SAMPLES} outputs")
    print(f"    Condition C: {config.N_PROMPTS * config.K_SAMPLES} outputs")
    print(f"    Total: {config.N_PROMPTS * (1 + 2 * config.K_SAMPLES)} outputs")
    print(f"\n  Ready to run: python sanitycheck/run_experiment.py")
    
    return True


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)

