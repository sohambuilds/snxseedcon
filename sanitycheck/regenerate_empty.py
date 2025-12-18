#!/usr/bin/env python3
"""
Regenerate empty outputs for Condition C only.

This script:
1. Loads existing all_results.json
2. Finds empty/whitespace-only outputs in Condition C
3. Re-generates just those with new seeds
4. Updates the JSON file

Usage:
    python sanitycheck/regenerate_empty.py
"""

import sys
import json
import torch
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sanitycheck.config as config
from sanitycheck.prompts import get_prompt
from sanitycheck.embedding_noise_single import SingleShotEmbeddingNoise
from src.model_loader import load_model

# Import generation utils if available, otherwise define inline
try:
    from sanitycheck.generation_utils import generate_text
except ImportError:
    from src.generation import GenerationConfig, generate_solution
    def generate_text(model, tokenizer, prompt, max_new_tokens, min_new_tokens, do_sample, temperature):
        cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
        )
        return generate_solution(model, tokenizer, prompt, cfg)


def is_empty_output(text: str) -> bool:
    """Check if output is empty or whitespace-only."""
    if not text:
        return True
    # Check if it's just whitespace/newlines
    if not text.strip():
        return True
    # Check if it's very short (likely garbage)
    if len(text.strip()) < 50:
        return True
    return False


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    REGENERATE EMPTY CONDITION C OUTPUTS                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Paths
    results_path = Path(config.OUTPUT_DIR) / "all_results.json"
    backup_path = Path(config.OUTPUT_DIR) / "all_results_backup.json"
    
    if not results_path.exists():
        raise SystemExit(f"Missing results file: {results_path}")
    
    # Load existing results
    print(f"Loading results from {results_path}...")
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = data["results"]
    
    # Find empty Condition C outputs
    empty_indices = []
    for i, r in enumerate(results):
        if r["condition"] == "C" and is_empty_output(r.get("generated_text", "")):
            empty_indices.append(i)
    
    print(f"\nFound {len(empty_indices)} empty Condition C outputs to regenerate")
    
    if not empty_indices:
        print("Nothing to regenerate!")
        return
    
    # Show what we'll regenerate
    print("\nOutputs to regenerate:")
    for idx in empty_indices[:10]:  # Show first 10
        r = results[idx]
        print(f"  - Prompt {r['prompt_idx']}, Sample {r['sample_idx']} (seed {r.get('seed', '?')})")
    if len(empty_indices) > 10:
        print(f"  ... and {len(empty_indices) - 10} more")
    
    # Confirm
    print("\n" + "="*60)
    response = input("Proceed with regeneration? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Backup original
    print(f"\nBacking up to {backup_path}...")
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Load model
    print(f"\nLoading model: {config.MODEL_NAME}")
    torch_dtype = getattr(torch, config.TORCH_DTYPE)
    model, tokenizer = load_model(
        model_name=config.MODEL_NAME,
        device_map=config.DEVICE_MAP,
        torch_dtype=torch_dtype,
        load_in_8bit=config.LOAD_IN_8BIT,
    )
    
    # Setup noise injector
    print(f"\nSetting up noise injector (sigma={config.SIGMA_SCALE}, scope={config.NOISE_SCOPE})")
    noise_injector = SingleShotEmbeddingNoise(
        sigma_scale=config.SIGMA_SCALE,
        noise_scope=config.NOISE_SCOPE,
    )
    noise_injector.attach_to_model(model)
    
    # Regenerate each empty output
    print(f"\nRegenerating {len(empty_indices)} outputs...")
    
    regenerated_count = 0
    still_empty_count = 0
    
    for i, idx in enumerate(empty_indices):
        r = results[idx]
        prompt_idx = r["prompt_idx"]
        sample_idx = r["sample_idx"]
        old_seed = r.get("seed", sample_idx)
        
        # Use a new seed (offset by 1000 to avoid collision)
        new_seed = old_seed + 1000
        
        print(f"  [{i+1}/{len(empty_indices)}] p{prompt_idx} s{sample_idx} (seed {old_seed} -> {new_seed})...", end=" ", flush=True)
        
        # Get prompt
        prompt_text = get_prompt(prompt_idx)
        
        # Set seed and activate noise
        noise_injector.set_seed(new_seed)
        noise_injector.activate()
        
        # Generate
        try:
            generated = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_text,
                max_new_tokens=config.MAX_NEW_TOKENS,
                min_new_tokens=getattr(config, 'MIN_NEW_TOKENS', 64),
                do_sample=False,  # Greedy for Condition C
                temperature=1.0,
            )
        except Exception as e:
            generated = ""
            print(f"ERROR: {e}")
        
        # Deactivate noise
        noise_injector.deactivate()
        
        # Check if still empty
        if is_empty_output(generated):
            still_empty_count += 1
            print(f"STILL EMPTY ({len(generated)} chars)")
        else:
            regenerated_count += 1
            print(f"OK ({len(generated)} chars)")
        
        # Update result
        results[idx]["generated_text"] = generated
        results[idx]["timestamp"] = datetime.now().isoformat()
        results[idx]["seed"] = new_seed
        results[idx]["regenerated"] = True
        
        # Save progress periodically
        if (i + 1) % 10 == 0:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Final save
    print("\nSaving updated results...")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Also update individual files if they exist
    output_dir = Path(config.OUTPUT_DIR)
    for idx in empty_indices:
        r = results[idx]
        filename = f"{r['condition']}_{r['prompt_idx']}_{r['sample_idx']}.txt"
        filepath = output_dir / filename
        if filepath.exists() or True:  # Always write
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Condition {r['condition']}\n")
                f.write(f"# Prompt {r['prompt_idx']}, Sample {r['sample_idx']}\n")
                f.write(f"# Timestamp: {r['timestamp']}\n")
                f.write(f"# Seed: {r['seed']} (regenerated)\n")
                f.write(f"\n{'='*80}\n")
                f.write(f"PROMPT:\n{r['prompt_text']}\n")
                f.write(f"\n{'='*80}\n")
                f.write(f"GENERATED OUTPUT:\n{r['generated_text']}\n")
    
    # Summary
    print("\n" + "="*60)
    print("REGENERATION COMPLETE")
    print("="*60)
    print(f"  Total attempted: {len(empty_indices)}")
    print(f"  Successfully regenerated: {regenerated_count}")
    print(f"  Still empty: {still_empty_count}")
    print(f"\n  Results saved to: {results_path}")
    print(f"  Backup at: {backup_path}")


if __name__ == "__main__":
    main()

