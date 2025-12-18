#!/usr/bin/env python3
"""
Main script to run the sanity check experiment.

Usage:
    python sanitycheck/run_experiment.py

This will:
1. Load the model specified in config.py
2. Run all three conditions (A, B, C) on all prompts
3. Save outputs to sanitycheck/outputs/

After completion, you can run the evaluation metrics separately.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sanitycheck.config as config
from sanitycheck.prompts import get_all_prompts
from sanitycheck.inference import SanityCheckInference


def main():
    """Run the complete sanity check experiment."""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                        SANITY CHECK EXPERIMENT                               ║
║                                                                              ║
║              Embedding-Level Noise for Creativity in                         ║
║                   Verifiable Generation                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    print("\nExperiment Configuration:")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  Prompts: {config.N_PROMPTS}")
    print(f"  Samples per condition: 1 (A), {config.K_SAMPLES} (B), {config.K_SAMPLES} (C)")
    print(f"  Temperature (Condition B): {config.TEMPERATURE}")
    print(f"  Sigma scale (Condition C): {config.SIGMA_SCALE}")
    print(f"  Output directory: {config.OUTPUT_DIR}")
    
    # Confirm before proceeding
    print("\n" + "="*80)
    response = input("Proceed with experiment? [y/N]: ")
    if response.lower() != 'y':
        print("Experiment cancelled.")
        return
    
    # Load prompts
    prompts = get_all_prompts()[:config.N_PROMPTS]
    
    # Initialize inference engine
    inference = SanityCheckInference(config, prompts)
    
    # Setup (load model)
    inference.setup()
    
    # Run all conditions
    results = inference.run_all_conditions()
    
    # Save results
    inference.save_results(Path(config.OUTPUT_DIR))
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nGenerated {len(results)} total outputs:")
    print(f"  Condition A (deterministic): {sum(1 for r in results if r.condition == 'A')}")
    print(f"  Condition B (temperature): {sum(1 for r in results if r.condition == 'B')}")
    print(f"  Condition C (embed noise): {sum(1 for r in results if r.condition == 'C')}")
    print(f"\nOutputs saved to: {config.OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Inspect outputs manually")
    print("  2. Plan evaluation metrics")
    print("  3. Run systematic evaluation")


if __name__ == "__main__":
    main()

