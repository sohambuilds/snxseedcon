#!/usr/bin/env python3
"""
Utility to quickly inspect generated outputs.

Usage:
    # View summary statistics
    python sanitycheck/inspect_outputs.py

    # View specific output
    python sanitycheck/inspect_outputs.py --file A_0_0.txt
    
    # Compare same prompt across conditions
    python sanitycheck/inspect_outputs.py --compare 0
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_all_results(output_dir: Path):
    """Load consolidated results from JSON."""
    json_path = output_dir / "all_results.json"
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        print("Run the experiment first: python sanitycheck/run_experiment.py")
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def print_summary(data):
    """Print summary statistics of results."""
    metadata = data['metadata']
    results = data['results']
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"\nModel: {metadata['model']}")
    print(f"Total outputs: {metadata['total_outputs']}")
    print(f"Prompts: {metadata['n_prompts']}")
    print(f"Samples per condition: {metadata['k_samples']}")
    
    # Count by condition
    by_condition = defaultdict(int)
    lengths = defaultdict(list)
    
    for r in results:
        cond = r['condition']
        by_condition[cond] += 1
        lengths[cond].append(len(r['generated_text']))
    
    print("\n" + "-"*80)
    print("Outputs by Condition:")
    print("-"*80)
    
    for cond in ['A', 'B', 'C']:
        count = by_condition[cond]
        avg_len = sum(lengths[cond]) / len(lengths[cond]) if lengths[cond] else 0
        min_len = min(lengths[cond]) if lengths[cond] else 0
        max_len = max(lengths[cond]) if lengths[cond] else 0
        
        cond_name = {
            'A': 'Deterministic',
            'B': 'Temperature',
            'C': 'Embedding Noise'
        }[cond]
        
        print(f"\n  Condition {cond} ({cond_name}):")
        print(f"    Count: {count}")
        print(f"    Avg length: {avg_len:.0f} chars")
        print(f"    Range: {min_len} - {max_len} chars")


def print_output_file(output_dir: Path, filename: str):
    """Print contents of a specific output file."""
    filepath = output_dir / filename
    if not filepath.exists():
        print(f"Error: {filepath} not found")
        return
    
    with open(filepath, 'r', encoding='utf-8') as f:
        print(f.read())


def compare_prompt(data, prompt_idx: int):
    """Compare outputs for the same prompt across conditions."""
    results = data['results']
    
    print("\n" + "="*80)
    print(f"COMPARING PROMPT {prompt_idx} ACROSS CONDITIONS")
    print("="*80)
    
    # Get results for this prompt
    prompt_results = [r for r in results if r['prompt_idx'] == prompt_idx]
    
    if not prompt_results:
        print(f"No results found for prompt {prompt_idx}")
        return
    
    # Show prompt
    print(f"\nPrompt:")
    print("-" * 80)
    print(prompt_results[0]['prompt_text'])
    
    # Show one example from each condition
    for cond in ['A', 'B', 'C']:
        cond_results = [r for r in prompt_results if r['condition'] == cond]
        if not cond_results:
            continue
        
        cond_name = {
            'A': 'Deterministic',
            'B': 'Temperature Sampling',
            'C': 'Embedding Noise'
        }[cond]
        
        # Show first sample for this condition
        r = cond_results[0]
        print(f"\n{'='*80}")
        print(f"Condition {cond}: {cond_name}")
        print(f"Sample {r['sample_idx']}")
        print('='*80)
        print(r['generated_text'])
        
        # Show diversity info for B and C
        if len(cond_results) > 1:
            lengths = [len(r['generated_text']) for r in cond_results]
            print(f"\n[{len(cond_results)} total samples, lengths: {min(lengths)}-{max(lengths)} chars]")


def main():
    parser = argparse.ArgumentParser(description="Inspect sanity check outputs")
    parser.add_argument(
        '--output-dir',
        default='sanitycheck/outputs',
        help='Output directory'
    )
    parser.add_argument(
        '--file',
        help='Specific output file to view'
    )
    parser.add_argument(
        '--compare',
        type=int,
        metavar='PROMPT_IDX',
        help='Compare outputs for a specific prompt across conditions'
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    # View specific file
    if args.file:
        print_output_file(output_dir, args.file)
        return
    
    # Load results
    data = load_all_results(output_dir)
    if data is None:
        return
    
    # Compare prompt
    if args.compare is not None:
        compare_prompt(data, args.compare)
        return
    
    # Default: show summary
    print_summary(data)
    
    print("\n" + "="*80)
    print("USAGE EXAMPLES")
    print("="*80)
    print("\n  View specific output:")
    print("    python sanitycheck/inspect_outputs.py --file A_0_0.txt")
    print("\n  Compare prompt 0 across conditions:")
    print("    python sanitycheck/inspect_outputs.py --compare 0")


if __name__ == "__main__":
    main()

