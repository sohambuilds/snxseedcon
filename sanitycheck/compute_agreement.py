#!/usr/bin/env python3
"""
Compute agreement between multiple LLM judge runs.

Usage:
    python sanitycheck/compute_agreement.py --runs run2 run3 run4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations


def load_judgments(jsonl_path: Path) -> dict[str, dict]:
    """Load judgments from JSONL file, keyed by (condition, prompt_idx, sample_idx)."""
    judgments = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            key = (j["condition"], j["prompt_idx"], j["sample_idx"])
            judgments[key] = {
                "creative": j.get("creative", False),
                "valid": j.get("valid", False),
                "error": j.get("error"),
            }
    return judgments


def compute_pairwise_agreement(j1: dict, j2: dict, keys: list) -> dict:
    """Compute pairwise agreement between two judgment sets."""
    creative_agree = 0
    valid_agree = 0
    both_agree = 0
    total = 0
    
    for key in keys:
        if key in j1 and key in j2:
            total += 1
            if j1[key]["creative"] == j2[key]["creative"]:
                creative_agree += 1
            if j1[key]["valid"] == j2[key]["valid"]:
                valid_agree += 1
            if (j1[key]["creative"] == j2[key]["creative"] and 
                j1[key]["valid"] == j2[key]["valid"]):
                both_agree += 1
    
    return {
        "total": total,
        "creative_agree": creative_agree,
        "creative_rate": creative_agree / total if total > 0 else 0,
        "valid_agree": valid_agree,
        "valid_rate": valid_agree / total if total > 0 else 0,
        "both_agree": both_agree,
        "both_rate": both_agree / total if total > 0 else 0,
    }


def compute_unanimous_agreement(all_judgments: list[dict], keys: list) -> dict:
    """Compute how often all judges agree."""
    creative_unanimous = 0
    valid_unanimous = 0
    both_unanimous = 0
    total = 0
    
    for key in keys:
        values = [j.get(key) for j in all_judgments if key in j]
        if len(values) < 2:
            continue
        
        total += 1
        creative_vals = [v["creative"] for v in values]
        valid_vals = [v["valid"] for v in values]
        
        if len(set(creative_vals)) == 1:  # All same
            creative_unanimous += 1
        if len(set(valid_vals)) == 1:
            valid_unanimous += 1
        if len(set(creative_vals)) == 1 and len(set(valid_vals)) == 1:
            both_unanimous += 1
    
    return {
        "total": total,
        "creative_unanimous": creative_unanimous,
        "creative_rate": creative_unanimous / total if total > 0 else 0,
        "valid_unanimous": valid_unanimous,
        "valid_rate": valid_unanimous / total if total > 0 else 0,
        "both_unanimous": both_unanimous,
        "both_rate": both_unanimous / total if total > 0 else 0,
    }


def compute_majority_vote(all_judgments: list[dict], keys: list) -> dict:
    """Compute majority vote results."""
    results = {"B": [], "C": []}
    
    for key in keys:
        values = [j.get(key) for j in all_judgments if key in j]
        if not values:
            continue
        
        creative_votes = sum(1 for v in values if v["creative"])
        valid_votes = sum(1 for v in values if v["valid"])
        n = len(values)
        
        majority_creative = creative_votes > n / 2
        majority_valid = valid_votes > n / 2
        
        cond = key[0]
        if cond in results:
            results[cond].append({
                "key": key,
                "creative": majority_creative,
                "valid": majority_valid,
                "creative_votes": f"{creative_votes}/{n}",
                "valid_votes": f"{valid_votes}/{n}",
            })
    
    return results


def fleiss_kappa(all_judgments: list[dict], keys: list, field: str) -> float:
    """
    Compute Fleiss' Kappa for inter-rater reliability.
    """
    n_raters = len(all_judgments)
    n_subjects = 0
    
    # Count agreements
    P_i_sum = 0
    category_counts = defaultdict(int)
    
    for key in keys:
        values = [j.get(key) for j in all_judgments if key in j]
        if len(values) != n_raters:
            continue
        
        n_subjects += 1
        n_true = sum(1 for v in values if v[field])
        n_false = n_raters - n_true
        
        category_counts[True] += n_true
        category_counts[False] += n_false
        
        # P_i = (1 / n(n-1)) * sum(n_ij * (n_ij - 1))
        P_i = (n_true * (n_true - 1) + n_false * (n_false - 1)) / (n_raters * (n_raters - 1))
        P_i_sum += P_i
    
    if n_subjects == 0:
        return 0.0
    
    # P_bar = mean of P_i
    P_bar = P_i_sum / n_subjects
    
    # P_e = sum of (p_j)^2 where p_j is proportion of all ratings in category j
    total_ratings = n_subjects * n_raters
    P_e = sum((count / total_ratings) ** 2 for count in category_counts.values())
    
    # Kappa = (P_bar - P_e) / (1 - P_e)
    if P_e == 1:
        return 1.0
    
    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa


def main():
    ap = argparse.ArgumentParser(description="Compute agreement between LLM judge runs")
    ap.add_argument("--runs", nargs="+", default=["run2", "run3", "run4"],
                    help="Run folder names")
    ap.add_argument("--base-dir", default="sanitycheck/outputs",
                    help="Base directory containing run folders")
    ap.add_argument("--file", default="groq_binary_judgments.jsonl",
                    help="JSONL filename in each run folder")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    
    # Load all judgments
    all_judgments = []
    run_names = []
    
    for run in args.runs:
        jsonl_path = base_dir / run / args.file
        if not jsonl_path.exists():
            print(f"Warning: {jsonl_path} not found, skipping")
            continue
        
        judgments = load_judgments(jsonl_path)
        all_judgments.append(judgments)
        run_names.append(run)
        print(f"Loaded {len(judgments)} judgments from {run}")
    
    if len(all_judgments) < 2:
        raise SystemExit("Need at least 2 runs to compute agreement")
    
    # Get common keys
    common_keys = set(all_judgments[0].keys())
    for j in all_judgments[1:]:
        common_keys &= set(j.keys())
    common_keys = sorted(common_keys)
    
    print(f"\nCommon samples across all runs: {len(common_keys)}")
    
    # Split by condition
    keys_b = [k for k in common_keys if k[0] == "B"]
    keys_c = [k for k in common_keys if k[0] == "C"]
    
    print(f"  Condition B: {len(keys_b)}")
    print(f"  Condition C: {len(keys_c)}")
    
    # Compute metrics
    print("\n" + "="*70)
    print("PAIRWISE AGREEMENT")
    print("="*70)
    
    for (i, name_i), (j, name_j) in combinations(enumerate(run_names), 2):
        agreement = compute_pairwise_agreement(all_judgments[i], all_judgments[j], common_keys)
        print(f"\n{name_i} vs {name_j} (n={agreement['total']}):")
        print(f"  Creative: {agreement['creative_agree']}/{agreement['total']} = {agreement['creative_rate']*100:.1f}%")
        print(f"  Valid:    {agreement['valid_agree']}/{agreement['total']} = {agreement['valid_rate']*100:.1f}%")
        print(f"  Both:     {agreement['both_agree']}/{agreement['total']} = {agreement['both_rate']*100:.1f}%")
    
    print("\n" + "="*70)
    print("UNANIMOUS AGREEMENT (all runs agree)")
    print("="*70)
    
    unanimous = compute_unanimous_agreement(all_judgments, common_keys)
    print(f"\nAll {len(run_names)} runs (n={unanimous['total']}):")
    print(f"  Creative: {unanimous['creative_unanimous']}/{unanimous['total']} = {unanimous['creative_rate']*100:.1f}%")
    print(f"  Valid:    {unanimous['valid_unanimous']}/{unanimous['total']} = {unanimous['valid_rate']*100:.1f}%")
    print(f"  Both:     {unanimous['both_unanimous']}/{unanimous['total']} = {unanimous['both_rate']*100:.1f}%")
    
    print("\n" + "="*70)
    print("FLEISS' KAPPA (inter-rater reliability)")
    print("="*70)
    
    kappa_creative = fleiss_kappa(all_judgments, common_keys, "creative")
    kappa_valid = fleiss_kappa(all_judgments, common_keys, "valid")
    
    print(f"\n  Creative: κ = {kappa_creative:.3f}")
    print(f"  Valid:    κ = {kappa_valid:.3f}")
    print("\n  Interpretation:")
    print("    κ < 0.20  = Poor")
    print("    0.21-0.40 = Fair")
    print("    0.41-0.60 = Moderate")
    print("    0.61-0.80 = Substantial")
    print("    0.81-1.00 = Almost Perfect")
    
    # Majority vote summary
    print("\n" + "="*70)
    print("MAJORITY VOTE SUMMARY")
    print("="*70)
    
    majority = compute_majority_vote(all_judgments, common_keys)
    
    for cond in ["B", "C"]:
        if cond not in majority or not majority[cond]:
            continue
        
        items = majority[cond]
        creative_pass = sum(1 for i in items if i["creative"])
        valid_pass = sum(1 for i in items if i["valid"])
        both_pass = sum(1 for i in items if i["creative"] and i["valid"])
        total = len(items)
        
        print(f"\nCondition {cond} (n={total}, majority vote):")
        print(f"  Creative: {creative_pass}/{total} = {creative_pass/total*100:.1f}%")
        print(f"  Valid:    {valid_pass}/{total} = {valid_pass/total*100:.1f}%")
        print(f"  Both:     {both_pass}/{total} = {both_pass/total*100:.1f}%")
    
    # Final B vs C comparison using majority vote
    if "B" in majority and "C" in majority and majority["B"] and majority["C"]:
        print("\n" + "-"*70)
        print("B vs C (Majority Vote):")
        print("-"*70)
        
        b_items = majority["B"]
        c_items = majority["C"]
        
        b_creative = sum(1 for i in b_items if i["creative"]) / len(b_items)
        c_creative = sum(1 for i in c_items if i["creative"]) / len(c_items)
        b_valid = sum(1 for i in b_items if i["valid"]) / len(b_items)
        c_valid = sum(1 for i in c_items if i["valid"]) / len(c_items)
        b_both = sum(1 for i in b_items if i["creative"] and i["valid"]) / len(b_items)
        c_both = sum(1 for i in c_items if i["creative"] and i["valid"]) / len(c_items)
        
        print(f"  Creative: B={b_creative*100:.1f}%  C={c_creative*100:.1f}%  Δ={(c_creative-b_creative)*100:+.1f}%")
        print(f"  Valid:    B={b_valid*100:.1f}%  C={c_valid*100:.1f}%  Δ={(c_valid-b_valid)*100:+.1f}%")
        print(f"  Both:     B={b_both*100:.1f}%  C={c_both*100:.1f}%  Δ={(c_both-b_both)*100:+.1f}%")
    
    # Save results
    output_path = base_dir / "agreement_analysis.json"
    results = {
        "runs": run_names,
        "n_common": len(common_keys),
        "unanimous": unanimous,
        "fleiss_kappa": {
            "creative": kappa_creative,
            "valid": kappa_valid,
        },
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved analysis to {output_path}")


if __name__ == "__main__":
    main()

