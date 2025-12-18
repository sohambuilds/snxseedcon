#!/usr/bin/env python3
"""
Compute minimal sanity check metrics (per sanitycheckplan.md §8-§9).

Inputs:
- Consolidated results JSON (default: sanitycheck/outputs/all_results.json)
- Annotation CSV produced by make_annotations.py (default: sanitycheck/outputs/annotations.csv)

Metrics (for B vs C by default):
1) Validity@k: fraction of outputs labeled valid (per prompt, per condition)
2) AlgDiversity@k: # distinct alg_class among VALID outputs (per prompt, per condition)
3) SigDiversity@k: # distinct constraint signatures among VALID outputs (per prompt, per condition)
Optional:
4) EffectiveAlgDiversity = AlgDiversity@k * Validity@k
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ALG_CLASS_CHOICES = {"greedy", "dp", "graph", "math", "brute", "ds", "other"}


def _load_results(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _norm_bool(v: str) -> Optional[bool]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"y", "yes", "true", "1"}:
        return True
    if s in {"n", "no", "false", "0"}:
        return False
    if s == "":
        return None
    return None


def _norm_alg(v: str) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s == "":
        return None
    if s not in ALG_CLASS_CHOICES:
        return "other"
    return s


@dataclass
class RowKey:
    condition: str
    prompt_idx: int
    sample_idx: int


def _key(cond, p, s) -> RowKey:
    return RowKey(condition=str(cond), prompt_idx=int(p), sample_idx=int(s))


def _load_annotations(path: Path) -> dict[RowKey, dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        out: dict[RowKey, dict] = {}
        for row in r:
            k = _key(row["condition"], row["prompt_idx"], row["sample_idx"])
            out[k] = row
        return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute basic sanity check metrics from annotations")
    ap.add_argument("--results", default="sanitycheck/outputs/all_results.json")
    ap.add_argument("--annotations", default="sanitycheck/outputs/annotations.csv")
    ap.add_argument("--out", default="sanitycheck/outputs/metrics_summary.json")
    ap.add_argument("--include-a", action="store_true", help="Include Condition A in reporting")
    args = ap.parse_args()

    results_path = Path(args.results)
    ann_path = Path(args.annotations)
    out_path = Path(args.out)

    if not results_path.exists():
        raise SystemExit(f"Missing results JSON: {results_path}")
    if not ann_path.exists():
        raise SystemExit(
            f"Missing annotations CSV: {ann_path}\n"
            f"Create it with: python sanitycheck/make_annotations.py"
        )

    data = _load_results(results_path)
    results = data.get("results", [])
    anns = _load_annotations(ann_path)

    keep_conditions = {"B", "C"} if not args.include_a else {"A", "B", "C"}

    # prompt_idx -> condition -> list of (valid?, alg_class?, signature)
    per_prompt: dict[int, dict[str, list[tuple[Optional[bool], Optional[str], str]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    missing_ann = 0
    for r in results:
        cond = r.get("condition")
        if cond not in keep_conditions:
            continue
        k = _key(cond, r.get("prompt_idx"), r.get("sample_idx"))
        ann = anns.get(k)
        if ann is None:
            missing_ann += 1
            continue

        valid = _norm_bool(ann.get("valid", ""))
        alg = _norm_alg(ann.get("alg_class", ""))

        sig_override = (ann.get("sig_override") or "").strip()
        sig_auto = (ann.get("sig_auto") or "").strip()
        sig = sig_override if sig_override else sig_auto

        per_prompt[int(r.get("prompt_idx"))][cond].append((valid, alg, sig))

    # Compute per-prompt metrics
    prompt_rows = []
    for p_idx in sorted(per_prompt.keys()):
        for cond in sorted(per_prompt[p_idx].keys()):
            triples = per_prompt[p_idx][cond]
            labeled = [t for t in triples if t[0] is not None]
            valid_triples = [t for t in labeled if t[0] is True]

            validity_k = (len(valid_triples) / len(labeled)) if labeled else 0.0
            alg_set = {t[1] for t in valid_triples if t[1] is not None}
            sig_set = {t[2] for t in valid_triples if t[2]}

            alg_div = len(alg_set)
            sig_div = len(sig_set)
            eff_alg_div = alg_div * validity_k

            prompt_rows.append(
                {
                    "prompt_idx": p_idx,
                    "condition": cond,
                    "n_total": len(triples),
                    "n_labeled_validity": len(labeled),
                    "n_valid": len(valid_triples),
                    "validity_at_k": validity_k,
                    "alg_diversity_at_k": alg_div,
                    "sig_diversity_at_k": sig_div,
                    "effective_alg_diversity": eff_alg_div,
                }
            )

    # Aggregate summary (avg over prompts)
    by_cond = defaultdict(list)
    for row in prompt_rows:
        by_cond[row["condition"]].append(row)

    summary = {}
    for cond, rows in by_cond.items():
        summary[cond] = {
            "prompts": len({r["prompt_idx"] for r in rows}),
            "avg_validity_at_k": _mean([r["validity_at_k"] for r in rows]),
            "avg_alg_diversity_at_k": _mean([r["alg_diversity_at_k"] for r in rows]),
            "avg_sig_diversity_at_k": _mean([r["sig_diversity_at_k"] for r in rows]),
            "avg_effective_alg_diversity": _mean([r["effective_alg_diversity"] for r in rows]),
        }

    out = {
        "metadata": data.get("metadata", {}),
        "missing_annotations": missing_ann,
        "per_prompt": prompt_rows,
        "summary": summary,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Print a compact table
    print("\n=== BASIC METRICS SUMMARY (avg over prompts) ===")
    for cond in sorted(summary.keys()):
        s = summary[cond]
        print(
            f"\nCondition {cond}:"
            f"\n  avg Validity@k: {s['avg_validity_at_k']:.3f}"
            f"\n  avg AlgDiversity@k (valid only): {s['avg_alg_diversity_at_k']:.3f}"
            f"\n  avg SigDiversity@k (valid only): {s['avg_sig_diversity_at_k']:.3f}"
            f"\n  avg EffectiveAlgDiversity: {s['avg_effective_alg_diversity']:.3f}"
        )

    if missing_ann:
        print(f"\nWARNING: {missing_ann} result rows had no matching annotation row.")
        print("Make sure you regenerated annotations after regenerating outputs.")

    print(f"\nWrote summary JSON to: {out_path}")


if __name__ == "__main__":
    main()


