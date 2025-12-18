#!/usr/bin/env python3
"""
Create a CSV annotation sheet for sanity check metrics.

Workflow (minimal, per sanitycheckplan.md):
1) Run inference to produce `sanitycheck/outputs/all_results.json`
2) Run this script to create `sanitycheck/outputs/annotations.csv`
3) Manually fill:
   - valid  (y/n)
   - alg_class (one of: greedy, dp, graph, math, brute, ds, other)
   - (optional) sig_override (if heuristics mis-detect structure)
4) Run `python sanitycheck/compute_basic_metrics.py`

We avoid LLM judging by default; the plan emphasizes human-verifiable outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from signature_utils import build_signature


ALG_CLASS_CHOICES = ["greedy", "dp", "graph", "math", "brute", "ds", "other"]


def _load_results(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create annotation CSV for sanity check outputs")
    ap.add_argument(
        "--results",
        default="sanitycheck/outputs/all_results.json",
        help="Path to consolidated results JSON",
    )
    ap.add_argument(
        "--out",
        default="sanitycheck/outputs/annotations.csv",
        help="Path to write annotations CSV",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing annotation CSV",
    )
    ap.add_argument(
        "--include-a",
        action="store_true",
        help="Include Condition A rows too (default: only B and C per §9 compare)",
    )
    args = ap.parse_args()

    results_path = Path(args.results)
    out_path = Path(args.out)

    if not results_path.exists():
        raise SystemExit(f"Missing results JSON: {results_path}")

    if out_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Refusing to overwrite existing: {out_path}\n"
            f"Run with --overwrite if you intend to regenerate it."
        )

    data = _load_results(results_path)
    rows = data.get("results", [])

    # Default: focus on B vs C (plan §9), unless user opts in to A
    keep_conditions = {"B", "C"} if not args.include_a else {"A", "B", "C"}

    _ensure_parent(out_path)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "prompt_idx",
                "sample_idx",
                "timestamp",
                "seed",
                # labels (human)
                "valid",  # y/n
                "alg_class",  # one of ALG_CLASS_CHOICES
                "notes",
                # auto signature (heuristic)
                "sig_auto",
                "input_type_auto",
                "n_bucket_auto",
                "multi_test_auto",
                "complexity_hint_auto",
                # optional override (human)
                "sig_override",
            ],
        )
        w.writeheader()

        n_written = 0
        for r in rows:
            cond = r.get("condition")
            if cond not in keep_conditions:
                continue

            gen = r.get("generated_text") or ""
            sig = build_signature(gen)

            w.writerow(
                {
                    "condition": cond,
                    "prompt_idx": r.get("prompt_idx"),
                    "sample_idx": r.get("sample_idx"),
                    "timestamp": r.get("timestamp"),
                    "seed": r.get("seed"),
                    "valid": "",
                    "alg_class": "",
                    "notes": "",
                    "sig_auto": sig.to_compact(),
                    "input_type_auto": sig.input_type,
                    "n_bucket_auto": sig.n_bucket,
                    "multi_test_auto": "y" if sig.multi_test else "n",
                    "complexity_hint_auto": sig.complexity_hint or "",
                    "sig_override": "",
                }
            )
            n_written += 1

    print(f"Wrote {n_written} rows to {out_path}")
    print("Fill columns: valid, alg_class (and optional sig_override), then run:")
    print("  python sanitycheck/compute_basic_metrics.py")
    print(f"Alg class choices: {', '.join(ALG_CLASS_CHOICES)}")


if __name__ == "__main__":
    main()


