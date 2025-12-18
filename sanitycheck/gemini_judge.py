#!/usr/bin/env python3
"""
Use Gemini Flash to score generated competitive programming problems.

Scores:
  1. Creativity/Uniqueness (1-10): How novel/interesting is the problem idea?
  2. Validity (1-10 + pass/fail): Is it a well-formed, solvable problem?

Usage:
    # Set your API key
    export GEMINI_API_KEY="your-key-here"
    
    # Run on all B+C outputs (or subset)
    python sanitycheck/gemini_judge.py
    
    # Limit to N samples per condition (faster test)
    python sanitycheck/gemini_judge.py --n-per-condition 10

Outputs:
    sanitycheck/outputs/gemini_judgments.jsonl   (per-output scores)
    sanitycheck/outputs/gemini_judgments_summary.json (aggregated)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

try:
    import google.generativeai as genai
except ImportError:
    raise SystemExit(
        "Missing google-generativeai. Install with:\n"
        "  pip install google-generativeai"
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-2.0-flash"  # Latest flash model
RATE_LIMIT_DELAY = 0.5  # seconds between API calls (be nice)

JUDGE_PROMPT = """\
You are an expert judge for competitive programming problems (Codeforces style).

I will show you a generated problem. Please evaluate it on two dimensions:

1. **Creativity / Uniqueness** (1-10):
   - 1-3: Very generic, seen many times (e.g., "find max in array")
   - 4-6: Standard problem with minor twist
   - 7-8: Interesting combination or novel constraint
   - 9-10: Highly original idea, would stand out in a contest

2. **Validity / Solvability** (1-10 + PASS/FAIL):
   - Does it have a clear problem statement?
   - Are constraints internally consistent?
   - Is there a plausible solution within the stated constraints?
   - PASS = a competent programmer could solve it; FAIL = broken/unsolvable/incoherent

Respond in **exactly** this JSON format (no markdown, no extra text):
{
  "creativity_score": <int 1-10>,
  "creativity_reason": "<brief reason>",
  "validity_score": <int 1-10>,
  "validity_pass": <true or false>,
  "validity_reason": "<brief reason>"
}

---
PROBLEM TO EVALUATE:
\"\"\"
{problem_text}
\"\"\"
"""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Judgment:
    condition: str
    prompt_idx: int
    sample_idx: int
    creativity_score: int
    creativity_reason: str
    validity_score: int
    validity_pass: bool
    validity_reason: str
    raw_response: str
    error: Optional[str] = None


def load_results(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", [])


def parse_judgment(text: str) -> dict:
    """Parse JSON from Gemini response (may have markdown fences)."""
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------

def init_gemini(api_key: str, model_name: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def judge_problem(model, problem_text: str) -> tuple[dict, str]:
    """
    Call Gemini to judge a single problem.
    Returns (parsed_dict, raw_response_text).
    """
    prompt = JUDGE_PROMPT.format(problem_text=problem_text[:8000])  # truncate if huge
    response = model.generate_content(prompt)
    raw = response.text
    parsed = parse_judgment(raw)
    return parsed, raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Gemini Flash judge for sanity check outputs")
    ap.add_argument(
        "--results",
        default="sanitycheck/outputs/all_results.json",
        help="Path to consolidated results JSON",
    )
    ap.add_argument(
        "--out-jsonl",
        default="sanitycheck/outputs/gemini_judgments.jsonl",
        help="Output JSONL with per-problem judgments",
    )
    ap.add_argument(
        "--out-summary",
        default="sanitycheck/outputs/gemini_judgments_summary.json",
        help="Output JSON with aggregated summary",
    )
    ap.add_argument(
        "--n-per-condition",
        type=int,
        default=None,
        help="Limit to N samples per condition (for quick test)",
    )
    ap.add_argument(
        "--include-a",
        action="store_true",
        help="Include Condition A (default: only B and C)",
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
        help="Gemini model name",
    )
    args = ap.parse_args()

    # API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY environment variable")

    results_path = Path(args.results)
    if not results_path.exists():
        raise SystemExit(f"Missing results: {results_path}")

    out_jsonl = Path(args.out_jsonl)
    out_summary = Path(args.out_summary)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Load results
    all_results = load_results(results_path)
    keep_conditions = {"B", "C"} if not args.include_a else {"A", "B", "C"}
    
    # Filter
    filtered = [r for r in all_results if r.get("condition") in keep_conditions]
    
    # Optionally limit per condition
    if args.n_per_condition:
        limited = []
        counts = {c: 0 for c in keep_conditions}
        for r in filtered:
            c = r["condition"]
            if counts[c] < args.n_per_condition:
                limited.append(r)
                counts[c] += 1
        filtered = limited

    print(f"Judging {len(filtered)} outputs with {args.model}...")
    print(f"Conditions: {keep_conditions}")

    # Init Gemini
    model = init_gemini(api_key, args.model)

    # Judge each
    judgments: list[Judgment] = []
    
    with open(out_jsonl, "w", encoding="utf-8") as f_out:
        for i, r in enumerate(filtered):
            cond = r["condition"]
            prompt_idx = r["prompt_idx"]
            sample_idx = r["sample_idx"]
            text = r.get("generated_text") or ""

            print(f"  [{i+1}/{len(filtered)}] {cond} p{prompt_idx} s{sample_idx}...", end=" ", flush=True)

            # Skip empty/whitespace
            if not text.strip():
                j = Judgment(
                    condition=cond,
                    prompt_idx=prompt_idx,
                    sample_idx=sample_idx,
                    creativity_score=0,
                    creativity_reason="Empty output",
                    validity_score=0,
                    validity_pass=False,
                    validity_reason="Empty output",
                    raw_response="",
                    error="empty_output",
                )
                judgments.append(j)
                f_out.write(json.dumps(asdict(j)) + "\n")
                print("SKIP (empty)")
                continue

            try:
                parsed, raw = judge_problem(model, text)
                j = Judgment(
                    condition=cond,
                    prompt_idx=prompt_idx,
                    sample_idx=sample_idx,
                    creativity_score=parsed.get("creativity_score", 0),
                    creativity_reason=parsed.get("creativity_reason", ""),
                    validity_score=parsed.get("validity_score", 0),
                    validity_pass=parsed.get("validity_pass", False),
                    validity_reason=parsed.get("validity_reason", ""),
                    raw_response=raw,
                )
                print(f"C={j.creativity_score} V={j.validity_score} {'PASS' if j.validity_pass else 'FAIL'}")
            except Exception as e:
                j = Judgment(
                    condition=cond,
                    prompt_idx=prompt_idx,
                    sample_idx=sample_idx,
                    creativity_score=0,
                    creativity_reason="",
                    validity_score=0,
                    validity_pass=False,
                    validity_reason="",
                    raw_response="",
                    error=str(e),
                )
                print(f"ERROR: {e}")

            judgments.append(j)
            f_out.write(json.dumps(asdict(j)) + "\n")
            f_out.flush()

            time.sleep(RATE_LIMIT_DELAY)

    # Aggregate
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    summary = {}
    for cond in sorted(keep_conditions):
        cond_js = [j for j in judgments if j.condition == cond and j.error is None]
        if not cond_js:
            summary[cond] = {"n": 0}
            continue

        creativity_scores = [j.creativity_score for j in cond_js]
        validity_scores = [j.validity_score for j in cond_js]
        validity_passes = [j.validity_pass for j in cond_js]

        stats = {
            "n": len(cond_js),
            "creativity_mean": sum(creativity_scores) / len(creativity_scores),
            "creativity_min": min(creativity_scores),
            "creativity_max": max(creativity_scores),
            "validity_mean": sum(validity_scores) / len(validity_scores),
            "validity_min": min(validity_scores),
            "validity_max": max(validity_scores),
            "validity_pass_rate": sum(validity_passes) / len(validity_passes),
        }
        summary[cond] = stats

        print(f"\nCondition {cond} (n={stats['n']}):")
        print(f"  Creativity:  mean={stats['creativity_mean']:.2f}  range=[{stats['creativity_min']}, {stats['creativity_max']}]")
        print(f"  Validity:    mean={stats['validity_mean']:.2f}  range=[{stats['validity_min']}, {stats['validity_max']}]")
        print(f"  Pass rate:   {stats['validity_pass_rate']*100:.1f}%")

    # Compare B vs C
    if "B" in summary and "C" in summary and summary["B"]["n"] > 0 and summary["C"]["n"] > 0:
        print("\n" + "-"*60)
        print("B vs C Comparison:")
        print("-"*60)
        b, c = summary["B"], summary["C"]
        print(f"  Creativity:  B={b['creativity_mean']:.2f}  C={c['creativity_mean']:.2f}  Δ={c['creativity_mean']-b['creativity_mean']:+.2f}")
        print(f"  Validity:    B={b['validity_mean']:.2f}  C={c['validity_mean']:.2f}  Δ={c['validity_mean']-b['validity_mean']:+.2f}")
        print(f"  Pass rate:   B={b['validity_pass_rate']*100:.1f}%  C={c['validity_pass_rate']*100:.1f}%")

    # Save summary
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {out_jsonl}")
    print(f"Wrote {out_summary}")


if __name__ == "__main__":
    main()

