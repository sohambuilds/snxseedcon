#!/usr/bin/env python3
"""
Use Groq API to score generated competitive programming problems.

Scores:
  1. Creativity/Uniqueness (1-10): How novel/interesting is the problem idea?
  2. Validity (1-10 + pass/fail): Is it a well-formed, solvable problem?

Usage:
    export GROQ_API_KEY="your-key-here"
    python sanitycheck/groq_judge.py --n-per-condition 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit(
        "Missing openai package. Install with:\n"
        "  pip install openai"
    )


DEFAULT_MODEL = "openai/gpt-oss-20b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
RATE_LIMIT_DELAY = 0.3

JUDGE_PROMPT = """\
You are an expert judge for competitive programming problems.

Rate this problem on TWO dimensions:

1. CREATIVITY (1-10): How novel/original is the problem idea?
   1-3 = Very generic (find max, simple loop)
   4-6 = Standard with minor twist  
   7-10 = Interesting/original idea

2. VALIDITY (1-10): Is it a well-formed, solvable problem?
   Also mark PASS if solvable, FAIL if broken/incoherent.

Return ONLY valid JSON, no other text:
{"creativity_score": 5, "creativity_reason": "brief reason", "validity_score": 7, "validity_pass": true, "validity_reason": "brief reason"}

PROBLEM TO JUDGE:
"""


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


def extract_json(text: str) -> dict:
    """Extract JSON from response."""
    text = text.strip()
    
    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Find JSON by matching braces
    start = text.find('{')
    if start == -1:
        raise ValueError(f"No JSON object found in: {text[:200]}")
    
    depth = 0
    end = -1
    for i, c in enumerate(text[start:], start):
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = i
                break
    
    if end == -1:
        raise ValueError(f"Unclosed JSON object in: {text[:200]}")
    
    json_str = text[start:end+1]
    return json.loads(json_str)


def call_groq(client: OpenAI, model: str, problem_text: str) -> str:
    """Call Groq API using responses.create endpoint."""
    full_prompt = JUDGE_PROMPT + problem_text
    
    response = client.responses.create(
        input=full_prompt,
        model=model,
    )
    return response.output_text


def main():
    ap = argparse.ArgumentParser(description="Groq API judge for sanity check")
    ap.add_argument("--results", default="sanitycheck/outputs/all_results.json")
    ap.add_argument("--out-jsonl", default="sanitycheck/outputs/groq_judgments.jsonl")
    ap.add_argument("--out-summary", default="sanitycheck/outputs/groq_judgments_summary.json")
    ap.add_argument("--n-per-condition", type=int, default=None)
    ap.add_argument("--include-a", action="store_true")
    ap.add_argument("--model", default=os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
                    help="Groq model (default: openai/gpt-oss-20b)")
    ap.add_argument("--debug", action="store_true", help="Print raw responses on error")
    args = ap.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Set GROQ_API_KEY environment variable")

    results_path = Path(args.results)
    if not results_path.exists():
        raise SystemExit(f"Missing results: {results_path}")

    out_jsonl = Path(args.out_jsonl)
    out_summary = Path(args.out_summary)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Initialize Groq client (OpenAI-compatible)
    client = OpenAI(
        api_key=api_key,
        base_url=GROQ_BASE_URL,
    )

    all_results = load_results(results_path)
    keep_conditions = {"B", "C"} if not args.include_a else {"A", "B", "C"}
    
    filtered = [r for r in all_results if r.get("condition") in keep_conditions]
    
    if args.n_per_condition:
        limited = []
        counts = {c: 0 for c in keep_conditions}
        for r in filtered:
            c = r["condition"]
            if counts[c] < args.n_per_condition:
                limited.append(r)
                counts[c] += 1
        filtered = limited

    print(f"Judging {len(filtered)} outputs with Groq ({args.model})...")
    print(f"Conditions: {keep_conditions}")

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
                    condition=cond, prompt_idx=prompt_idx, sample_idx=sample_idx,
                    creativity_score=0, creativity_reason="Empty output",
                    validity_score=0, validity_pass=False, validity_reason="Empty output",
                    raw_response="", error="empty_output",
                )
                judgments.append(j)
                f_out.write(json.dumps(asdict(j)) + "\n")
                print("SKIP (empty)")
                continue

            raw = ""
            try:
                # Truncate very long texts
                truncated = text[:5000] if len(text) > 5000 else text
                raw = call_groq(client, args.model, truncated)
                parsed = extract_json(raw)
                    
                j = Judgment(
                    condition=cond, prompt_idx=prompt_idx, sample_idx=sample_idx,
                    creativity_score=int(parsed.get("creativity_score", 0)),
                    creativity_reason=str(parsed.get("creativity_reason", "")),
                    validity_score=int(parsed.get("validity_score", 0)),
                    validity_pass=bool(parsed.get("validity_pass", False)),
                    validity_reason=str(parsed.get("validity_reason", "")),
                    raw_response=raw,
                )
                print(f"C={j.creativity_score} V={j.validity_score} {'PASS' if j.validity_pass else 'FAIL'}")
                
            except Exception as e:
                error_msg = str(e)
                if args.debug and raw:
                    print(f"\nDEBUG raw response:\n{raw[:500]}\n")
                j = Judgment(
                    condition=cond, prompt_idx=prompt_idx, sample_idx=sample_idx,
                    creativity_score=0, creativity_reason="",
                    validity_score=0, validity_pass=False, validity_reason="",
                    raw_response=raw,
                    error=error_msg,
                )
                print(f"ERROR: {error_msg[:80]}")

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
        n_errors = sum(1 for j in judgments if j.condition == cond and j.error)
        
        if not cond_js:
            summary[cond] = {"n": 0, "n_errors": n_errors}
            continue

        creativity_scores = [j.creativity_score for j in cond_js]
        validity_scores = [j.validity_score for j in cond_js]
        validity_passes = [j.validity_pass for j in cond_js]

        stats = {
            "n": len(cond_js),
            "n_errors": n_errors,
            "creativity_mean": sum(creativity_scores) / len(creativity_scores),
            "creativity_min": min(creativity_scores),
            "creativity_max": max(creativity_scores),
            "validity_mean": sum(validity_scores) / len(validity_scores),
            "validity_min": min(validity_scores),
            "validity_max": max(validity_scores),
            "validity_pass_rate": sum(validity_passes) / len(validity_passes),
        }
        summary[cond] = stats

        print(f"\nCondition {cond} (n={stats['n']}, errors={stats['n_errors']}):")
        print(f"  Creativity:  mean={stats['creativity_mean']:.2f}  range=[{stats['creativity_min']}, {stats['creativity_max']}]")
        print(f"  Validity:    mean={stats['validity_mean']:.2f}  range=[{stats['validity_min']}, {stats['validity_max']}]")
        print(f"  Pass rate:   {stats['validity_pass_rate']*100:.1f}%")

    # Compare B vs C
    if "B" in summary and "C" in summary and summary["B"].get("n", 0) > 0 and summary["C"].get("n", 0) > 0:
        print("\n" + "-"*60)
        print("B vs C Comparison:")
        print("-"*60)
        b, c = summary["B"], summary["C"]
        print(f"  Creativity:  B={b['creativity_mean']:.2f}  C={c['creativity_mean']:.2f}  Δ={c['creativity_mean']-b['creativity_mean']:+.2f}")
        print(f"  Validity:    B={b['validity_mean']:.2f}  C={c['validity_mean']:.2f}  Δ={c['validity_mean']-b['validity_mean']:+.2f}")
        print(f"  Pass rate:   B={b['validity_pass_rate']*100:.1f}%  C={c['validity_pass_rate']*100:.1f}%")

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {out_jsonl}")
    print(f"Wrote {out_summary}")


if __name__ == "__main__":
    main()
