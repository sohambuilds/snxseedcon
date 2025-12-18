#!/usr/bin/env python3
"""
Binary pass/fail judgment for generated competitive programming problems.

Simpler than groq_judge.py - just asks:
  - Is this creative/non-generic? (true/false)
  - Is this a valid problem? (true/false)

More lenient criteria suitable for weaker models like deepseek-6.7b.

Usage:
    export GROQ_API_KEY="your-key-here"
    python sanitycheck/groq_judge2.py --n-per-condition 10
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

# Lenient prompt for weaker models
JUDGE_PROMPT = """\
You are evaluating a generated competitive programming problem.

Answer TWO yes/no questions:

1. CREATIVE: Does this contain ANY problem idea (not just boilerplate/code/empty)?
   - YES if there's an actual problem statement with some task to solve
   - NO if it's just code, empty, boilerplate, or completely incoherent

2. VALID: Could this problem theoretically be solved?
   - YES if there's a clear task, even if constraints are missing or vague
   - NO if the problem is fundamentally broken, contradictory, or unsolvable

Be LENIENT - we're evaluating a small model. Accept partial/incomplete problems.

Return ONLY valid JSON:
{"creative": true, "creative_reason": "brief", "valid": true, "valid_reason": "brief"}

PROBLEM:
"""


@dataclass
class Judgment:
    condition: str
    prompt_idx: int
    sample_idx: int
    creative: bool
    creative_reason: str
    valid: bool
    valid_reason: str
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
    ap = argparse.ArgumentParser(description="Binary pass/fail Groq judge")
    ap.add_argument("--results", default="sanitycheck/outputs/all_results.json")
    ap.add_argument("--out-jsonl", default="sanitycheck/outputs/groq_binary_judgments.jsonl")
    ap.add_argument("--out-summary", default="sanitycheck/outputs/groq_binary_summary.json")
    ap.add_argument("--n-per-condition", type=int, default=None)
    ap.add_argument("--include-a", action="store_true")
    ap.add_argument("--model", default=os.environ.get("GROQ_MODEL", DEFAULT_MODEL))
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

    # Initialize Groq client
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
    print(f"Mode: Binary pass/fail (lenient)")

    judgments: list[Judgment] = []
    
    with open(out_jsonl, "w", encoding="utf-8") as f_out:
        for i, r in enumerate(filtered):
            cond = r["condition"]
            prompt_idx = r["prompt_idx"]
            sample_idx = r["sample_idx"]
            text = r.get("generated_text") or ""

            print(f"  [{i+1}/{len(filtered)}] {cond} p{prompt_idx} s{sample_idx}...", end=" ", flush=True)

            # Skip empty/whitespace - auto-fail
            if not text.strip():
                j = Judgment(
                    condition=cond, prompt_idx=prompt_idx, sample_idx=sample_idx,
                    creative=False, creative_reason="Empty output",
                    valid=False, valid_reason="Empty output",
                    raw_response="", error="empty_output",
                )
                judgments.append(j)
                f_out.write(json.dumps(asdict(j)) + "\n")
                print("SKIP (empty) -> C=✗ V=✗")
                continue

            raw = ""
            try:
                truncated = text[:5000] if len(text) > 5000 else text
                raw = call_groq(client, args.model, truncated)
                parsed = extract_json(raw)
                
                # Handle various boolean formats
                creative = parsed.get("creative", False)
                if isinstance(creative, str):
                    creative = creative.lower() in ("true", "yes", "1")
                    
                valid = parsed.get("valid", False)
                if isinstance(valid, str):
                    valid = valid.lower() in ("true", "yes", "1")
                    
                j = Judgment(
                    condition=cond, prompt_idx=prompt_idx, sample_idx=sample_idx,
                    creative=bool(creative),
                    creative_reason=str(parsed.get("creative_reason", "")),
                    valid=bool(valid),
                    valid_reason=str(parsed.get("valid_reason", "")),
                    raw_response=raw,
                )
                c_mark = "✓" if j.creative else "✗"
                v_mark = "✓" if j.valid else "✗"
                print(f"C={c_mark} V={v_mark}")
                
            except Exception as e:
                error_msg = str(e)
                if args.debug and raw:
                    print(f"\nDEBUG raw response:\n{raw[:500]}\n")
                j = Judgment(
                    condition=cond, prompt_idx=prompt_idx, sample_idx=sample_idx,
                    creative=False, creative_reason="",
                    valid=False, valid_reason="",
                    raw_response=raw,
                    error=error_msg,
                )
                print(f"ERROR: {error_msg[:60]}")

            judgments.append(j)
            f_out.write(json.dumps(asdict(j)) + "\n")
            f_out.flush()

            time.sleep(RATE_LIMIT_DELAY)

    # Aggregate
    print("\n" + "="*60)
    print("SUMMARY (Binary Pass/Fail)")
    print("="*60)

    summary = {}
    for cond in sorted(keep_conditions):
        cond_js = [j for j in judgments if j.condition == cond]
        n_errors = sum(1 for j in cond_js if j.error and j.error != "empty_output")
        n_empty = sum(1 for j in cond_js if j.error == "empty_output")
        
        # Include all judgments (empty ones count as False)
        creative_passes = sum(1 for j in cond_js if j.creative)
        valid_passes = sum(1 for j in cond_js if j.valid)
        both_passes = sum(1 for j in cond_js if j.creative and j.valid)
        
        total = len(cond_js)
        
        stats = {
            "n_total": total,
            "n_empty": n_empty,
            "n_errors": n_errors,
            "creative_pass": creative_passes,
            "creative_rate": creative_passes / total if total > 0 else 0,
            "valid_pass": valid_passes,
            "valid_rate": valid_passes / total if total > 0 else 0,
            "both_pass": both_passes,
            "both_rate": both_passes / total if total > 0 else 0,
        }
        summary[cond] = stats

        print(f"\nCondition {cond} (n={total}, empty={n_empty}, errors={n_errors}):")
        print(f"  Creative: {creative_passes}/{total} = {stats['creative_rate']*100:.1f}%")
        print(f"  Valid:    {valid_passes}/{total} = {stats['valid_rate']*100:.1f}%")
        print(f"  Both:     {both_passes}/{total} = {stats['both_rate']*100:.1f}%")

    # Compare B vs C
    if "B" in summary and "C" in summary:
        print("\n" + "-"*60)
        print("B vs C Comparison:")
        print("-"*60)
        b, c = summary["B"], summary["C"]
        
        print(f"  Creative rate:  B={b['creative_rate']*100:.1f}%  C={c['creative_rate']*100:.1f}%  Δ={( c['creative_rate']-b['creative_rate'])*100:+.1f}%")
        print(f"  Valid rate:     B={b['valid_rate']*100:.1f}%  C={c['valid_rate']*100:.1f}%  Δ={(c['valid_rate']-b['valid_rate'])*100:+.1f}%")
        print(f"  Both rate:      B={b['both_rate']*100:.1f}%  C={c['both_rate']*100:.1f}%  Δ={(c['both_rate']-b['both_rate'])*100:+.1f}%")
        
        # Key insight
        print("\n  Interpretation:")
        if c['creative_rate'] > b['creative_rate'] and c['valid_rate'] >= b['valid_rate'] * 0.9:
            print("  ✓ Embedding noise (C) shows higher creativity with similar validity")
        elif c['creative_rate'] > b['creative_rate']:
            print("  ~ Embedding noise (C) shows higher creativity but lower validity")
        elif c['creative_rate'] < b['creative_rate']:
            print("  ✗ Temperature (B) shows higher creativity than embedding noise (C)")
        else:
            print("  ~ Similar creative rates between B and C")

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {out_jsonl}")
    print(f"Wrote {out_summary}")


if __name__ == "__main__":
    main()

