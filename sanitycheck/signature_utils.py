"""
Heuristic extraction for "constraint structure" signatures.

This is intentionally simple (per sanitycheckplan.md §8.3):
- Input structure type (array/tree/grid/graph/string/etc.)
- A coarse N upper-bound bucket (if detectable)
- Multi-testcase flag
- Optional explicit complexity hint if present (e.g. "O(N log N)")

These are not "metrics" by themselves; they are helpers used by the basic
metrics scripts to count distinct signatures among *valid* outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


_RE_O_COMPLEXITY = re.compile(r"O\\s*\\(([^\\)]{1,40})\\)", re.IGNORECASE)
_RE_TESTCASES = re.compile(r"\\btest\\s*cases\\b|\\beach\\s*test\\s*case\\b", re.IGNORECASE)


def _normalize_text(s: str) -> str:
    return (s or "").lower()


def detect_input_type(text: str) -> str:
    t = _normalize_text(text)

    # Strong signals first
    if any(k in t for k in ["grid", "matrix", "n rows", "m columns", "cell (i, j)", "2d"]):
        return "grid"
    if any(k in t for k in ["tree", "rooted tree", "parent", "subtree", "lca"]):
        return "tree"
    if any(k in t for k in ["graph", "vertices", "edges", "adjacency", "shortest path", "bfs", "dfs"]):
        return "graph"
    if any(k in t for k in ["string", "substring", "palindrome", "characters"]):
        return "string"
    if any(k in t for k in ["permutation", "permute"]):
        return "permutation"
    if any(k in t for k in ["array", "sequence", "list of integers", "a1", "a_i", "ai", "elements"]):
        return "array"

    # Weak fallback
    if "integer" in t or "numbers" in t:
        return "numeric"
    return "unknown"


def _parse_sci_like(token: str) -> Optional[int]:
    """
    Parse common CP constraint formats into an int.
    Supports: 2e5, 1e6, 200000, 10^5, 10^6, 1_000_000
    """
    if not token:
        return None
    tok = token.strip().lower().replace(",", "").replace("_", "")

    # 10^5
    m = re.fullmatch(r"10\\s*\\^\\s*(\\d{1,2})", tok)
    if m:
        return 10 ** int(m.group(1))

    # 2e5, 1e6, 5e4
    m = re.fullmatch(r"(\\d{1,3})\\s*e\\s*(\\d{1,2})", tok)
    if m:
        return int(m.group(1)) * (10 ** int(m.group(2)))

    # plain integer
    m = re.fullmatch(r"\\d{1,10}", tok)
    if m:
        return int(tok)

    return None


_RE_N_BOUND = re.compile(
    r"\\b(?:n|m|q|k)\\b\\s*(?:<=|<|≤)\\s*([0-9][0-9_.,]*\\s*(?:e\\s*\\d{1,2}|\\^\\s*\\d{1,2})?)",
    re.IGNORECASE,
)


def extract_n_max(text: str) -> Optional[int]:
    """
    Best-effort extraction of an upper bound for N/M/Q/K.
    Returns the largest value found across common symbols.
    """
    t = text or ""
    vals: list[int] = []
    for m in _RE_N_BOUND.finditer(t):
        raw = m.group(1)
        v = _parse_sci_like(raw)
        if v is not None:
            vals.append(v)
    return max(vals) if vals else None


def bucket_n(n_max: Optional[int]) -> str:
    if n_max is None:
        return "unknown"
    # coarse CP buckets
    if n_max <= 200:
        return "<=200"
    if n_max <= 2000:
        return "<=2e3"
    if n_max <= 20000:
        return "<=2e4"
    if n_max <= 200000:
        return "<=2e5"
    if n_max <= 1000000:
        return "<=1e6"
    return ">1e6"


def detect_multi_testcase(text: str) -> bool:
    t = text or ""
    if _RE_TESTCASES.search(t):
        return True
    # Very common format line: "The first line contains an integer t"
    if re.search(r"first\\s+line\\s+contains\\s+an?\\s+integer\\s+t\\b", t, re.IGNORECASE):
        return True
    return False


def extract_complexity_hint(text: str) -> Optional[str]:
    m = _RE_O_COMPLEXITY.search(text or "")
    if not m:
        return None
    hint = m.group(1).strip()
    # keep it short / stable
    return hint[:40]


@dataclass(frozen=True)
class ConstraintSignature:
    input_type: str
    n_bucket: str
    multi_test: bool
    complexity_hint: Optional[str]

    def to_compact(self) -> str:
        # Complexity hint is optional; include only if present so signatures stay comparable
        base = f"{self.input_type}|{self.n_bucket}|{'T' if self.multi_test else 'F'}"
        return f"{base}|O({self.complexity_hint})" if self.complexity_hint else base


def build_signature(text: str) -> ConstraintSignature:
    input_type = detect_input_type(text)
    n_max = extract_n_max(text)
    n_bucket = bucket_n(n_max)
    multi_test = detect_multi_testcase(text)
    complexity_hint = extract_complexity_hint(text)
    return ConstraintSignature(
        input_type=input_type,
        n_bucket=n_bucket,
        multi_test=multi_test,
        complexity_hint=complexity_hint,
    )


