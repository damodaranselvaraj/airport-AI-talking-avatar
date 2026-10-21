"""Calibrate the strong/medium evidence thresholds from a committed question set.

Runs real retrieval (OpenAI embeddings + Chroma + BM25 + Cohere rerank) over
  * the in-scope gold set  evaluation/question_set/in_scope.json  (you supply it), and
  * the must-refuse traps  data/unknown_traps.json,
prints the top rerank-score distribution for each, and suggests thresholds that
separate them (precision first: refusing a real question is better than
answering an unsupported one).

    python scripts/calibrate_evidence.py                 # report only
    python scripts/calibrate_evidence.py --write         # also write config/evidence_thresholds.json

Gold-set format: see evaluation/question_set/README.md.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
MIN_IN_SCOPE_TO_WRITE = 10


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile (pct in 0..100) of a non-empty sequence."""
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def suggest_thresholds(in_scores: Sequence[float], trap_scores: Sequence[float]) -> dict:
    """Pick (strong, medium) from the two score distributions.

    strong = 10th percentile of in-scope top scores (>= 90% of real questions pass),
             raised above the best trap score if the distributions overlap.
    medium = between the median trap score and half of strong.
    """
    if not in_scores:
        raise ValueError("need at least one in-scope score")
    strong = percentile(in_scores, 10)
    max_trap = max(trap_scores) if trap_scores else 0.0
    overlap = bool(trap_scores) and strong <= max_trap
    if overlap:
        strong = max_trap + 0.01
    medium = max(statistics.median(trap_scores) if trap_scores else 0.0, strong * 0.5)
    strong, medium = round(min(strong, 1.0), 2), round(min(medium, strong), 2)
    coverage = sum(1 for s in in_scores if s >= strong) / len(in_scores)
    return {
        "strong": strong,
        "medium": medium,
        "overlap": overlap,
        "in_scope_coverage": round(coverage, 3),
        "max_trap_score": round(max_trap, 3),
    }


def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["questions"] if isinstance(data, dict) else data
    for item in items:
        if "question" not in item:
            raise ValueError(f"gold-set entry missing 'question': {item!r}")
    return items


def top_scores(
    retrieve: Callable[[str, int], tuple[list, Optional[object]]],
    questions: Sequence[dict],
    key: str,
    top_k: int,
) -> list[dict]:
    """Retrieve each question; record top score and whether expected_source was found."""
    rows = []
    for q in questions:
        passages, error = retrieve(q[key], top_k)
        if error is not None:
            raise RuntimeError("Cohere rerank failed; scores are not on the rerank scale. Retry.")
        expected = q.get("expected_source")
        sources = [p.source for p in passages]
        rows.append(
            {
                "id": q.get("id", q[key][:30]),
                "top": max((p.score for p in passages), default=0.0),
                "hit": None if not expected else any(expected in s for s in sources),
            }
        )
    return rows


def _print_rows(title: str, rows: list[dict]) -> None:
    print(f"\n== {title} ({len(rows)}) ==")
    for r in sorted(rows, key=lambda r: r["top"]):
        hit = "" if r["hit"] is None else ("  source:OK" if r["hit"] else "  source:MISS")
        print(f"  {r['top']:.3f}  {r['id']}{hit}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default=str(ROOT / "evaluation" / "question_set" / "in_scope.json"))
    ap.add_argument("--traps", default=str(ROOT / "data" / "unknown_traps.json"))
    ap.add_argument("--output", default=str(ROOT / "config" / "evidence_thresholds.json"))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--write", action="store_true", help="write the suggested thresholds (calibrated=true)")
    args = ap.parse_args(argv)

    q_path = Path(args.questions)
    if not q_path.is_file():
        print(f"No gold set at {q_path}.\nCreate it first; format in evaluation/question_set/README.md.", file=sys.stderr)
        return 2

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from config.settings import get_settings
    from app.retrieval.retrievers import HybridRetriever

    retriever = HybridRetriever.from_settings(get_settings())
    gold = load_questions(q_path)
    traps = [
        {"id": t["id"], "question": t["question_en"]}
        for t in json.loads(Path(args.traps).read_text(encoding="utf-8"))["items"]
    ]
    in_rows = top_scores(retriever.retrieve_with_status, gold, "question", args.top_k)
    trap_rows = top_scores(retriever.retrieve_with_status, traps, "question", args.top_k)
    _print_rows("In-scope gold set", in_rows)
    _print_rows("Must-refuse traps", trap_rows)

    suggestion = suggest_thresholds([r["top"] for r in in_rows], [r["top"] for r in trap_rows])
    misses = [r["id"] for r in in_rows if r["hit"] is False]
    print(f"\nSuggested: strong={suggestion['strong']}  medium={suggestion['medium']}")
    print(f"In-scope questions at STRONG: {suggestion['in_scope_coverage']:.0%}; best trap score: {suggestion['max_trap_score']}")
    if suggestion["overlap"]:
        print("WARNING: in-scope and trap scores overlap; strong was raised above the best trap.")
    if misses:
        print(f"WARNING: expected source not retrieved for: {', '.join(misses)} (a retrieval problem, not a threshold one)")

    if not args.write:
        print("\n(report only; pass --write to save)")
        return 0
    if len(in_rows) < MIN_IN_SCOPE_TO_WRITE:
        print(f"Refusing to write: need >= {MIN_IN_SCOPE_TO_WRITE} in-scope questions (have {len(in_rows)}).", file=sys.stderr)
        return 3
    payload = {
        "version": 1,
        "calibrated": True,
        "strong": suggestion["strong"],
        "medium": suggestion["medium"],
        "notes": f"Calibrated from {len(in_rows)} in-scope questions and {len(trap_rows)} traps by scripts/calibrate_evidence.py.",
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
