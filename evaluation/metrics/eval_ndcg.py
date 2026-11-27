"""Normalized Discounted Cumulative Gain (NDCG@K).

NDCG rewards putting relevant chunks near the top of the ranking::

    DCG@k  = sum_{i=1..k} gain_i / log2(i + 1)
    NDCG@k = DCG@k / IDCG@k        (IDCG = DCG of the best possible ordering)

Gains are binary by default (relevant = 1) or graded (``--graded``: the
overlap score itself, zeroed below the relevance threshold). The computation
uses ``sklearn.metrics.ndcg_score``; the single-chunk case, which sklearn
rejects, is handled explicitly. No LLM and no API key are needed.

Run standalone::

    python metrics/eval_ndcg.py --input golden_dataset.json --output results/ndcg_report.json
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import ndcg_score  # reference DCG/IDCG implementation

if __package__:
    from . import eval_common as ec
else:  # executed as a script
    import eval_common as ec

METRIC = "ndcg"


def ndcg_at_k(gains: Sequence[float], k: int) -> float:
    """NDCG@k for ``gains`` listed in retrieval order (index 0 = rank 1).

    The ideal ordering is the same retrieved set sorted by gain, the usual
    convention when only the retrieved list is judged.
    """
    g = np.asarray(gains, dtype=float)
    if g.size == 0 or not np.any(g > 0):
        return 0.0                      # nothing retrieved, or nothing relevant -> 0
    if g.size == 1:
        return 1.0                      # sklearn needs >= 2 documents; a lone relevant doc is ideal
    # y_score is strictly decreasing, so sklearn ranks documents exactly in retrieval order (no ties).
    predicted_order = np.arange(g.size, 0, -1, dtype=float)
    return float(ndcg_score(g.reshape(1, -1), predicted_order.reshape(1, -1), k=k))


def evaluate_ndcg(
    contexts: Sequence[Sequence[str]],
    reference_contexts: Sequence[Sequence[str]],
    references: Sequence[str],
    questions: Sequence[str] | None = None,
    *,
    config: Mapping[str, Any],
    k_values: Sequence[int] | None = None,
    graded: bool = False,
) -> ec.MetricReport:
    """Compute NDCG@K over parallel lists (same inputs as :func:`eval_mrr.evaluate_mrr`).

    Args:
        graded: use graded gains (overlap score) instead of binary 0/1 gains.
    """
    n = len(contexts)
    if not (n == len(reference_contexts) == len(references)):
        raise ec.EvalError("contexts, reference_contexts and references must have the same length")
    qs = questions or [""] * n
    _, _, default_ks = ec.eval_settings(config)
    ks = sorted(set(k_values or default_ks))
    rel_cfg = config.get("evaluation", {}).get("relevance", {})
    mode, threshold = rel_cfg.get("mode", "auto"), float(rel_cfg.get("threshold", 0.6))
    clock = ec.timed()

    rows: list[dict[str, Any]] = []
    for i in range(n):
        scores = ec.relevance_scores(contexts[i], reference_contexts[i], references[i], mode)
        # Binary gains, or graded gains with sub-threshold noise removed.
        gains = [s if (graded and s >= threshold) else float(s >= threshold) for s in scores]
        row: dict[str, Any] = {"index": i, "question": qs[i], "n_contexts": len(contexts[i]),
                               "gains": [round(g, 4) for g in gains]}
        for k in ks:
            row[f"ndcg@{k}"] = ndcg_at_k(gains, k)
        rows.append(row)

    agg: dict[str, Any] = {f"ndcg@{k}": ec.mean_of(rows, f"ndcg@{k}") for k in ks}
    agg["count_scored"] = n
    headline = f"ndcg@{ks[-1]}"
    meta = ec.base_metadata(config, Path("<in-memory>"), n, clock(), relevance_mode=mode,
                            relevance_threshold=threshold, gains="graded" if graded else "binary")
    return ec.MetricReport(METRIC, headline, rows, agg, meta)


def run(
    input_path: str | None = None,
    output_path: str | None = None,
    config_path: str | None = None,
    limit: int | None = None,
    write_csv: bool = True,
    graded: bool = False,
) -> ec.MetricReport:
    """Load the golden dataset, evaluate, write the JSON/CSV report and return it."""
    cfg = ec.load_config(config_path)
    inp, out = ec.resolve_io(cfg, METRIC, input_path, output_path)
    samples = ec.load_golden_dataset(inp, limit)
    report = evaluate_ndcg(
        [s.contexts for s in samples], [s.reference_contexts for s in samples],
        [s.ground_truth for s in samples], [s.question for s in samples], config=cfg, graded=graded,
    )
    report.metadata["dataset"] = str(inp)
    ec.write_report(report, out, write_csv)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``python metrics/eval_ndcg.py --help``)."""
    parser = ec.make_parser(METRIC, "Normalized Discounted Cumulative Gain (NDCG@K)")
    parser.add_argument("--graded", action="store_true", help="use graded gains (overlap score) instead of binary relevance")
    args = parser.parse_args(argv)
    return ec.run_cli(METRIC, args, run, graded=args.graded)


if __name__ == "__main__":
    sys.exit(main())
