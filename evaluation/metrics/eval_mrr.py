"""Mean Reciprocal Rank (MRR) and Hit-Rate@K.

MRR answers: "how high does the *first* relevant chunk appear?"
For each question the reciprocal rank is ``1 / rank_of_first_relevant_context``
(0 when no retrieved chunk is relevant); MRR is the mean over questions.

Relevance labels come from overlap with the ground-truth chunks (or the
reference answer) - see ``evaluation.relevance`` in config.yaml. Fully
deterministic: no LLM and no API key needed.

Run standalone::

    python metrics/eval_mrr.py --input golden_dataset.json --output results/mrr_report.json
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__:
    from . import eval_common as ec
else:  # executed as a script
    import eval_common as ec

METRIC = "mrr"


def reciprocal_rank(labels: Sequence[int], k: int | None = None) -> tuple[float, int | None]:
    """Return ``(reciprocal_rank, first_relevant_rank)`` for ranked binary ``labels`` (rank is 1-based)."""
    window = list(labels)[:k] if k else list(labels)
    for rank, label in enumerate(window, start=1):
        if label:
            return 1.0 / rank, rank
    return 0.0, None  # nothing relevant was retrieved


def evaluate_mrr(
    contexts: Sequence[Sequence[str]],
    reference_contexts: Sequence[Sequence[str]],
    references: Sequence[str],
    questions: Sequence[str] | None = None,
    *,
    config: Mapping[str, Any],
    k_values: Sequence[int] | None = None,
) -> ec.MetricReport:
    """Compute MRR over parallel lists of retrieved contexts and ground truth.

    Args:
        contexts: per question, retrieved chunks in rank order.
        reference_contexts: per question, ground-truth chunks (may be empty lists).
        references: per question, the reference answer (fallback relevance signal).
        questions: optional question texts (only used to make the report readable).
        config: loaded config.yaml dict.
        k_values: cut-offs for Hit-Rate@K (default from config).
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
        labels = ec.relevance_labels(scores, threshold)
        rr, rank = reciprocal_rank(labels)
        row: dict[str, Any] = {
            "index": i, "question": qs[i], "n_contexts": len(contexts[i]),
            "first_relevant_rank": rank, "reciprocal_rank": rr,
        }
        for k in ks:  # Hit@K = 1 if any relevant chunk is within the top-k
            row[f"hit@{k}"] = float(any(labels[:k]))
        rows.append(row)

    agg: dict[str, Any] = {METRIC: ec.mean_of(rows, "reciprocal_rank"), "count_scored": n}
    for k in ks:
        agg[f"hit_rate@{k}"] = ec.mean_of(rows, f"hit@{k}")
    agg["n_no_relevant"] = sum(1 for r in rows if r["first_relevant_rank"] is None)
    meta = ec.base_metadata(config, Path("<in-memory>"), n, clock(), relevance_mode=mode, relevance_threshold=threshold)
    return ec.MetricReport(METRIC, METRIC, rows, agg, meta)


def run(
    input_path: str | None = None,
    output_path: str | None = None,
    config_path: str | None = None,
    limit: int | None = None,
    write_csv: bool = True,
) -> ec.MetricReport:
    """Load the golden dataset, evaluate, write the JSON/CSV report and return it."""
    cfg = ec.load_config(config_path)
    inp, out = ec.resolve_io(cfg, METRIC, input_path, output_path)
    samples = ec.load_golden_dataset(inp, limit)
    report = evaluate_mrr(
        [s.contexts for s in samples], [s.reference_contexts for s in samples],
        [s.ground_truth for s in samples], [s.question for s in samples], config=cfg,
    )
    report.metadata["dataset"] = str(inp)
    ec.write_report(report, out, write_csv)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``python metrics/eval_mrr.py --help``)."""
    args = ec.make_parser(METRIC, "Mean Reciprocal Rank and Hit-Rate@K").parse_args(argv)
    return ec.run_cli(METRIC, args, run)


if __name__ == "__main__":
    sys.exit(main())
