"""Context Recall and Recall@K.

"Did the retriever bring back everything needed to answer?"

* **context_recall** (RAGAS, LLM-judged): the reference answer is split into
  statements and an LLM checks which statements can be attributed to the
  retrieved contexts; the score is the attributable share. Needs a judge LLM.
* **recall@K** (no LLM): share of the ground-truth chunks (``reference_contexts``)
  that are covered by at least one of the top-K retrieved chunks. When a sample
  has no ground-truth chunks, the fraction of the reference answer's content
  words found in the top-K contexts is used instead.

Run standalone::

    python metrics/eval_context_recall.py --input golden_dataset.json \
        --output results/context_recall_report.json [--no-llm]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__:
    from . import eval_common as ec
else:  # executed as a script
    import eval_common as ec

METRIC = "context_recall"


def recall_at_k(
    contexts: Sequence[str],
    reference_contexts: Sequence[str],
    reference_answer: str,
    k: int,
    threshold: float,
) -> float:
    """Recall of the top-``k`` retrieved contexts against ground truth (0..1)."""
    top = list(contexts)[:k]
    if not top:
        return 0.0
    if reference_contexts:
        # A ground-truth chunk is "found" if any retrieved chunk overlaps it enough.
        found = sum(1 for ref in reference_contexts if any(ec.chunk_overlap(c, ref) >= threshold for c in top))
        return found / len(reference_contexts)
    # Fallback: coverage of the reference answer's content words by the union of the top-k contexts.
    gt = set(ec.content_tokens(reference_answer))
    if not gt:
        return 0.0
    seen: set[str] = set()
    for c in top:
        seen |= set(ec.content_tokens(c))
    return len(gt & seen) / len(gt)


def evaluate_context_recall(
    questions: Sequence[str],
    contexts: Sequence[Sequence[str]],
    references: Sequence[str],
    reference_contexts: Sequence[Sequence[str]] | None = None,
    *,
    config: Mapping[str, Any],
    k_values: Sequence[int] | None = None,
    use_llm: bool = True,
) -> ec.MetricReport:
    """Score retrieval recall for parallel lists of questions / retrieved contexts / reference answers.

    Args mirror :func:`eval_context_precision.evaluate_context_precision`.
    """
    n = len(questions)
    if not (n == len(contexts) == len(references)):
        raise ec.EvalError("questions, contexts and references must have the same length")
    refs_ctx = reference_contexts or [[] for _ in range(n)]
    concurrency, timeout, default_ks = ec.eval_settings(config)
    ks = sorted(set(k_values or default_ks))
    threshold = float(config.get("evaluation", {}).get("relevance", {}).get("threshold", 0.6))
    clock = ec.timed()

    # ---- 1) deterministic Recall@K ---------------------------------------------------------
    rows: list[dict[str, Any]] = []
    for i in range(n):
        row: dict[str, Any] = {"index": i, "question": questions[i], "n_contexts": len(contexts[i]),
                               "n_reference_contexts": len(refs_ctx[i])}
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(contexts[i], refs_ctx[i], references[i], k, threshold)
        rows.append(row)

    # ---- 2) RAGAS LLM-judged Context Recall -----------------------------------------------------
    results: list[ec.ScoreResult] = []
    if use_llm:
        from ragas.metrics.collections import ContextRecall  # attributes reference statements to contexts

        metric = ContextRecall(llm=ec.build_ragas_llm(config))  # judge LLM from config.yaml
        todo = [i for i in range(n) if contexts[i]]

        async def scorer(i: int) -> float | None:
            return ec.ragas_value(
                await metric.ascore(user_input=questions[i], retrieved_contexts=list(contexts[i]), reference=references[i])
            )

        by_index = dict(zip(todo, ec.score_async(todo, scorer, concurrency, timeout)))
        for i in range(n):
            res = by_index.get(i) or ec.ScoreResult(None, skipped="no retrieved contexts")
            rows[i][METRIC] = res.score
            if res.error:
                rows[i]["error"] = res.error
            if res.skipped:
                rows[i]["skipped"] = res.skipped
            results.append(res)

    # ---- 3) aggregate -------------------------------------------------------------------------
    agg: dict[str, Any] = {}
    if use_llm:
        stats = ec.summarize(r[METRIC] for r in rows)
        agg[METRIC] = stats["mean"]
        agg[f"{METRIC}_std"] = stats["std"]
        agg["count_scored"] = stats["count"]
        agg["count_failed"] = sum(1 for r in results if r.error)
        agg["count_skipped"] = sum(1 for r in results if r.skipped)
    for k in ks:
        agg[f"recall@{k}"] = ec.mean_of(rows, f"recall@{k}")
    if not use_llm:
        agg["count_scored"] = n
    headline = METRIC if use_llm else f"recall@{ks[-1]}"
    meta = ec.base_metadata(config, Path("<in-memory>"), n, clock(), relevance_threshold=threshold, llm_used=use_llm)
    return ec.MetricReport(METRIC, headline, rows, agg, meta)


def run(
    input_path: str | None = None,
    output_path: str | None = None,
    config_path: str | None = None,
    limit: int | None = None,
    write_csv: bool = True,
    no_llm: bool = False,
) -> ec.MetricReport:
    """Load the golden dataset, evaluate, write the JSON/CSV report and return it."""
    cfg = ec.load_config(config_path)
    inp, out = ec.resolve_io(cfg, METRIC, input_path, output_path)
    samples = ec.load_golden_dataset(inp, limit)
    report = evaluate_context_recall(
        [s.question for s in samples], [s.contexts for s in samples], [s.ground_truth for s in samples],
        [s.reference_contexts for s in samples], config=cfg, use_llm=not no_llm,
    )
    report.metadata["dataset"] = str(inp)
    ec.write_report(report, out, write_csv)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``python metrics/eval_context_recall.py --help``)."""
    parser = ec.make_parser(METRIC, "Context Recall (RAGAS) and Recall@K")
    parser.add_argument("--no-llm", action="store_true", help="skip the RAGAS judge; compute only Recall@K")
    args = parser.parse_args(argv)
    return ec.run_cli(METRIC, args, run, no_llm=args.no_llm)


if __name__ == "__main__":
    sys.exit(main())
