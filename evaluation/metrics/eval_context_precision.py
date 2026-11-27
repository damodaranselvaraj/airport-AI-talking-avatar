"""Context Precision and Precision@K.

Two complementary views of "did the retriever rank useful chunks first?":

* **context_precision** (RAGAS, LLM-judged): an LLM decides, for every retrieved
  chunk, whether it was useful for producing the *reference answer*; the score is
  the rank-aware average precision of those verdicts, so relevant chunks ranked
  at the top score higher. Needs a judge LLM (OpenAI or Anthropic).
* **precision@K** (no LLM): share of the top-K retrieved chunks that overlap a
  ground-truth chunk (see ``relevance`` in config.yaml). Cheap and deterministic.

Run standalone::

    python metrics/eval_context_precision.py --input golden_dataset.json \
        --output results/context_precision_report.json [--no-llm]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# Import the shared helpers as a package module (run_all) or a sibling file (standalone).
if __package__:
    from . import eval_common as ec
else:  # executed as a script: python metrics/eval_context_precision.py
    import eval_common as ec

METRIC = "context_precision"


def precision_at_k(labels: Sequence[int], k: int) -> float:
    """Fraction of relevant items among the first ``k`` retrieved (denominator = items actually retrieved)."""
    top = list(labels)[:k]
    return sum(top) / len(top) if top else 0.0


def evaluate_context_precision(
    questions: Sequence[str],
    contexts: Sequence[Sequence[str]],
    references: Sequence[str],
    reference_contexts: Sequence[Sequence[str]] | None = None,
    *,
    config: Mapping[str, Any],
    k_values: Sequence[int] | None = None,
    use_llm: bool = True,
) -> ec.MetricReport:
    """Score retrieval precision for parallel lists of questions / retrieved contexts / reference answers.

    Args:
        questions: user questions.
        contexts: per question, the retrieved chunks in rank order.
        references: per question, the reference (ground-truth) answer.
        reference_contexts: optional per question ground-truth chunks (improves Precision@K labels).
        config: loaded config.yaml dict (judge LLM, relevance settings, concurrency).
        k_values: cut-offs for Precision@K (default from config).
        use_llm: when False only Precision@K is computed (no API key needed).
    """
    n = len(questions)
    if not (n == len(contexts) == len(references)):
        raise ec.EvalError("questions, contexts and references must have the same length")
    refs_ctx = reference_contexts or [[] for _ in range(n)]
    concurrency, timeout, default_ks = ec.eval_settings(config)
    ks = sorted(set(k_values or default_ks))
    rel_cfg = config.get("evaluation", {}).get("relevance", {})
    mode, threshold = rel_cfg.get("mode", "auto"), float(rel_cfg.get("threshold", 0.6))
    clock = ec.timed()

    # ---- 1) deterministic Precision@K from overlap-based relevance labels ----------------
    rows: list[dict[str, Any]] = []
    for i in range(n):
        row: dict[str, Any] = {"index": i, "question": questions[i], "n_contexts": len(contexts[i])}
        scores = ec.relevance_scores(contexts[i], refs_ctx[i], references[i], mode)
        labels = ec.relevance_labels(scores, threshold)
        row["relevance_scores"] = [round(s, 4) for s in scores]
        for k in ks:
            row[f"precision@{k}"] = precision_at_k(labels, k) if contexts[i] else 0.0
        rows.append(row)

    # ---- 2) RAGAS LLM-judged Context Precision ---------------------------------------------
    llm_results: list[ec.ScoreResult] = []
    if use_llm:
        from ragas.metrics.collections import ContextPrecision  # RAGAS metric (reference-based variant)

        # The judge LLM is created from config.yaml -> judge_llm (OpenAI or Anthropic).
        metric = ContextPrecision(llm=ec.build_ragas_llm(config))
        todo = [i for i in range(n) if contexts[i]]  # RAGAS needs at least one retrieved chunk

        async def scorer(i: int) -> float | None:
            # user_input = question, reference = ground-truth answer, retrieved_contexts = ranked chunks.
            return ec.ragas_value(
                await metric.ascore(user_input=questions[i], reference=references[i], retrieved_contexts=list(contexts[i]))
            )

        scored = ec.score_async(todo, scorer, concurrency, timeout)
        by_index = dict(zip(todo, scored))
        for i in range(n):
            res = by_index.get(i) or ec.ScoreResult(None, skipped="no retrieved contexts")
            rows[i][METRIC] = res.score
            if res.error:
                rows[i]["error"] = res.error
            if res.skipped:
                rows[i]["skipped"] = res.skipped
            llm_results.append(res)

    # ---- 3) aggregate -----------------------------------------------------------------------
    agg: dict[str, Any] = {}
    if use_llm:
        stats = ec.summarize(r[METRIC] for r in rows)
        agg[METRIC] = stats["mean"]
        agg[f"{METRIC}_std"] = stats["std"]
        agg["count_scored"] = stats["count"]
        agg["count_failed"] = sum(1 for r in llm_results if r.error)
        agg["count_skipped"] = sum(1 for r in llm_results if r.skipped)
    for k in ks:
        agg[f"precision@{k}"] = ec.mean_of(rows, f"precision@{k}")
    if not use_llm:
        agg["count_scored"] = n
    headline = METRIC if use_llm else f"precision@{ks[-1]}"
    meta = ec.base_metadata(config, Path("<in-memory>"), n, clock(), relevance_mode=mode, relevance_threshold=threshold, llm_used=use_llm)
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
    report = evaluate_context_precision(
        [s.question for s in samples], [s.contexts for s in samples], [s.ground_truth for s in samples],
        [s.reference_contexts for s in samples], config=cfg, use_llm=not no_llm,
    )
    report.metadata["dataset"] = str(inp)
    ec.write_report(report, out, write_csv)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``python metrics/eval_context_precision.py --help``)."""
    parser = ec.make_parser(METRIC, "Context Precision (RAGAS) and Precision@K")
    parser.add_argument("--no-llm", action="store_true", help="skip the RAGAS judge; compute only Precision@K")
    args = parser.parse_args(argv)
    return ec.run_cli(METRIC, args, run, no_llm=args.no_llm)


if __name__ == "__main__":
    sys.exit(main())
