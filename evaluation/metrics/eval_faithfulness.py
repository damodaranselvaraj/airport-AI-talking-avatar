"""Faithfulness / Groundedness (RAGAS ``Faithfulness``).

This is the framework's hallucination and factual-consistency judge. RAGAS
breaks the answer into atomic statements and asks the judge LLM whether each
statement can be inferred from the *retrieved contexts*::

    faithfulness = supported statements / total statements        (0..1)

A score of 1.0 means every claim in the answer is grounded in the contexts; low
scores flag hallucinated or unsupported content. Because the judge is created
from ``judge_llm`` in config.yaml, switching between OpenAI and Anthropic needs
no code change (``RAG_EVAL_JUDGE_PROVIDER=anthropic``).

Run standalone::

    python metrics/eval_faithfulness.py --input golden_dataset.json \
        --output results/faithfulness_report.json
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__:
    from . import eval_common as ec
else:  # executed as a script
    import eval_common as ec

METRIC = "faithfulness"


def evaluate_faithfulness(
    questions: Sequence[str],
    predictions: Sequence[str],
    contexts: Sequence[Sequence[str]],
    *,
    config: Mapping[str, Any],
    hallucination_threshold: float = 0.8,
) -> ec.MetricReport:
    """Score whether each answer is grounded in its retrieved contexts.

    Args:
        questions: user questions.
        predictions: system answers to check.
        contexts: per question, the retrieved chunks the answer was built from.
        config: loaded config.yaml dict (judge LLM, concurrency, timeout).
        hallucination_threshold: samples scoring below this are flagged ``possible_hallucination``.
    """
    n = len(questions)
    if not (n == len(predictions) == len(contexts)):
        raise ec.EvalError("questions, predictions and contexts must have the same length")
    concurrency, timeout, _ = ec.eval_settings(config)
    clock = ec.timed()

    from ragas.metrics.collections import Faithfulness  # statement extraction + NLI-style verification

    metric = Faithfulness(llm=ec.build_ragas_llm(config))  # judge LLM from config.yaml
    # Nothing to ground an answer in (or no answer) -> skipped, not scored as 0.
    todo = [i for i in range(n) if predictions[i].strip() and contexts[i]]

    async def scorer(i: int) -> float | None:
        return ec.ragas_value(
            await metric.ascore(user_input=questions[i], response=predictions[i], retrieved_contexts=list(contexts[i]))
        )

    by_index = dict(zip(todo, ec.score_async(todo, scorer, concurrency, timeout)))
    rows: list[dict[str, Any]] = []
    results: list[ec.ScoreResult] = []
    for i in range(n):
        res = by_index.get(i) or ec.ScoreResult(None, skipped="empty answer or no retrieved contexts")
        results.append(res)
        row: dict[str, Any] = {"index": i, "question": questions[i], METRIC: res.score,
                               "possible_hallucination": (res.score < hallucination_threshold) if res.score is not None else None}
        if res.error:
            row["error"] = res.error
        if res.skipped:
            row["skipped"] = res.skipped
        rows.append(row)

    stats = ec.summarize(r[METRIC] for r in rows)
    flagged = sum(1 for r in rows if r["possible_hallucination"])
    agg = {METRIC: stats["mean"], f"{METRIC}_std": stats["std"], f"{METRIC}_min": stats["min"],
           "n_possible_hallucination": flagged,
           "hallucination_rate": (flagged / stats["count"]) if stats["count"] else None,
           "count_scored": stats["count"], "count_failed": sum(1 for r in results if r.error),
           "count_skipped": sum(1 for r in results if r.skipped)}
    meta = ec.base_metadata(config, Path("<in-memory>"), n, clock(), hallucination_threshold=hallucination_threshold,
                            judge=config.get("judge_llm", {}).get("model"),
                            judge_provider=config.get("judge_llm", {}).get("provider"))
    return ec.MetricReport(METRIC, METRIC, rows, agg, meta)


def run(
    input_path: str | None = None,
    output_path: str | None = None,
    config_path: str | None = None,
    limit: int | None = None,
    write_csv: bool = True,
    hallucination_threshold: float = 0.8,
) -> ec.MetricReport:
    """Load the golden dataset, evaluate, write the JSON/CSV report and return it."""
    cfg = ec.load_config(config_path)
    inp, out = ec.resolve_io(cfg, METRIC, input_path, output_path)
    samples = ec.load_golden_dataset(inp, limit)
    report = evaluate_faithfulness([s.question for s in samples], [s.answer for s in samples],
                                   [s.contexts for s in samples], config=cfg,
                                   hallucination_threshold=hallucination_threshold)
    report.metadata["dataset"] = str(inp)
    ec.write_report(report, out, write_csv)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``python metrics/eval_faithfulness.py --help``)."""
    parser = ec.make_parser(METRIC, "Faithfulness / groundedness (RAGAS)")
    parser.add_argument("--hallucination-threshold", type=float, default=0.8,
                        help="flag samples scoring below this as possible hallucinations")
    args = parser.parse_args(argv)
    return ec.run_cli(METRIC, args, run, hallucination_threshold=args.hallucination_threshold)


if __name__ == "__main__":
    sys.exit(main())
