"""Answer Relevance (RAGAS ``AnswerRelevancy``).

"Does the answer actually address the question?" RAGAS asks the judge LLM to
write ``strictness`` questions that the answer would be a good reply to, embeds
them together with the original question, and averages the cosine similarity.
Evasive / non-committal answers ("I can't confirm that...") are scored 0.

It needs two models, both configurable in config.yaml:

* the judge LLM        -> ``judge_llm``  (OpenAI or Anthropic)
* an embedding model   -> ``embeddings`` (OpenAI or a local HuggingFace model;
                          Anthropic has no embeddings API)

Run standalone::

    python metrics/eval_answer_relevance.py --input golden_dataset.json \
        --output results/answer_relevance_report.json
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__:
    from . import eval_common as ec
else:  # executed as a script
    import eval_common as ec

METRIC = "answer_relevance"


def evaluate_answer_relevance(
    questions: Sequence[str],
    predictions: Sequence[str],
    *,
    config: Mapping[str, Any],
    strictness: int | None = None,
) -> ec.MetricReport:
    """Score how relevant each answer is to its question.

    Args:
        questions: user questions.
        predictions: the system's answers (same length as ``questions``).
        config: loaded config.yaml dict (judge LLM, embeddings, concurrency).
        strictness: number of questions RAGAS generates per answer (default from config, 3).
    """
    n = len(questions)
    if n != len(predictions):
        raise ec.EvalError("questions and predictions must have the same length")
    concurrency, timeout, _ = ec.eval_settings(config)
    k = int(strictness or config.get("evaluation", {}).get("answer_relevance", {}).get("strictness", 3))
    clock = ec.timed()

    from ragas.metrics.collections import AnswerRelevancy  # LLM question generation + embedding similarity

    # Judge LLM and embedding model come from config.yaml (provider swap = config change only).
    metric = AnswerRelevancy(llm=ec.build_ragas_llm(config), embeddings=ec.build_ragas_embeddings(config), strictness=k)
    todo = [i for i in range(n) if predictions[i].strip()]  # RAGAS rejects empty responses

    async def scorer(i: int) -> float | None:
        return ec.ragas_value(await metric.ascore(user_input=questions[i], response=predictions[i]))

    by_index = dict(zip(todo, ec.score_async(todo, scorer, concurrency, timeout)))
    rows: list[dict[str, Any]] = []
    results: list[ec.ScoreResult] = []
    for i in range(n):
        res = by_index.get(i) or ec.ScoreResult(None, skipped="empty answer")
        results.append(res)
        row: dict[str, Any] = {"index": i, "question": questions[i], METRIC: res.score}
        if res.error:
            row["error"] = res.error
        if res.skipped:
            row["skipped"] = res.skipped
        rows.append(row)

    stats = ec.summarize(r[METRIC] for r in rows)
    agg = {METRIC: stats["mean"], f"{METRIC}_std": stats["std"], f"{METRIC}_min": stats["min"],
           f"{METRIC}_max": stats["max"], "count_scored": stats["count"],
           "count_failed": sum(1 for r in results if r.error), "count_skipped": sum(1 for r in results if r.skipped)}
    meta = ec.base_metadata(config, Path("<in-memory>"), n, clock(), strictness=k,
                            judge=config.get("judge_llm", {}).get("model"),
                            embeddings=config.get("embeddings", {}).get("model"))
    return ec.MetricReport(METRIC, METRIC, rows, agg, meta)


def run(
    input_path: str | None = None,
    output_path: str | None = None,
    config_path: str | None = None,
    limit: int | None = None,
    write_csv: bool = True,
    strictness: int | None = None,
) -> ec.MetricReport:
    """Load the golden dataset, evaluate, write the JSON/CSV report and return it."""
    cfg = ec.load_config(config_path)
    inp, out = ec.resolve_io(cfg, METRIC, input_path, output_path)
    samples = ec.load_golden_dataset(inp, limit)
    report = evaluate_answer_relevance([s.question for s in samples], [s.answer for s in samples],
                                       config=cfg, strictness=strictness)
    report.metadata["dataset"] = str(inp)
    ec.write_report(report, out, write_csv)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``python metrics/eval_answer_relevance.py --help``)."""
    parser = ec.make_parser(METRIC, "Answer Relevance (RAGAS)")
    parser.add_argument("--strictness", type=int, default=None, help="questions generated per answer (default: config, 3)")
    args = parser.parse_args(argv)
    return ec.run_cli(METRIC, args, run, strictness=args.strictness)


if __name__ == "__main__":
    sys.exit(main())
