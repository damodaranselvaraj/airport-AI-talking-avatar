"""ROUGE: recall-oriented n-gram / longest-common-subsequence overlap.

Uses Google's ``rouge-score`` package. For every sample and every requested
variant it reports precision, recall and F-measure:

* ``rouge1`` / ``rouge2`` - unigram / bigram overlap
* ``rougeL``              - longest common subsequence (sentence level)
* ``rougeLsum``           - LCS at summary level (newline separated sentences)

Deterministic; no LLM and no API key needed.

Run standalone::

    python metrics/eval_rouge.py --input golden_dataset.json --output results/rouge_report.json
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__:
    from . import eval_common as ec
else:  # executed as a script
    import eval_common as ec

METRIC = "rouge"
VALID_TYPES = ("rouge1", "rouge2", "rouge3", "rougeL", "rougeLsum")


def evaluate_rouge(
    predictions: Sequence[str],
    references: Sequence[str],
    questions: Sequence[str] | None = None,
    *,
    config: Mapping[str, Any],
    rouge_types: Sequence[str] | None = None,
    use_stemmer: bool | None = None,
) -> ec.MetricReport:
    """Compute ROUGE for parallel prediction/reference lists.

    Args:
        predictions: system answers (candidates).
        references: ground-truth answers.
        questions: optional, only used to make the report readable.
        config: loaded config.yaml dict (defaults for types / stemming).
        rouge_types: subset of :data:`VALID_TYPES`.
        use_stemmer: apply Porter stemming before matching (default from config).
    """
    from rouge_score import rouge_scorer

    n = len(predictions)
    if n != len(references):
        raise ec.EvalError("predictions and references must have the same length")
    r_cfg = config.get("evaluation", {}).get("rouge", {})
    types = list(rouge_types or r_cfg.get("types", ["rouge1", "rouge2", "rougeL"]))
    stem = bool(r_cfg.get("use_stemmer", True) if use_stemmer is None else use_stemmer)
    bad = [t for t in types if t not in VALID_TYPES]
    if bad:
        raise ec.EvalError(f"Unsupported ROUGE type(s) {bad}; choose from {VALID_TYPES}")
    qs = questions or [""] * n
    clock = ec.timed()

    scorer = rouge_scorer.RougeScorer(types, use_stemmer=stem)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        row: dict[str, Any] = {"index": i, "question": qs[i]}
        # score(target, prediction): rouge-score's argument order is (reference, candidate).
        result = scorer.score(references[i], predictions[i]) if predictions[i].strip() else {}
        for t in types:
            s = result.get(t)
            row[f"{t}_precision"] = s.precision if s else 0.0
            row[f"{t}_recall"] = s.recall if s else 0.0
            row[f"{t}_f1"] = s.fmeasure if s else 0.0
        rows.append(row)

    agg: dict[str, Any] = {}
    for t in types:
        for part in ("precision", "recall", "f1"):
            agg[f"{t}_{part}"] = ec.mean_of(rows, f"{t}_{part}")
    agg["count_scored"] = n
    headline = "rougeL_f1" if "rougeL" in types else f"{types[0]}_f1"
    meta = ec.base_metadata(config, Path("<in-memory>"), n, clock(), rouge_types=types, use_stemmer=stem)
    return ec.MetricReport(METRIC, headline, rows, agg, meta)


def run(
    input_path: str | None = None,
    output_path: str | None = None,
    config_path: str | None = None,
    limit: int | None = None,
    write_csv: bool = True,
    rouge_types: Sequence[str] | None = None,
    use_stemmer: bool | None = None,
) -> ec.MetricReport:
    """Load the golden dataset, evaluate, write the JSON/CSV report and return it."""
    cfg = ec.load_config(config_path)
    inp, out = ec.resolve_io(cfg, METRIC, input_path, output_path)
    samples = ec.load_golden_dataset(inp, limit)
    report = evaluate_rouge([s.answer for s in samples], [s.ground_truth for s in samples],
                            [s.question for s in samples], config=cfg,
                            rouge_types=rouge_types, use_stemmer=use_stemmer)
    report.metadata["dataset"] = str(inp)
    ec.write_report(report, out, write_csv)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``python metrics/eval_rouge.py --help``)."""
    parser = ec.make_parser(METRIC, "ROUGE (rouge-score)")
    parser.add_argument("--types", nargs="+", choices=VALID_TYPES, default=None, help="ROUGE variants (default: config)")
    parser.add_argument("--no-stemmer", action="store_true", help="disable Porter stemming")
    args = parser.parse_args(argv)
    return ec.run_cli(METRIC, args, run, rouge_types=args.types, use_stemmer=False if args.no_stemmer else None)


if __name__ == "__main__":
    sys.exit(main())
