"""BERTScore: semantic similarity between answer and reference.

Instead of matching exact words, BERTScore embeds every token with a
pre-trained transformer and greedily matches tokens by cosine similarity,
yielding precision, recall and F1. It rewards paraphrases that BLEU/ROUGE miss.

* The model is downloaded from Hugging Face on first use (needs internet once,
  then it is cached). ``distilbert-base-uncased`` keeps it light; set
  ``evaluation.bertscore.model_type`` to ``roberta-large`` for the library default.
* Runs locally on CPU or GPU. No LLM judge and no API key are needed.

Run standalone::

    python metrics/eval_bertscore.py --input golden_dataset.json --output results/bertscore_report.json
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__:
    from . import eval_common as ec
else:  # executed as a script
    import eval_common as ec

METRIC = "bertscore"


def evaluate_bertscore(
    predictions: Sequence[str],
    references: Sequence[str],
    questions: Sequence[str] | None = None,
    *,
    config: Mapping[str, Any],
    model_type: str | None = None,
    lang: str | None = None,
    rescale_with_baseline: bool | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> ec.MetricReport:
    """Compute BERTScore P/R/F1 for parallel prediction/reference lists.

    Args:
        predictions: system answers (candidates).
        references: ground-truth answers.
        questions: optional, only used to make the report readable.
        config: loaded config.yaml dict; the explicit keyword arguments below override it.
        model_type: Hugging Face encoder name (e.g. ``distilbert-base-uncased``).
        lang: language code, needed for baseline rescaling.
        rescale_with_baseline: rescale scores so typical values spread over 0..1
            (requires a baseline file shipped with bert-score for ``model_type``/``lang``).
        batch_size: encoder batch size.
        device: ``cpu`` / ``cuda`` / ``None`` (auto-detect).
    """
    n = len(predictions)
    if n != len(references):
        raise ec.EvalError("predictions and references must have the same length")
    b_cfg = config.get("evaluation", {}).get("bertscore", {})
    model_type = model_type or b_cfg.get("model_type", "distilbert-base-uncased")
    lang = lang or b_cfg.get("lang", "en")
    rescale = bool(b_cfg.get("rescale_with_baseline", False) if rescale_with_baseline is None else rescale_with_baseline)
    batch = int(batch_size or b_cfg.get("batch_size", 16))
    device = device or b_cfg.get("device")
    qs = questions or [""] * n
    clock = ec.timed()

    try:
        from bert_score import score as bert_score_fn  # heavy import (torch + transformers)
    except ImportError as exc:
        raise ec.EvalError(f"bert-score is not installed ({exc}); run: pip install bert-score") from exc

    # Empty candidates/references crash or produce NaN in some encoders, so only
    # score valid pairs and give the rest an explicit 0.0 (and a skipped note).
    valid = [i for i in range(n) if predictions[i].strip() and references[i].strip()]
    p_scores = r_scores = f_scores = []
    if valid:
        logger = ec.logger
        logger.info("Computing BERTScore with %s on %d pairs (first run downloads the model)...", model_type, len(valid))
        precision, recall, f1 = bert_score_fn(
            [predictions[i] for i in valid], [references[i] for i in valid],
            model_type=model_type, lang=lang, rescale_with_baseline=rescale,
            batch_size=batch, device=device, verbose=False,
        )
        p_scores, r_scores, f_scores = precision.tolist(), recall.tolist(), f1.tolist()

    by_index = {i: (p_scores[j], r_scores[j], f_scores[j]) for j, i in enumerate(valid)}
    rows: list[dict[str, Any]] = []
    for i in range(n):
        p, r, f = by_index.get(i, (0.0, 0.0, 0.0))
        row: dict[str, Any] = {"index": i, "question": qs[i], "bertscore_precision": p,
                               "bertscore_recall": r, "bertscore_f1": f}
        if i not in by_index:
            row["skipped"] = "empty prediction or reference"
        rows.append(row)

    agg = {"bertscore_f1": ec.mean_of(rows, "bertscore_f1"), "bertscore_precision": ec.mean_of(rows, "bertscore_precision"),
           "bertscore_recall": ec.mean_of(rows, "bertscore_recall"), "count_scored": len(valid),
           "count_skipped": n - len(valid)}
    meta = ec.base_metadata(config, Path("<in-memory>"), n, clock(), model_type=model_type, lang=lang,
                            rescale_with_baseline=rescale)
    return ec.MetricReport(METRIC, "bertscore_f1", rows, agg, meta)


def run(
    input_path: str | None = None,
    output_path: str | None = None,
    config_path: str | None = None,
    limit: int | None = None,
    write_csv: bool = True,
    model_type: str | None = None,
    rescale_with_baseline: bool | None = None,
    device: str | None = None,
) -> ec.MetricReport:
    """Load the golden dataset, evaluate, write the JSON/CSV report and return it."""
    cfg = ec.load_config(config_path)
    inp, out = ec.resolve_io(cfg, METRIC, input_path, output_path)
    samples = ec.load_golden_dataset(inp, limit)
    report = evaluate_bertscore([s.answer for s in samples], [s.ground_truth for s in samples],
                                [s.question for s in samples], config=cfg, model_type=model_type,
                                rescale_with_baseline=rescale_with_baseline, device=device)
    report.metadata["dataset"] = str(inp)
    ec.write_report(report, out, write_csv)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``python metrics/eval_bertscore.py --help``)."""
    parser = ec.make_parser(METRIC, "BERTScore (bert-score)")
    parser.add_argument("--model-type", default=None, help="Hugging Face encoder (default: config)")
    parser.add_argument("--rescale-with-baseline", action="store_true", default=None, help="rescale scores with the baseline")
    parser.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    args = parser.parse_args(argv)
    return ec.run_cli(METRIC, args, run, model_type=args.model_type,
                      rescale_with_baseline=args.rescale_with_baseline, device=args.device)


if __name__ == "__main__":
    sys.exit(main())
