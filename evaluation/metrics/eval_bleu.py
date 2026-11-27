"""BLEU: n-gram precision of the answer against the reference answer.

BLEU is a surface-overlap metric (1- to 4-gram precision with a brevity
penalty). It is a weak judge of free-form answers on its own, but cheap and
useful as a regression signal next to ROUGE and BERTScore.

Two interchangeable back-ends (``--backend`` or ``evaluation.bleu.backend``):

* ``sacrebleu`` (default) - the standard, reproducible implementation; smoothed
  sentence-level BLEU plus a corpus-level BLEU.
* ``nltk`` - ``nltk.translate.bleu_score`` with Chen & Cherry smoothing (method1)
  and a regex tokenizer, so no NLTK data download is needed.

Scores are 0..1 (``--scale unit``) or 0..100 (``--scale percent``).
Deterministic; no LLM and no API key needed.

Run standalone::

    python metrics/eval_bleu.py --input golden_dataset.json --output results/bleu_report.json
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__:
    from . import eval_common as ec
else:  # executed as a script
    import eval_common as ec

METRIC = "bleu"
BACKENDS = ("sacrebleu", "nltk")
_TOKEN = re.compile(r"\w+|[^\w\s]")   # words and individual punctuation marks


def _sacrebleu_scores(preds: Sequence[str], refs: Sequence[str]) -> tuple[list[float], float]:
    """Return (sentence-level scores, corpus score) on sacrebleu's native 0..100 scale."""
    import sacrebleu

    # smooth_method='exp' avoids zero scores when a short answer has no 4-gram match.
    sentence = [sacrebleu.sentence_bleu(p, [r], smooth_method="exp").score if p.strip() else 0.0
                for p, r in zip(preds, refs)]
    corpus = sacrebleu.corpus_bleu(list(preds), [list(refs)]).score
    return sentence, corpus


def _nltk_scores(preds: Sequence[str], refs: Sequence[str]) -> tuple[list[float], float]:
    """Return (sentence-level scores, corpus score) on NLTK's 0..1 scale, converted to 0..100."""
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu

    smooth = SmoothingFunction().method1       # add-epsilon smoothing for missing n-gram orders
    hyp_tokens = [_TOKEN.findall(p.lower()) for p in preds]
    ref_tokens = [_TOKEN.findall(r.lower()) for r in refs]
    sentence = [100 * sentence_bleu([r], h, smoothing_function=smooth) if h else 0.0
                for h, r in zip(hyp_tokens, ref_tokens)]
    corpus = 100 * corpus_bleu([[r] for r in ref_tokens], hyp_tokens, smoothing_function=smooth)
    return sentence, corpus


def evaluate_bleu(
    predictions: Sequence[str],
    references: Sequence[str],
    questions: Sequence[str] | None = None,
    *,
    config: Mapping[str, Any],
    backend: str | None = None,
    scale: str | None = None,
) -> ec.MetricReport:
    """Compute sentence-level and corpus BLEU for parallel prediction/reference lists.

    Args:
        predictions: system answers (hypotheses).
        references: ground-truth answers.
        questions: optional, only used to make the report readable.
        config: loaded config.yaml dict (defaults for backend/scale).
        backend: ``sacrebleu`` or ``nltk``.
        scale: ``unit`` (0..1) or ``percent`` (0..100).
    """
    n = len(predictions)
    if n != len(references):
        raise ec.EvalError("predictions and references must have the same length")
    bleu_cfg = config.get("evaluation", {}).get("bleu", {})
    backend = backend or bleu_cfg.get("backend", "sacrebleu")
    scale = scale or bleu_cfg.get("scale", "unit")
    if backend not in BACKENDS:
        raise ec.EvalError(f"bleu backend must be one of {BACKENDS}, got '{backend}'")
    if scale not in ("unit", "percent"):
        raise ec.EvalError("bleu scale must be 'unit' or 'percent'")
    qs = questions or [""] * n
    clock = ec.timed()

    sentence, corpus = (_sacrebleu_scores if backend == "sacrebleu" else _nltk_scores)(predictions, references)
    factor = 0.01 if scale == "unit" else 1.0   # both back-ends were normalised to 0..100 above
    rows = [{"index": i, "question": qs[i], "sentence_bleu": s * factor} for i, s in enumerate(sentence)]
    agg = {"corpus_bleu": corpus * factor, "sentence_bleu_mean": ec.mean_of(rows, "sentence_bleu"),
           "count_scored": n}
    meta = ec.base_metadata(config, Path("<in-memory>"), n, clock(), backend=backend, scale=scale)
    return ec.MetricReport(METRIC, "corpus_bleu", rows, agg, meta)


def run(
    input_path: str | None = None,
    output_path: str | None = None,
    config_path: str | None = None,
    limit: int | None = None,
    write_csv: bool = True,
    backend: str | None = None,
    scale: str | None = None,
) -> ec.MetricReport:
    """Load the golden dataset, evaluate, write the JSON/CSV report and return it."""
    cfg = ec.load_config(config_path)
    inp, out = ec.resolve_io(cfg, METRIC, input_path, output_path)
    samples = ec.load_golden_dataset(inp, limit)
    report = evaluate_bleu([s.answer for s in samples], [s.ground_truth for s in samples],
                           [s.question for s in samples], config=cfg, backend=backend, scale=scale)
    report.metadata["dataset"] = str(inp)
    ec.write_report(report, out, write_csv)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``python metrics/eval_bleu.py --help``)."""
    parser = ec.make_parser(METRIC, "BLEU (sacrebleu or nltk)")
    parser.add_argument("--backend", choices=BACKENDS, default=None, help="BLEU implementation (default: config)")
    parser.add_argument("--scale", choices=("unit", "percent"), default=None, help="report 0..1 or 0..100")
    args = parser.parse_args(argv)
    return ec.run_cli(METRIC, args, run, backend=args.backend, scale=args.scale)


if __name__ == "__main__":
    sys.exit(main())
