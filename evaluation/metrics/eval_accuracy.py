"""Answer Accuracy: is the system answer *correct* with respect to the reference?

Every sample is classified as correct (1) / incorrect (0); accuracy is the share
of correct samples, computed with ``sklearn.metrics.accuracy_score``. Three
correctness rules are reported for every sample and one of them decides the
headline accuracy (``--mode`` or ``evaluation.accuracy.mode`` in config.yaml):

* ``exact_match`` - normalised prediction equals the normalised reference (SQuAD style).
* ``contains``    - the normalised reference occurs inside the normalised prediction
                    (forgiving for verbose answers that still state the fact).
* ``f1``          - token-level F1 between prediction and reference is >= a threshold
                    (the default; robust for free-form generated answers).

Deterministic, no LLM and no API key needed.

Run standalone::

    python metrics/eval_accuracy.py --input golden_dataset.json --output results/accuracy_report.json
"""
from __future__ import annotations

import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from sklearn.metrics import accuracy_score  # accuracy = mean(prediction == target)

if __package__:
    from . import eval_common as ec
else:  # executed as a script
    import eval_common as ec

METRIC = "accuracy"
MODES = ("exact_match", "contains", "f1")
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: lower-case, drop punctuation and articles, collapse spaces."""
    text = text.lower().translate(_PUNCT)
    return " ".join(_ARTICLES.sub(" ", text).split())


def token_f1(prediction: str, reference: str) -> float:
    """Harmonic mean of token precision and recall between two normalised strings."""
    pred, ref = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not pred or not ref:
        return float(pred == ref)       # both empty -> perfect, one empty -> 0
    common = sum((Counter(pred) & Counter(ref)).values())   # multiset intersection size
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def evaluate_accuracy(
    predictions: Sequence[str],
    references: Sequence[str],
    questions: Sequence[str] | None = None,
    *,
    config: Mapping[str, Any],
    mode: str | None = None,
    f1_threshold: float | None = None,
) -> ec.MetricReport:
    """Compute answer accuracy for parallel lists of predictions and reference answers.

    Args:
        predictions: system answers.
        references: ground-truth answers.
        questions: optional, only used to make the report readable.
        config: loaded config.yaml dict (supplies defaults for ``mode`` / ``f1_threshold``).
        mode: ``exact_match`` | ``contains`` | ``f1`` (headline rule).
        f1_threshold: token-F1 needed to count as correct when mode is ``f1``.
    """
    n = len(predictions)
    if n != len(references):
        raise ec.EvalError("predictions and references must have the same length")
    acc_cfg = config.get("evaluation", {}).get("accuracy", {})
    mode = mode or acc_cfg.get("mode", "f1")
    threshold = float(f1_threshold if f1_threshold is not None else acc_cfg.get("f1_threshold", 0.5))
    if mode not in MODES:
        raise ec.EvalError(f"accuracy mode must be one of {MODES}, got '{mode}'")
    qs = questions or [""] * n
    clock = ec.timed()

    rows: list[dict[str, Any]] = []
    for i in range(n):
        pred_n, ref_n = normalize_answer(predictions[i]), normalize_answer(references[i])
        em = int(bool(ref_n) and pred_n == ref_n)
        contains = int(bool(ref_n) and ref_n in pred_n)
        f1 = token_f1(predictions[i], references[i])
        decision = {"exact_match": em, "contains": contains, "f1": int(f1 >= threshold)}[mode]
        rows.append({"index": i, "question": qs[i], "exact_match": em, "contains": contains,
                     "token_f1": f1, "correct": decision})

    # sklearn: y_true is all-ones ("every answer should be correct"), y_pred is our verdicts.
    accuracy = float(accuracy_score([1] * n, [r["correct"] for r in rows])) if n else None
    agg: dict[str, Any] = {
        METRIC: accuracy,
        "exact_match_rate": ec.mean_of(rows, "exact_match"),
        "contains_rate": ec.mean_of(rows, "contains"),
        "token_f1_mean": ec.mean_of(rows, "token_f1"),
        "n_correct": sum(r["correct"] for r in rows),
        "count_scored": n,
    }
    meta = ec.base_metadata(config, Path("<in-memory>"), n, clock(), decision_rule=mode, f1_threshold=threshold)
    return ec.MetricReport(METRIC, METRIC, rows, agg, meta)


def run(
    input_path: str | None = None,
    output_path: str | None = None,
    config_path: str | None = None,
    limit: int | None = None,
    write_csv: bool = True,
    mode: str | None = None,
    f1_threshold: float | None = None,
) -> ec.MetricReport:
    """Load the golden dataset, evaluate, write the JSON/CSV report and return it."""
    cfg = ec.load_config(config_path)
    inp, out = ec.resolve_io(cfg, METRIC, input_path, output_path)
    samples = ec.load_golden_dataset(inp, limit)
    report = evaluate_accuracy(
        [s.answer for s in samples], [s.ground_truth for s in samples], [s.question for s in samples],
        config=cfg, mode=mode, f1_threshold=f1_threshold,
    )
    report.metadata["dataset"] = str(inp)
    ec.write_report(report, out, write_csv)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``python metrics/eval_accuracy.py --help``)."""
    parser = ec.make_parser(METRIC, "Answer accuracy (exact match / contains / token-F1)")
    parser.add_argument("--mode", choices=MODES, default=None, help="headline correctness rule (default: config)")
    parser.add_argument("--f1-threshold", type=float, default=None, help="token-F1 needed to be correct in f1 mode")
    args = parser.parse_args(argv)
    return ec.run_cli(METRIC, args, run, mode=args.mode, f1_threshold=args.f1_threshold)


if __name__ == "__main__":
    sys.exit(main())
