#!/usr/bin/env python3
"""Run every evaluation module sequentially and consolidate one summary report.

Each module under ``metrics/`` is imported and its ``run()`` function is called
with the same dataset and config, so a module that fails (missing API key, model
download problem, ...) is recorded in the summary but does not stop the others.

Outputs (in ``--results-dir``):

* ``<metric>_report.json`` / ``.csv``  - the individual module reports
* ``summary_report.json``              - headline + aggregate scores for every metric
* ``summary_report.csv``               - the same scores in long format (metric, score, value)
* ``summary_report.html``              - a self-contained, shareable table
* ``per_sample_scores.csv``            - every numeric per-sample score side by side

Examples::

    python run_all_evaluations.py --input golden_dataset.json
    python run_all_evaluations.py --include mrr ndcg rouge bleu          # only these
    python run_all_evaluations.py --exclude bertscore faithfulness       # all except these
    python run_all_evaluations.py --skip-llm                             # no API key needed
"""
from __future__ import annotations

import argparse
import csv
import html
import importlib
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from metrics import eval_common as ec

logger = logging.getLogger("rag_eval.run_all")


@dataclass(frozen=True)
class MetricSpec:
    """How the orchestrator treats one metric module."""

    name: str    # key used by --include / --exclude and in config.yaml -> metrics
    module: str  # importable module path
    llm: str     # "required" (needs a judge LLM), "optional" (has a --no-llm mode) or "none"


# Execution order follows the project's metric table.
REGISTRY: tuple[MetricSpec, ...] = (
    MetricSpec("context_precision", "metrics.eval_context_precision", "optional"),
    MetricSpec("context_recall", "metrics.eval_context_recall", "optional"),
    MetricSpec("mrr", "metrics.eval_mrr", "none"),
    MetricSpec("ndcg", "metrics.eval_ndcg", "none"),
    MetricSpec("accuracy", "metrics.eval_accuracy", "none"),
    MetricSpec("answer_relevance", "metrics.eval_answer_relevance", "required"),
    MetricSpec("faithfulness", "metrics.eval_faithfulness", "required"),
    MetricSpec("bleu", "metrics.eval_bleu", "none"),
    MetricSpec("rouge", "metrics.eval_rouge", "none"),
    MetricSpec("bertscore", "metrics.eval_bertscore", "none"),
)
_NAMES = [m.name for m in REGISTRY]


def select_metrics(cfg: dict[str, Any], include: Sequence[str] | None, exclude: Sequence[str] | None,
                   skip_llm: bool) -> tuple[list[MetricSpec], list[tuple[str, str]]]:
    """Decide which metrics run. Returns ``(to_run, [(name, reason_skipped), ...])``.

    Precedence: ``--include`` replaces the config flags; ``--exclude`` always
    removes; ``--skip-llm`` drops modules that cannot run without a judge LLM.
    """
    unknown = [n for n in list(include or []) + list(exclude or []) if n not in _NAMES]
    if unknown:
        raise ec.EvalError(f"Unknown metric(s): {', '.join(unknown)}. Available: {', '.join(_NAMES)}")
    flags = cfg.get("metrics") or {}
    run: list[MetricSpec] = []
    skipped: list[tuple[str, str]] = []
    for spec in REGISTRY:
        enabled = spec.name in include if include else bool(flags.get(spec.name, True))
        if not enabled:
            skipped.append((spec.name, "disabled in config" if not include else "not in --include"))
        elif exclude and spec.name in exclude:
            skipped.append((spec.name, "excluded by --exclude"))
        elif skip_llm and spec.llm == "required":
            skipped.append((spec.name, "needs a judge LLM (--skip-llm)"))
        else:
            run.append(spec)
    return run, skipped


def run_metric(spec: MetricSpec, dataset: Path, results_dir: Path, config_path: str | None,
               limit: int | None, skip_llm: bool) -> dict[str, Any]:
    """Run one module and return a summary row (never raises)."""
    clock = ec.timed()
    out = results_dir / f"{spec.name}_report.json"
    row: dict[str, Any] = {"metric": spec.name, "status": "ok", "report": str(out), "error": None}
    logger.info("▶ %s", spec.name)
    try:
        module = importlib.import_module(spec.module)
        kwargs: dict[str, Any] = {}
        if spec.llm == "optional" and skip_llm:
            kwargs["no_llm"] = True      # run only the deterministic @K scores
        report = module.run(input_path=str(dataset), output_path=str(out), config_path=config_path,
                            limit=limit, **kwargs)
        row.update(headline=report.headline, headline_value=report.headline_value,
                   aggregate=report.aggregate, n_samples=report.metadata.get("n_samples"))
        if report.aggregate.get("count_scored") == 0:
            row["status"], row["error"] = "failed", "no sample could be scored (see the report's per-sample errors)"
    except ec.EvalError as exc:
        row.update(status="failed", error=ec.scrub(str(exc)))
    except ImportError as exc:
        row.update(status="failed", error=f"missing dependency: {exc}")
    except Exception as exc:  # noqa: BLE001 - isolate one module's crash from the rest of the run
        logger.exception("Metric %s crashed", spec.name)
        row.update(status="failed", error=ec.scrub(f"{type(exc).__name__}: {exc}"))
    row["duration_seconds"] = clock()
    level = logging.INFO if row["status"] == "ok" else logging.WARNING
    logger.log(level, "  %s -> %s%s", spec.name, row["status"], f" ({row['error']})" if row["error"] else "")
    return row


# --------------------------------------------------------------------------- writers
def _numeric_aggregates(row: dict[str, Any]) -> dict[str, float]:
    """The numeric entries of a row's aggregate block (what appears in the long-format table)."""
    return {k: v for k, v in (row.get("aggregate") or {}).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Long-format CSV: one line per (metric, score)."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "score", "value", "is_headline", "status"])
        for r in rows:
            scores = _numeric_aggregates(r)
            if not scores:
                w.writerow([r["metric"], "", "", "", r["status"]])
            for name, value in scores.items():
                w.writerow([r["metric"], name, value, name == r.get("headline"), r["status"]])


def write_summary_html(summary: dict[str, Any], path: Path) -> None:
    """Self-contained HTML (no external assets) with a headline table and a detail table."""
    esc = html.escape

    def fmt(v: Any) -> str:
        return f"{v:.4f}" if isinstance(v, float) else esc(str(v))

    head_rows, detail_rows = [], []
    for r in summary["metrics"]:
        ok = r["status"] == "ok"
        value = r.get("headline_value")
        head_rows.append(
            f"<tr class='{'ok' if ok else 'bad'}'><td>{esc(r['metric'])}</td><td>{esc(str(r.get('headline', '')))}</td>"
            f"<td class='num'>{fmt(value) if value is not None else '–'}</td><td>{esc(r['status'])}</td>"
            f"<td class='num'>{r['duration_seconds']}</td><td>{esc(r['error'] or '')}</td></tr>"
        )
        for name, v in _numeric_aggregates(r).items():
            detail_rows.append(f"<tr><td>{esc(r['metric'])}</td><td>{esc(name)}</td><td class='num'>{fmt(v)}</td></tr>")
    skipped = "".join(f"<li>{esc(n)} – {esc(why)}</li>" for n, why in summary["skipped"]) or "<li>none</li>"
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>RAG evaluation summary</title>
<style>
 :root{{color-scheme:light dark;--bg:#fff;--fg:#1d2330;--line:#d9dde5;--head:#eef1f6;--bad:#fdecea}}
 @media (prefers-color-scheme:dark){{:root{{--bg:#14171f;--fg:#e6e9f0;--line:#2c3342;--head:#1d2230;--bad:#3a1f1f}}}}
 body{{font:15px/1.5 system-ui,sans-serif;margin:2rem auto;max-width:960px;padding:0 1rem;background:var(--bg);color:var(--fg)}}
 table{{border-collapse:collapse;width:100%;margin:1rem 0 2rem}}
 th,td{{border:1px solid var(--line);padding:.4rem .6rem;text-align:left}} th{{background:var(--head)}}
 td.num{{text-align:right;font-variant-numeric:tabular-nums}} tr.bad{{background:var(--bad)}}
</style></head><body>
<h1>RAG evaluation summary</h1>
<p>Run <b>{esc(summary['run_id'])}</b> · dataset <code>{esc(summary['dataset'])}</code> · {summary['n_samples']} samples ·
judge: {esc(str(summary['judge']))}</p>
<h2>Headline scores</h2>
<table><tr><th>Metric</th><th>Headline score</th><th>Value</th><th>Status</th><th>Seconds</th><th>Note</th></tr>
{''.join(head_rows)}</table>
<h2>All aggregate scores</h2>
<table><tr><th>Metric</th><th>Score</th><th>Value</th></tr>{''.join(detail_rows)}</table>
<h2>Skipped</h2><ul>{skipped}</ul></body></html>"""
    path.write_text(doc, encoding="utf-8")


def write_per_sample_csv(results_dir: Path, rows: list[dict[str, Any]], path: Path) -> None:
    """Join every module's numeric per-sample scores on the sample index."""
    merged: dict[int, dict[str, Any]] = {}
    ignore = {"index", "n_contexts", "n_reference_contexts"}
    for r in rows:
        if r["status"] != "ok":
            continue
        report = json.loads(Path(r["report"]).read_text(encoding="utf-8"))
        for sample in report.get("per_sample", []):
            line = merged.setdefault(sample["index"], {"index": sample["index"], "question": sample.get("question", "")})
            for key, val in sample.items():
                if key not in ignore and isinstance(val, (int, float)) and not isinstance(val, bool):
                    line[f"{r['metric']}.{key}"] = val
    if not merged:
        return
    cols: list[str] = []
    for line in merged.values():
        cols += [c for c in line if c not in cols]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(merged[i] for i in sorted(merged))


# --------------------------------------------------------------------------- main
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line interface of the orchestrator."""
    p = argparse.ArgumentParser(description="Run all RAG evaluation modules and consolidate the results.")
    p.add_argument("--input", "-i", default=None, help="golden dataset (default: paths.golden_dataset in config)")
    p.add_argument("--config", "-c", default=None, help="config.yaml path")
    p.add_argument("--results-dir", default=None, help="output folder (default: paths.results_dir in config)")
    p.add_argument("--include", nargs="+", metavar="METRIC", help="run only these metrics")
    p.add_argument("--exclude", nargs="+", metavar="METRIC", help="run everything except these metrics")
    p.add_argument("--skip-llm", action="store_true",
                   help="no judge LLM: skip faithfulness/answer_relevance, run only @K scores of precision/recall")
    p.add_argument("--formats", nargs="+", choices=["json", "csv", "html"], default=["json", "csv", "html"],
                   help="summary formats to write")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N samples")
    p.add_argument("--fail-on-error", action="store_true", help="exit with status 1 if any metric failed")
    p.add_argument("--list", action="store_true", help="list available metrics and exit")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected metrics and write the consolidated reports."""
    args = parse_args(argv)
    ec.setup_logging(args.log_level)
    if args.list:
        for spec in REGISTRY:
            print(f"{spec.name:<20} judge LLM: {spec.llm}")
        return 0
    try:
        cfg = ec.load_config(args.config)
        dataset, _ = ec.resolve_io(cfg, "summary", args.input, None)
        results_dir = Path(args.results_dir) if args.results_dir else ec.resolve_path(cfg, ec.cfg_get(cfg, "paths.results_dir", "results"))
        results_dir.mkdir(parents=True, exist_ok=True)
        samples = ec.load_golden_dataset(dataset, args.limit)   # fail fast on a bad dataset
        to_run, skipped = select_metrics(cfg, args.include, args.exclude, args.skip_llm)
    except ec.EvalError as exc:
        logger.error("%s", exc)
        return 2
    if not to_run:
        logger.error("No metrics selected. Skipped: %s", skipped)
        return 2

    rows = [run_metric(s, dataset, results_dir, args.config, args.limit, args.skip_llm) for s in to_run]

    now = datetime.now(timezone.utc)
    judge = ec.cfg_get(cfg, "judge_llm", {})
    summary = {
        "run_id": now.strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": now.isoformat(),
        "dataset": str(dataset),
        "n_samples": len(samples),
        "judge": f"{judge.get('provider')}/{judge.get('model')}" if not args.skip_llm else "not used (--skip-llm)",
        "metrics": rows,
        "skipped": skipped,
    }
    base = results_dir / "summary_report"
    if "json" in args.formats:
        base.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if "csv" in args.formats:
        write_summary_csv(rows, base.with_suffix(".csv"))
        write_per_sample_csv(results_dir, rows, results_dir / "per_sample_scores.csv")
    if "html" in args.formats:
        write_summary_html(summary, base.with_suffix(".html"))

    print("\n" + "=" * 72 + f"\nSummary ({len(samples)} samples) -> {results_dir}\n" + "=" * 72)
    for r in rows:
        v = r.get("headline_value")
        print(f"{r['metric']:<20} {r.get('headline', ''):<22} {('%.4f' % v) if v is not None else '   -  '}   [{r['status']}]"
              + (f"  {r['error']}" if r["error"] else ""))
    failed = [r["metric"] for r in rows if r["status"] != "ok"]
    if failed:
        logger.warning("%d metric(s) failed: %s", len(failed), ", ".join(failed))
    return 1 if (failed and args.fail_on_error) else 0


if __name__ == "__main__":
    sys.exit(main())
