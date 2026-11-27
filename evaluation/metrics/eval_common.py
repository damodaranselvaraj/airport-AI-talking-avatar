"""Shared helpers for every ``metrics/eval_*.py`` module and the orchestrator.

This is the only file the metric modules have in common. It keeps the ten
metric scripts small and consistent by providing:

* configuration loading (YAML + ``${ENV}`` interpolation + env overrides),
* golden-dataset loading with field-alias tolerance,
* rank-metric relevance labelling (which retrieved context counts as "relevant"),
* a uniform report structure written as JSON + CSV,
* an async scoring loop with concurrency, timeouts and per-sample error capture,
* factories that build the RAGAS judge LLM / embeddings for OpenAI or Anthropic.

Heavy third-party imports (ragas, openai, anthropic) happen *inside* the
functions that need them, so the non-LLM metrics run without those packages
configured and without any API key.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Mapping, Sequence

import yaml  # PyYAML: configuration file parser

logger = logging.getLogger("rag_eval")

# Project root = the folder that contains config.yaml (parent of metrics/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# =============================================================================
# Exceptions: one small hierarchy so CLIs can print a clean message and exit.
# =============================================================================
class EvalError(Exception):
    """Base class for expected, user-facing evaluation failures."""


class ConfigError(EvalError):
    """Configuration is missing, malformed or references an unset secret."""


class DatasetError(EvalError):
    """The golden dataset cannot be read or violates the expected schema."""


# =============================================================================
# Logging and secret scrubbing
# =============================================================================
def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once; safe to call repeatedly (e.g. from run_all)."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    # Third-party HTTP clients are very chatty at INFO; keep our own logs readable.
    for noisy in ("httpx", "httpcore", "openai", "anthropic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_KEY_LIKE = re.compile(r"\b(?:sk|key|ant)-[A-Za-z0-9_\-]{8,}")


def scrub(text: str) -> str:
    """Mask secrets in text before it is logged or stored in a report.

    Provider error messages occasionally echo part of an API key, and reports
    are meant to be shared, so every error string goes through this function.
    """
    for name, value in os.environ.items():
        # Any env var that looks like a credential has its exact value masked.
        if value and len(value) >= 8 and re.search(r"(KEY|TOKEN|SECRET)", name.upper()):
            text = text.replace(value, "***")
    return _KEY_LIKE.sub("***", text)


# =============================================================================
# Configuration
# =============================================================================
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def load_env() -> None:
    """Load ``.env`` (if python-dotenv is installed). Real env vars win."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:  # python-dotenv is optional
        logger.debug("python-dotenv not installed; skipping .env loading")


def _interpolate(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` inside string values."""
    if isinstance(value, str):
        return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Read config.yaml, expand env references and apply provider overrides.

    Resolution order for the file: explicit ``path`` > ``$RAG_EVAL_CONFIG`` >
    ``<project>/config.yaml``.
    """
    load_env()
    cfg_path = Path(path or os.environ.get("RAG_EVAL_CONFIG") or DEFAULT_CONFIG_PATH)
    if not cfg_path.is_file():
        raise ConfigError(f"Config file not found: {cfg_path}")
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {cfg_path}: {exc}") from exc
    cfg: dict[str, Any] = _interpolate(raw)

    # Environment overrides make provider swaps a one-liner (no file edit needed).
    overrides = {
        "RAG_EVAL_JUDGE_PROVIDER": ("judge_llm", "provider"),
        "RAG_EVAL_JUDGE_MODEL": ("judge_llm", "model"),
        "RAG_EVAL_EMBEDDING_PROVIDER": ("embeddings", "provider"),
        "RAG_EVAL_EMBEDDING_MODEL": ("embeddings", "model"),
    }
    for env_name, (section, key) in overrides.items():
        if os.environ.get(env_name):
            cfg.setdefault(section, {})[key] = os.environ[env_name]
    cfg["_config_path"] = str(cfg_path)
    # Relative paths in the config are resolved against the config file's folder.
    cfg["_base_dir"] = str(cfg_path.parent.resolve())
    return cfg


def cfg_get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """Fetch ``a.b.c`` from nested dicts, returning ``default`` if any hop is missing."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return default if node is None else node


def resolve_path(cfg: Mapping[str, Any], value: str | os.PathLike[str]) -> Path:
    """Resolve a config path relative to the config file's folder (absolute paths pass through)."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else Path(cfg.get("_base_dir", PROJECT_ROOT)) / p


# =============================================================================
# Golden dataset loading
# =============================================================================
# Canonical field -> accepted aliases (so RAGAS-style exports also load).
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "question": ("question", "user_input", "query"),
    "contexts": ("contexts", "retrieved_contexts", "context"),
    "ground_truth": ("ground_truth", "reference", "ground_truths", "reference_answer"),
    "answer": ("answer", "response", "prediction"),
}


@dataclass
class Sample:
    """One golden-dataset record, normalised to the canonical field names."""

    index: int                                   # position in the dataset (stable sample id)
    question: str
    contexts: list[str]                          # retrieved contexts, in rank order
    ground_truth: str                            # reference answer
    answer: str                                  # system answer under evaluation
    reference_contexts: list[str] = field(default_factory=list)   # ground-truth chunks
    metadata: dict[str, Any] = field(default_factory=dict)        # everything else (type, source, ...)


def _first_present(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    """Return the value of the first alias that exists in ``record`` (else ``None``)."""
    for name in names:
        if name in record:
            return record[name]
    return None


def _as_str_list(value: Any) -> list[str]:
    """Coerce None / str / list into a clean list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(v) for v in value if v is not None and str(v).strip()]


def load_golden_dataset(path: str | os.PathLike[str], limit: int | None = None) -> list[Sample]:
    """Load the golden dataset from ``.json`` (list or ``{"samples": [...]}``) or ``.jsonl``.

    Raises:
        DatasetError: file missing, unparsable, empty, or a record misses a required field.
    """
    p = Path(path)
    if not p.is_file():
        raise DatasetError(f"Golden dataset not found: {p}")
    try:
        if p.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            payload = json.loads(p.read_text(encoding="utf-8"))
            # Accept a bare list or the wrapped export written by the generator.
            records = payload if isinstance(payload, list) else (payload.get("samples") or payload.get("data") or [])
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as exc:
        raise DatasetError(f"Cannot parse {p}: {exc}") from exc
    if not records:
        raise DatasetError(f"{p} contains no samples")

    samples: list[Sample] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise DatasetError(f"Record {i} is not a JSON object")
        q = _first_present(rec, _FIELD_ALIASES["question"])
        gt = _first_present(rec, _FIELD_ALIASES["ground_truth"])
        ans = _first_present(rec, _FIELD_ALIASES["answer"])
        ctx = _first_present(rec, _FIELD_ALIASES["contexts"])
        if isinstance(gt, list):  # legacy RAGAS "ground_truths": [..] -> first reference
            gt = gt[0] if gt else None
        missing = [name for name, v in (("question", q), ("ground_truth", gt), ("answer", ans)) if v is None]
        if missing or ctx is None:
            missing += ["contexts"] if ctx is None else []
            raise DatasetError(f"Record {i} is missing required field(s): {', '.join(missing)}")
        known = {a for aliases in _FIELD_ALIASES.values() for a in aliases} | {"reference_contexts"}
        samples.append(
            Sample(
                index=i,
                question=str(q),
                contexts=_as_str_list(ctx),
                ground_truth=str(gt),
                answer=str(ans),
                reference_contexts=_as_str_list(rec.get("reference_contexts")),
                metadata={k: v for k, v in rec.items() if k not in known},
            )
        )
    if limit:
        samples = samples[: max(limit, 0)]
    logger.info("Loaded %d samples from %s", len(samples), p)
    return samples


# =============================================================================
# Relevance labelling for rank-based metrics (Precision@K, Recall@K, MRR, NDCG)
# =============================================================================
# Tiny stop-word list: enough to stop "the/of/is" from inflating overlap scores.
_STOPWORDS = frozenset(
    "a an the and or but if of to in on at by for from with as is are was were be been it its this "
    "that these those there here you your we our i my he she they them his her their can will would "
    "should may might must not no do does did has have had which what who whom when where how".split()
)
_WORD = re.compile(r"[a-z0-9]+")


def content_tokens(text: str) -> list[str]:
    """Lower-case alphanumeric tokens without stop-words (the unit of overlap scoring)."""
    return [t for t in _WORD.findall(text.lower()) if t not in _STOPWORDS]


def containment(candidate: str, reference: str) -> float:
    """Share of ``candidate``'s distinct content tokens that also occur in ``reference`` (0..1)."""
    cand, ref = set(content_tokens(candidate)), set(content_tokens(reference))
    return len(cand & ref) / len(cand) if cand else 0.0


def chunk_overlap(a: str, b: str) -> float:
    """Boundary-tolerant overlap between two chunks.

    The retriever and the testset generator split documents differently, so a
    retrieved chunk is usually a *sub-span* (or super-set) of a ground-truth
    chunk. Taking the larger containment direction handles both cases; chunks
    shorter than 3 content tokens are ignored to avoid accidental matches.
    """
    if min(len(set(content_tokens(a))), len(set(content_tokens(b)))) < 3:
        return 0.0
    return max(containment(a, b), containment(b, a))


def relevance_scores(
    contexts: Sequence[str],
    reference_contexts: Sequence[str],
    ground_truth: str,
    mode: str = "auto",
) -> list[float]:
    """Score every retrieved context for relevance on a 0..1 scale (rank order preserved).

    Modes: ``reference_contexts`` (overlap with ground-truth chunks),
    ``ground_truth`` (coverage of the reference answer's content words), or
    ``auto`` (reference_contexts when available, otherwise ground_truth).
    """
    use_refs = mode == "reference_contexts" or (mode == "auto" and bool(reference_contexts))
    if use_refs:
        return [max((chunk_overlap(c, r) for r in reference_contexts), default=0.0) for c in contexts]
    gt_tokens = set(content_tokens(ground_truth))
    if not gt_tokens:
        return [0.0 for _ in contexts]
    # Share of the reference answer's content words that appear in the context.
    return [len(gt_tokens & set(content_tokens(c))) / len(gt_tokens) for c in contexts]


def relevance_labels(scores: Sequence[float], threshold: float) -> list[int]:
    """Binarise relevance scores at ``threshold`` (>= counts as relevant)."""
    return [1 if s >= threshold else 0 for s in scores]


def has_relevance_signal(sample: Sample, mode: str = "auto") -> bool:
    """True if the sample carries what ``relevance_scores`` needs for the chosen mode."""
    if mode == "reference_contexts":
        return bool(sample.reference_contexts)
    if mode == "ground_truth":
        return bool(content_tokens(sample.ground_truth))
    return bool(sample.reference_contexts) or bool(content_tokens(sample.ground_truth))


# =============================================================================
# Reports
# =============================================================================
@dataclass
class MetricReport:
    """Uniform result object returned by every module's ``run`` / ``evaluate_*`` function."""

    metric: str                      # machine name, e.g. "mrr"
    headline: str                    # key inside ``aggregate`` that represents the module's main score
    per_sample: list[dict[str, Any]]
    aggregate: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def headline_value(self) -> float | None:
        """Value of the headline aggregate (``None`` if it could not be computed)."""
        value = self.aggregate.get(self.headline)
        return value if isinstance(value, (int, float)) and not math.isnan(value) else None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form written to disk."""
        return {
            "metric": self.metric,
            "headline": {"name": self.headline, "value": self.headline_value},
            "aggregate": self.aggregate,
            "metadata": self.metadata,
            "per_sample": self.per_sample,
        }


def summarize(values: Iterable[float | None]) -> dict[str, float | int | None]:
    """Descriptive statistics over scores; ``None``/NaN entries are counted as ``n_failed``."""
    vals_all = list(values)
    vals = [float(v) for v in vals_all if v is not None and not math.isnan(float(v))]
    out: dict[str, float | int | None] = {"count": len(vals), "n_failed": len(vals_all) - len(vals)}
    if vals:
        out.update(
            mean=statistics.fmean(vals),
            median=statistics.median(vals),
            std=statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            min=min(vals),
            max=max(vals),
        )
    else:
        out.update(mean=None, median=None, std=None, min=None, max=None)
    return out


def mean_of(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    """Mean of ``key`` over rows where it is a real number (``None`` if there are none)."""
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and not math.isnan(r[key])]
    return statistics.fmean(vals) if vals else None


def _csv_cell(value: Any) -> Any:
    """Flatten lists/dicts to JSON so every per-sample row fits one CSV cell."""
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value


def write_report(report: MetricReport, output: str | os.PathLike[str], write_csv: bool = True) -> tuple[Path, Path | None]:
    """Write ``<output>`` (JSON, full report) and a sibling ``.csv`` (per-sample rows)."""
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    csv_path: Path | None = None
    if write_csv and report.per_sample:
        csv_path = out.with_suffix(".csv")
        columns: list[str] = []
        for row in report.per_sample:  # union of keys keeps columns stable across heterogeneous rows
            columns += [k for k in row if k not in columns]
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for row in report.per_sample:
                writer.writerow({k: _csv_cell(row.get(k)) for k in columns})
    logger.info("Wrote %s%s", out, f" and {csv_path.name}" if csv_path else "")
    return out, csv_path


# =============================================================================
# CLI scaffolding shared by every module
# =============================================================================
def make_parser(metric: str, description: str) -> argparse.ArgumentParser:
    """Build the common argparse parser (``--input/--output/--config/--limit/--log-level``)."""
    p = argparse.ArgumentParser(prog=f"eval_{metric}.py", description=description)
    p.add_argument("--input", "-i", default=None, help="golden dataset (.json/.jsonl); default: paths.golden_dataset in config")
    p.add_argument("--output", "-o", default=None, help=f"report path; default: <results_dir>/{metric}_report.json")
    p.add_argument("--config", "-c", default=None, help="path to config.yaml (default: project config.yaml)")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N samples")
    p.add_argument("--no-csv", action="store_true", help="do not write the per-sample CSV next to the JSON report")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def resolve_io(cfg: Mapping[str, Any], metric: str, input_path: str | None, output_path: str | None) -> tuple[Path, Path]:
    """Fill in default input/output paths from the config."""
    inp = Path(input_path) if input_path else resolve_path(cfg, cfg_get(cfg, "paths.golden_dataset", "golden_dataset.json"))
    out = Path(output_path) if output_path else resolve_path(cfg, cfg_get(cfg, "paths.results_dir", "results")) / f"{metric}_report.json"
    return inp, out


def run_cli(metric: str, args: argparse.Namespace, run_fn: Callable[..., MetricReport], **kwargs: Any) -> int:
    """Execute ``run_fn`` for a CLI invocation and convert exceptions to exit codes.

    Exit codes: 0 success, 2 expected user-facing error (config/dataset/missing
    package), 1 unexpected failure, 3 every sample failed to score.
    """
    setup_logging(args.log_level)
    try:
        report = run_fn(
            input_path=args.input, output_path=args.output, config_path=args.config,
            limit=args.limit, write_csv=not args.no_csv, **kwargs,
        )
    except EvalError as exc:
        logger.error("%s", scrub(str(exc)))
        return 2
    except ImportError as exc:
        logger.error("Missing dependency for %s: %s (see requirements.txt)", metric, exc)
        return 2
    except Exception:  # noqa: BLE001 - last-resort guard so the CLI never prints a bare traceback to users
        logger.exception("Unexpected failure while evaluating %s", metric)
        return 1
    print_summary(report)
    scored = report.aggregate.get("count_scored")
    return 3 if scored == 0 else 0


def print_summary(report: MetricReport) -> None:
    """Human-readable console summary of a report's aggregate numbers."""
    print(f"\n=== {report.metric} ===")
    for key, value in report.aggregate.items():
        if isinstance(value, float):
            print(f"  {key:<28} {value:.4f}")
        elif not isinstance(value, (dict, list)):
            print(f"  {key:<28} {value}")


# =============================================================================
# Async scoring loop for LLM-judged (RAGAS) metrics
# =============================================================================
@dataclass
class ScoreResult:
    """Outcome for one sample: a score, or an error / skip reason."""

    score: float | None
    error: str | None = None
    skipped: str | None = None


async def _score_all(
    items: Sequence[Any],
    scorer: Callable[[Any], Awaitable[float | None]],
    concurrency: int,
    timeout: float,
) -> list[ScoreResult]:
    """Run ``scorer`` over ``items`` with bounded concurrency; never raises per item."""
    sem = asyncio.Semaphore(max(1, concurrency))   # bound simultaneous judge-LLM calls (rate limits)
    done = 0

    async def one(item: Any) -> ScoreResult:
        nonlocal done
        async with sem:
            try:
                value = await asyncio.wait_for(scorer(item), timeout=timeout)
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    return ScoreResult(None, error="metric returned no value")
                return ScoreResult(float(value))
            except Exception as exc:  # noqa: BLE001 - one bad sample must not abort the run
                return ScoreResult(None, error=scrub(f"{type(exc).__name__}: {exc}")[:500])
            finally:
                done += 1
                if done % 5 == 0 or done == len(items):
                    logger.info("  scored %d/%d", done, len(items))

    return list(await asyncio.gather(*(one(it) for it in items)))


def score_async(
    items: Sequence[Any],
    scorer: Callable[[Any], Awaitable[float | None]],
    concurrency: int = 4,
    timeout: float = 180.0,
) -> list[ScoreResult]:
    """Synchronous entry point for :func:`_score_all` (creates its own event loop)."""
    return asyncio.run(_score_all(items, scorer, concurrency, timeout))


def timed() -> Callable[[], float]:
    """Return a callable that reports seconds elapsed since ``timed()`` was called."""
    start = time.perf_counter()
    return lambda: round(time.perf_counter() - start, 3)


# =============================================================================
# RAGAS judge LLM / embeddings factories (OpenAI <-> Anthropic swap lives here)
# =============================================================================
_DEFAULT_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def get_api_key(section: Mapping[str, Any], provider: str) -> str:
    """Read the API key from the environment variable named in the config (never from the file)."""
    env_name = section.get("api_key_env") or _DEFAULT_KEY_ENV.get(provider)
    if not env_name:
        raise ConfigError(f"No API key variable known for provider '{provider}'")
    value = os.environ.get(env_name)
    if not value:
        raise ConfigError(
            f"Environment variable {env_name} is not set. Export it or add it to .env "
            f"(provider='{provider}'). Use --no-llm / --skip-llm to run only non-LLM scores."
        )
    return value


def build_ragas_llm(cfg: Mapping[str, Any], section: str = "judge_llm") -> Any:
    """Create a RAGAS *instructor-based* LLM for the ``collections`` metrics.

    RAGAS >= 0.4 metrics call ``llm.agenerate(prompt, PydanticModel)``; the
    ``llm_factory`` wraps a provider SDK client (async) with the ``instructor``
    library to obtain that structured output. Swapping OpenAI <-> Anthropic is
    only a matter of ``provider`` + the client class.
    """
    from ragas.llms import llm_factory  # imported lazily: only LLM metrics need ragas

    sec = cfg.get(section) or {}
    provider = str(sec.get("provider", "openai")).lower()
    model = sec.get("model")
    if not model:
        raise ConfigError(f"{section}.model is not configured")
    key = get_api_key(sec, provider)
    timeout = float(sec.get("timeout_seconds", 120))
    # Extra model arguments forwarded by RAGAS to every request.
    model_args = {"temperature": float(sec.get("temperature", 0.0)), "max_tokens": int(sec.get("max_tokens", 2048))}

    if provider == "openai":
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=key, base_url=sec.get("base_url") or None, timeout=timeout)
    elif provider == "anthropic":
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=key, timeout=timeout)
    else:
        raise ConfigError(f"Unsupported {section}.provider '{provider}' (use 'openai' or 'anthropic')")
    logger.info("RAGAS judge LLM: provider=%s model=%s", provider, model)
    return llm_factory(model, provider=provider, client=client, **model_args)


def build_ragas_embeddings(cfg: Mapping[str, Any]) -> Any:
    """Create RAGAS embeddings (needed by Answer Relevance) for OpenAI or a local HF model."""
    from ragas.embeddings.base import embedding_factory

    sec = cfg.get("embeddings") or {}
    provider = str(sec.get("provider", "openai")).lower()
    model = sec.get("model")
    if provider == "openai":
        from openai import AsyncOpenAI

        # Async client so the metric can await ``aembed_text``.
        client = AsyncOpenAI(api_key=get_api_key(sec, "openai"), timeout=float(sec.get("timeout_seconds", 120)))
        return embedding_factory("openai", model=model or "text-embedding-3-small", client=client)
    if provider == "huggingface":
        # Runs locally through sentence-transformers; no API key needed.
        return embedding_factory("huggingface", model=model or "sentence-transformers/all-MiniLM-L6-v2")
    if provider == "anthropic":
        raise ConfigError("Anthropic has no embeddings API; set embeddings.provider to 'openai' or 'huggingface'.")
    raise ConfigError(f"Unsupported embeddings.provider '{provider}'")


def ragas_value(result: Any) -> float | None:
    """Extract the numeric score from a RAGAS ``MetricResult`` (or a bare number)."""
    value = getattr(result, "value", result)
    return None if value is None else float(value)


def eval_settings(cfg: Mapping[str, Any]) -> tuple[int, float, list[int]]:
    """Return ``(concurrency, per-sample timeout, k_values)`` from the evaluation block."""
    ev = cfg.get("evaluation") or {}
    ks = sorted({int(k) for k in ev.get("k_values", [1, 3, 5]) if int(k) > 0})
    return int(ev.get("concurrency", 4)), float(ev.get("per_sample_timeout_seconds", 180)), ks or [1, 3, 5]


def base_metadata(cfg: Mapping[str, Any], input_path: Path, n: int, seconds: float, **extra: Any) -> dict[str, Any]:
    """Metadata block recorded in every report (what was evaluated, with what, how long)."""
    return {
        "dataset": str(input_path),
        "n_samples": n,
        "config": cfg.get("_config_path"),
        "duration_seconds": seconds,
        "python": sys.version.split()[0],
        **extra,
    }
