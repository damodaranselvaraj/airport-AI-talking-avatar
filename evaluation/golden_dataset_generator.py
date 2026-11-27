#!/usr/bin/env python3
"""Golden dataset generator.

Pipeline (each step is a separate, importable function)::

    .md file / folder
        -> load_markdown_documents()          LangChain Documents
        -> generate_synthetic_records()       RAGAS TestsetGenerator: question, reference
                                              contexts (ground-truth chunks), reference answer
        -> attach_system_outputs()            an AnswerProvider adds `contexts` (retrieved) + `answer`
        -> validate_records()                 schema, null and duplicate checks
        -> export_dataset()                   golden_dataset.json (+ optional .jsonl)

Output record::

    {
      "question":           "...",
      "contexts":           ["retrieved chunk 1", ...],   # what the system retrieved / used (rank order)
      "ground_truth":       "reference answer",
      "answer":             "answer produced by the system under test",
      "reference_contexts": ["ground-truth chunk", ...],  # used by Precision/Recall/MRR/NDCG
      "question_type":      "simple | reasoning | multi_context",
      "source":             "airport_sample.md"
    }

Examples::

    python golden_dataset_generator.py --source sample_data/airport_sample.md --num-questions 20
    python golden_dataset_generator.py --source ../kiosk/data --answer-provider kiosk_app \
        --distribution simple=0.6,reasoning=0.2,multi_context=0.2 --output golden_dataset.json
    python golden_dataset_generator.py --seed-file my_questions.json --source docs.md   # no synthesis
"""
from __future__ import annotations

import argparse
import contextlib
import difflib
import importlib.metadata
import json
import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence

from metrics import eval_common as ec

from ragas.testset.transforms import default_transforms
from ragas.testset.transforms.splitters.headline import HeadlineSplitter

logger = logging.getLogger("rag_eval.generator")

SCHEMA_VERSION = "1.0"
REQUIRED_FIELDS = ("question", "contexts", "ground_truth", "answer")
QUESTION_TYPES = ("simple", "reasoning", "multi_context")
DEFAULT_DISTRIBUTION = {"simple": 0.5, "reasoning": 0.25, "multi_context": 0.25}


class DatasetValidationError(ec.EvalError):
    """Raised in strict mode when the generated dataset contains invalid records."""


# =============================================================================
# 1. Input: configurable .md data source
# =============================================================================
def load_markdown_documents(source: str | os.PathLike[str], split_by_headers: bool = False) -> list[Any]:
    """Read one ``.md`` file, or every ``*.md`` below a directory, as LangChain ``Document`` objects.

    Args:
        source: path to a Markdown file or a folder containing Markdown files.
        split_by_headers: split each file at ``#``/``##``/``###`` headings so every
            section becomes its own document (headings are kept in the text).
    """
    from langchain_core.documents import Document

    path = Path(source).expanduser()
    if path.is_dir():
        files = sorted([*path.rglob("*.md"), *path.rglob("*.markdown")])
    elif path.is_file():
        if path.suffix.lower() not in (".md", ".markdown"):
            raise ec.EvalError(f"Data source must be a Markdown (.md) file or a folder of them, got: {path}")
        files = [path]
    else:
        raise ec.EvalError(f"Data source not found: {path}")
    if not files:
        raise ec.EvalError(f"No .md files found under {path}")

    splitter = None
    if split_by_headers:
        from langchain_text_splitters import MarkdownHeaderTextSplitter

        # strip_headers=False keeps headings inside the chunk text (useful context for questions).
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False
        )

    docs: list[Any] = []
    for f in files:
        text = f.read_text(encoding="utf-8-sig", errors="replace")  # utf-8-sig drops a BOM if present
        if not text.strip():
            logger.warning("Skipping empty file %s", f)
            continue
        if splitter is None:
            docs.append(Document(page_content=text, metadata={"source": f.name}))
        else:
            docs += [Document(page_content=d.page_content, metadata={"source": f.name, **d.metadata})
                     for d in splitter.split_text(text)]
    if not docs:
        raise ec.EvalError(f"All Markdown files under {path} are empty")
    logger.info("Loaded %d document(s) from %d file(s)", len(docs), len(files))
    return docs


# =============================================================================
# 2. Question-type distribution
# =============================================================================
def parse_distribution(value: str | Mapping[str, float] | None) -> dict[str, float]:
    """Parse ``simple=0.5,reasoning=0.3,multi_context=0.2`` (or a mapping) into normalised weights."""
    if value is None:
        return dict(DEFAULT_DISTRIBUTION)
    if isinstance(value, str):
        try:
            value = {k.strip(): float(v) for k, v in (p.split("=") for p in value.split(",") if p.strip())}
        except ValueError as exc:
            raise ec.EvalError("distribution must look like 'simple=0.5,reasoning=0.25,multi_context=0.25'") from exc
    unknown = [k for k in value if k not in QUESTION_TYPES]
    if unknown:
        raise ec.EvalError(f"Unknown question type(s) {unknown}; choose from {QUESTION_TYPES}")
    if any(float(w) < 0 for w in value.values()):
        raise ec.EvalError("distribution weights must be >= 0")
    total = sum(float(w) for w in value.values())
    if total <= 0:
        raise ec.EvalError("distribution weights must sum to more than 0")
    return {k: float(value.get(k, 0.0)) / total for k in QUESTION_TYPES if float(value.get(k, 0.0)) > 0}


def allocate_counts(total: int, distribution: Mapping[str, float]) -> dict[str, int]:
    """Largest-remainder split of ``total`` questions across types (for logging/expectations)."""
    raw = {k: total * w for k, w in distribution.items()}
    counts = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[: total - sum(counts.values())]:
        counts[k] += 1
    return counts


# =============================================================================
# 3. Model construction (OpenAI <-> Anthropic swap lives in config.yaml)
# =============================================================================
def build_chat_model(section: Mapping[str, Any]) -> Any:
    """LangChain chat model from a ``generator_llm``-style config block."""
    provider = str(section.get("provider", "openai")).lower()
    model = section.get("model")
    if not model:
        raise ec.ConfigError("generator_llm.model is not configured")
    key = ec.get_api_key(section, provider)            # read from the environment, never from the file
    timeout = float(section.get("timeout_seconds", 120))
    common = {"temperature": float(section.get("temperature", 0.2)), "max_tokens": int(section.get("max_tokens", 2048))}
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=key, base_url=section.get("base_url") or None,
                          timeout=timeout, max_retries=2, **common)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, api_key=key, timeout=timeout, max_retries=2, **common)
    raise ec.ConfigError(f"Unsupported provider '{provider}' (use 'openai' or 'anthropic')")


def build_lc_embeddings(section: Mapping[str, Any]) -> Any:
    """LangChain embeddings for the TestsetGenerator knowledge graph (OpenAI or local HuggingFace)."""
    provider = str(section.get("provider", "openai")).lower()
    model = section.get("model")
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model or "text-embedding-3-small", api_key=ec.get_api_key(section, "openai"))
    if provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ec.ConfigError("embeddings.provider=huggingface needs: pip install langchain-huggingface sentence-transformers") from exc
        return HuggingFaceEmbeddings(model_name=model or "sentence-transformers/all-MiniLM-L6-v2")
    raise ec.ConfigError(f"Unsupported embeddings.provider '{provider}' (use 'openai' or 'huggingface')")


def _message_text(content: Any) -> str:
    """Flatten a LangChain message ``content`` (str or list of blocks, as Anthropic returns) to text."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


# =============================================================================
# 4. Synthetic generation with RAGAS TestsetGenerator
# =============================================================================
def build_query_distribution(llm: Any, distribution: Mapping[str, float]) -> list[tuple[Any, float]]:
    """Translate simple/reasoning/multi_context weights into RAGAS query synthesizers.

    RAGAS >= 0.2 replaced the old "evolution" types with query synthesizers:

    * ``simple``        -> ``SingleHopSpecificQuerySynthesizer``  (one chunk, a specific fact)
    * ``reasoning``     -> ``MultiHopAbstractQuerySynthesizer``   (abstract question needing inference over related chunks)
    * ``multi_context`` -> ``MultiHopSpecificQuerySynthesizer``   (specific facts spread over several chunks)
    """
    from ragas.testset.synthesizers import (
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
        SingleHopSpecificQuerySynthesizer,
    )

    factory = {
        "simple": SingleHopSpecificQuerySynthesizer,
        "reasoning": MultiHopAbstractQuerySynthesizer,
        "multi_context": MultiHopSpecificQuerySynthesizer,
    }
    # RAGAS expects a list of (synthesizer, probability) pairs summing to 1.
    return [(factory[t](llm=llm), w) for t, w in distribution.items()]


def _type_from_synthesizer(name: str | None) -> str:
    """Map a RAGAS synthesizer name back to our question-type vocabulary."""
    n = (name or "").lower()
    if "abstract" in n:
        return "reasoning"
    if "multi_hop" in n or "multihop" in n:
        return "multi_context"
    return "simple"


def _infer_source(reference_contexts: Sequence[str], documents: Sequence[Any]) -> str | None:
    """Best-effort: which input file does the first ground-truth chunk come from?"""
    if not reference_contexts:
        return None
    # Compare the first ~80 characters, ignoring whitespace differences and trailing punctuation
    # (a short chunk may end where the source sentence continues).
    needle = " ".join(reference_contexts[0].split())[:80].rstrip(".,;:!? ")
    for d in documents:
        if needle and needle in " ".join(d.page_content.split()):
            return d.metadata.get("source")
    return None


def testset_to_records(testset: Any, documents: Sequence[Any] = ()) -> list[dict[str, Any]]:
    """Convert a RAGAS ``Testset`` into plain dict records (answer/contexts are filled in later)."""
    df = testset.to_pandas()        # columns: user_input, reference_contexts, reference, synthesizer_name
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        refs = [str(c) for c in (row.get("reference_contexts") or [])]
        records.append({
            "question": row.get("user_input"),
            "ground_truth": row.get("reference"),
            "reference_contexts": refs,
            "question_type": _type_from_synthesizer(row.get("synthesizer_name")),
            "source": _infer_source(refs, documents),
        })
    return records

class SafeHeadlineSplitter(HeadlineSplitter):
    """Prevent splitting failures when a node has no extracted headlines."""

    async def split(self, node):
        # Some document nodes do not receive headlines because the default
        # HeadlinesExtractor processes only documents exceeding 500 tokens.
        # Preserve those nodes instead of raising a missing-property error.
        if node.get_property("headlines") is None:
            return [node], []

        # Use RAGAS's normal splitting logic when headlines are available.
        return await super().split(node)


def generate_synthetic_records(documents: Sequence[Any], cfg: Mapping[str, Any], num_questions: int,
                               distribution: Mapping[str, float]) -> list[dict[str, Any]]:
    """Use RAGAS ``TestsetGenerator`` to create question / ground-truth-context / reference-answer triples."""
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    from ragas.llms.base import LangchainLLMWrapper
    from ragas.testset import TestsetGenerator

    # TestsetGenerator still takes the LangChain-wrapped LLM/embeddings (the wrappers emit a
    # DeprecationWarning in RAGAS 0.4 - harmless, the generator API has not migrated yet).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        llm = LangchainLLMWrapper(build_chat_model(cfg.get("generator_llm") or cfg.get("judge_llm") or {}))
        embeddings = LangchainEmbeddingsWrapper(build_lc_embeddings(cfg.get("embeddings") or {}))
    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)

    def attempt(dist: Mapping[str, float]) -> list[dict[str, Any]]:
        logger.info("Generating %d question(s); target mix %s", num_questions, allocate_counts(num_questions, dist))
        
        generator = TestsetGenerator(
            llm=llm,
            embedding_model=embeddings,
        )

        # Build RAGAS's default transformations explicitly.
        transforms = default_transforms(
            documents=list(documents),
            llm=generator.llm,
            embedding_model=generator.embedding_model,
        )

        # Replace the default splitter with one that handles missing headlines.
        transforms = [
            SafeHeadlineSplitter(min_tokens=500)
            if isinstance(transform, HeadlineSplitter)
            else transform
            for transform in transforms
        ]

        # Generate the synthetic golden dataset using the safe transformations.
        testset = generator.generate_with_langchain_docs(
            list(documents),
            testset_size=num_questions,
            transforms=transforms,
            query_distribution=build_query_distribution(llm, dist),
            raise_exceptions=False,
        )
        return testset_to_records(testset, documents)

    records = attempt(distribution)
    if not records and set(distribution) != {"simple"}:
        # Multi-hop synthesizers need related chunks; tiny inputs may not have any.
        logger.warning("No questions produced for %s; retrying with simple questions only", dict(distribution))
        records = attempt({"simple": 1.0})
    if not records:
        raise ec.EvalError("RAGAS produced no questions. Use a longer/more document-rich source or lower --num-questions.")
    return records


def load_seed_records(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load hand-written golden questions (skips synthesis).

    Expected JSON list of objects with ``question`` and ``ground_truth`` (alias ``reference``)
    and, optionally, ``reference_contexts``, ``question_type``, ``source``.
    """
    try:
        items = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ec.EvalError(f"Cannot read seed file {path}: {exc}") from exc
    if not isinstance(items, list):
        raise ec.EvalError("Seed file must contain a JSON list of objects")
    return [{"question": it.get("question"), "ground_truth": it.get("ground_truth", it.get("reference")),
             "reference_contexts": it.get("reference_contexts") or [],
             "question_type": it.get("question_type", "simple"), "source": it.get("source")} for it in items]


# =============================================================================
# 5. Answer providers: fill `contexts` (retrieved) and `answer` (system output)
# =============================================================================
@dataclass
class ProviderOutput:
    """What an answer provider returns for one question."""

    contexts: list[str]                              # retrieved chunks in rank order
    answer: str                                      # system answer
    extra: dict[str, Any] = field(default_factory=dict)   # additional columns stored on the record


class AnswerProvider(Protocol):
    """Anything that can answer a question; implement this to plug in another RAG system."""

    name: str

    def answer(self, question: str, reference_contexts: Sequence[str]) -> ProviderOutput: ...


_BASELINE_SYSTEM = (
    "You answer questions for an airport information kiosk using ONLY the numbered context passages. "
    "Reply in one to three short plain sentences. If the passages do not contain the answer, reply exactly: "
    "I can't confirm that from the airport information I have."
)


class LLMBaselineProvider:
    """Answers from the ground-truth chunks with the generator LLM.

    ``contexts`` = RAGAS reference chunks, so retrieval metrics are trivially perfect;
    use it to sanity-check generation metrics, then switch to ``kiosk_app`` to evaluate
    the real retriever.
    """

    name = "llm_baseline"

    def __init__(self, chat_model: Any) -> None:
        self._model = chat_model

    def answer(self, question: str, reference_contexts: Sequence[str]) -> ProviderOutput:
        numbered = "\n\n".join(f"[{i}] {c.strip()}" for i, c in enumerate(reference_contexts, 1))
        reply = self._model.invoke([("system", _BASELINE_SYSTEM), ("human", f"Passages:\n{numbered}\n\nQuestion: {question}")])
        return ProviderOutput(list(reference_contexts), _message_text(reply.content).strip(), {"answer_status": "answered"})


class KioskAppProvider:
    """Runs the real kiosk RAG path (retrieval -> evidence gate -> generator -> validator).

    It imports the kiosk's own packages (``config.settings``, ``app.retrieval``, ``app.workflow``),
    so ``kiosk_app.root`` must point at the kiosk project and its env vars (OPENAI_API_KEY,
    COHERE_API_KEY, TYPESAFE_API_KEY) and indexes must exist. The guardrail/intent router are
    intentionally bypassed: they decide *whether* to answer, while this evaluates *how well* the
    RAG pipeline answers questions it does accept.
    """

    name = "kiosk_app"

    def __init__(self, root: Path, top_k: int | None, fallback: str) -> None:
        if not root.is_dir():
            raise ec.ConfigError(f"kiosk_app.root does not exist: {root}")
        self._root, self._fallback = root.resolve(), fallback
        sys.path.insert(0, str(self._root))     # make the kiosk's `app` and `config` packages importable
        try:
            with self._in_root():
                from app.retrieval.retrievers import HybridRetriever
                from app.retrieval.search import AirportSearch
                from app.workflow.answer import OpenAIAnswerGenerator
                from app.workflow.evidence import EvidenceAssessor, EvidenceThresholds
                from app.workflow.validation import CitationOverlapValidator
                from config.settings import get_settings

                s = get_settings()                          # raises if required env vars are missing
                self._search = AirportSearch(HybridRetriever.from_settings(s), top_k=top_k or s.answer_passages)
                self._generator = OpenAIAnswerGenerator(api_key=s.openai_api_key, model=s.openai_chat_model,
                                                        max_tokens=s.answer_max_tokens, timeout=s.node_timeout_seconds)
                self._assessor = EvidenceAssessor(EvidenceThresholds.from_file(s.evidence_thresholds_path))
                self._validator = CitationOverlapValidator()
        except ImportError as exc:
            raise ec.ConfigError(f"Cannot import the kiosk app from {self._root}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - pydantic/Chroma/etc. errors become one actionable message
            raise ec.ConfigError(f"Kiosk app failed to initialise: {ec.scrub(str(exc))[:300]}") from exc

    @contextlib.contextmanager
    def _in_root(self) -> Iterator[None]:
        """The kiosk resolves .env / data paths relative to its own folder, so run calls from there."""
        old = Path.cwd()
        os.chdir(self._root)
        try:
            yield
        finally:
            os.chdir(old)

    def answer(self, question: str, reference_contexts: Sequence[str]) -> ProviderOutput:
        with self._in_root():
            found = self._search.search_airport(question)
            passages = list(found.passages)
            contexts = [p.text for p in passages]
            extra: dict[str, Any] = {"chunk_ids": [p.chunk_id for p in passages], "retrieval_scores": [p.score for p in passages],
                                     "rerank_applied": found.rerank_applied}
            tier = self._assessor.assess(passages, found.rerank_applied, found.retrieval_degraded)
            extra["evidence_tier"] = tier.value
            if tier.value != "strong":                         # same gate as the kiosk workflow
                return ProviderOutput(contexts, self._fallback, {**extra, "answer_status": "fallback_weak_evidence"})
            gen = self._generator.generate(question, passages)
            if not gen.can_answer or not gen.text:
                return ProviderOutput(contexts, self._fallback, {**extra, "answer_status": "fallback_cannot_answer"})
            cited = [passages[i] for i in gen.used if 0 <= i < len(passages)]
            if not self._validator.validate(gen.text, cited).supported:   # same validator as the kiosk
                return ProviderOutput(contexts, self._fallback, {**extra, "answer_status": "fallback_unsupported"})
            return ProviderOutput(contexts, gen.text, {**extra, "answer_status": "answered"})


def build_answer_provider(name: str, cfg: Mapping[str, Any]) -> AnswerProvider:
    """Create the provider selected by ``generation.answer_provider`` / ``--answer-provider``."""
    if name == "llm_baseline":
        return LLMBaselineProvider(build_chat_model(cfg.get("generator_llm") or cfg.get("judge_llm") or {}))
    if name == "kiosk_app":
        k = cfg.get("kiosk_app") or {}
        return KioskAppProvider(ec.resolve_path(cfg, k.get("root", "../kiosk")), k.get("top_k"),
                                k.get("fallback_answer", "I can't confirm that from the airport information I have."))
    raise ec.ConfigError(f"Unknown answer provider '{name}' (choose llm_baseline or kiosk_app)")


def attach_system_outputs(records: list[dict[str, Any]], provider: AnswerProvider) -> list[dict[str, Any]]:
    """Fill ``contexts`` and ``answer`` for every record. A failing question is kept with an empty answer
    so that validation reports (and drops) it instead of silently shifting the dataset."""
    for i, rec in enumerate(records, 1):
        try:
            out = provider.answer(str(rec.get("question") or ""), rec.get("reference_contexts") or [])
            rec["contexts"], rec["answer"] = out.contexts, out.answer
            rec.update(out.extra)
        except Exception as exc:  # noqa: BLE001 - API/network errors are per-question
            logger.warning("Provider failed on question %d: %s", i, ec.scrub(f"{type(exc).__name__}: {exc}")[:200])
            rec.setdefault("contexts", [])
            rec["answer"] = ""
            rec["answer_status"] = "provider_error"
        if i % 5 == 0 or i == len(records):
            logger.info("  answered %d/%d", i, len(records))
    return records


# =============================================================================
# 6. Validation: schema, nulls, duplicates
# =============================================================================
@dataclass
class ValidationReport:
    """Outcome of :func:`validate_records`."""

    total_in: int
    total_out: int = 0
    issues: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        by_code: dict[str, int] = {}
        for i in self.issues:
            by_code[i["code"]] = by_code.get(i["code"], 0) + 1
        return {"total_in": self.total_in, "total_out": self.total_out,
                "dropped": self.total_in - self.total_out, "issues_by_code": by_code, "issues": self.issues}


def _normalize_question(q: str) -> str:
    """Case/punctuation/whitespace-insensitive form used for duplicate detection."""
    return " ".join(re.sub(r"[^\w\s]", " ", q.casefold()).split())


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def validate_records(
    records: Sequence[Any],
    *,
    strict: bool = False,
    near_duplicate_threshold: float = 0.92,
    drop_near_duplicates: bool = True,
    min_question_chars: int = 8,
) -> tuple[list[dict[str, Any]], ValidationReport]:
    """Validate and clean records.

    Checks: required fields present, correct types (schema); null / blank question,
    ground truth, answer (null); exact and near-duplicate questions (duplicates).
    Records with an *error* are dropped (or raise in ``strict`` mode); *warnings*
    (e.g. no retrieved contexts) keep the record.
    """
    report = ValidationReport(total_in=len(records))
    kept: list[dict[str, Any]] = []
    kept_norm: list[str] = []

    def issue(idx: int, severity: str, code: str, message: str) -> None:
        report.issues.append({"index": idx, "severity": severity, "code": code, "message": message})

    for idx, rec in enumerate(records):
        before = len(report.issues)
        # ---- schema ------------------------------------------------------------------
        if not isinstance(rec, dict):
            issue(idx, "error", "SCHEMA_NOT_OBJECT", "record is not a JSON object")
            continue
        missing = [f for f in REQUIRED_FIELDS if f not in rec]
        if missing:
            issue(idx, "error", "SCHEMA_MISSING_FIELD", f"missing field(s): {', '.join(missing)}")
            continue
        for f in ("question", "ground_truth", "answer"):
            if rec[f] is not None and not isinstance(rec[f], str):
                issue(idx, "error", "SCHEMA_TYPE", f"'{f}' must be a string, got {type(rec[f]).__name__}")
        ctx = rec["contexts"]
        if ctx is not None and not (isinstance(ctx, list) and all(isinstance(c, str) for c in ctx)):
            issue(idx, "error", "SCHEMA_TYPE", "'contexts' must be a list of strings")
        refs = rec.get("reference_contexts")
        if refs is not None and not (isinstance(refs, list) and all(isinstance(c, str) for c in refs)):
            issue(idx, "error", "SCHEMA_TYPE", "'reference_contexts' must be a list of strings")
        # ---- nulls / blanks ----------------------------------------------------------
        for f in ("question", "ground_truth", "answer"):
            if _blank(rec[f]):
                issue(idx, "error", "NULL_FIELD", f"'{f}' is null or empty")
        if ctx is None:
            issue(idx, "error", "NULL_FIELD", "'contexts' is null")
        elif isinstance(ctx, list):
            if not ctx:
                issue(idx, "warning", "EMPTY_CONTEXTS", "no retrieved contexts (retrieval metrics will score 0)")
            elif any(_blank(c) for c in ctx):
                issue(idx, "error", "BLANK_CONTEXT", "contexts contains a blank string")
        if isinstance(rec["question"], str) and 0 < len(rec["question"].strip()) < min_question_chars:
            issue(idx, "error", "SHORT_QUESTION", f"question shorter than {min_question_chars} characters")
        if any(i["severity"] == "error" for i in report.issues[before:]):
            continue
        # ---- duplicates --------------------------------------------------------------
        norm = _normalize_question(rec["question"])
        if norm in kept_norm:
            issue(idx, "error", "DUPLICATE_QUESTION", "exact duplicate of an earlier question")
            continue
        similar = next((o for o in kept_norm if difflib.SequenceMatcher(None, norm, o).ratio() >= near_duplicate_threshold), None)
        if similar is not None:
            issue(idx, "error" if drop_near_duplicates else "warning", "NEAR_DUPLICATE_QUESTION",
                  f"question is >= {near_duplicate_threshold:.2f} similar to an earlier one")
            if drop_near_duplicates:
                continue
        # ---- keep a cleaned copy -----------------------------------------------------
        clean = dict(rec)
        for f in ("question", "ground_truth", "answer"):
            clean[f] = rec[f].strip()
        clean["contexts"] = [c.strip() for c in rec["contexts"]]
        kept.append(clean)
        kept_norm.append(norm)

    report.total_out = len(kept)
    errors = [i for i in report.issues if i["severity"] == "error"]
    if strict and errors:
        first = "; ".join(f"#{i['index']} {i['code']}: {i['message']}" for i in errors[:5])
        raise DatasetValidationError(f"{len(errors)} invalid record(s) in strict mode - {first}")
    return kept, report


# =============================================================================
# 7. Export
# =============================================================================
def export_dataset(records: Sequence[dict[str, Any]], output: str | os.PathLike[str],
                   metadata: Mapping[str, Any], jsonl: bool = False) -> list[Path]:
    """Write the golden dataset as JSON (``{"schema_version", "metadata", "samples"}``) and optionally JSON Lines.

    The JSON Lines file (one record per line) is the Hugging Face ``datasets`` JSON format,
    loadable with ``datasets.load_dataset("json", data_files=...)``.
    """
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION, "metadata": dict(metadata), "samples": list(records)}
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(out)                                  # atomic: never leaves a half-written dataset
    written = [out]
    if jsonl:
        lines = out.with_suffix(".jsonl")
        with lines.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        written.append(lines)
    logger.info("Exported %d sample(s) to %s", len(records), ", ".join(str(p) for p in written))
    return written


# =============================================================================
# 8. Orchestration
# =============================================================================
def build_golden_dataset(
    cfg: Mapping[str, Any],
    *,
    source: str | None = None,
    num_questions: int | None = None,
    distribution: str | Mapping[str, float] | None = None,
    answer_provider: str | None = None,
    seed_file: str | None = None,
    split_by_headers: bool | None = None,
    strict: bool | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the full pipeline and return ``(clean_records, metadata)`` without writing files."""
    gen_cfg, val_cfg = cfg.get("generation") or {}, cfg.get("validation") or {}
    n = int(num_questions or gen_cfg.get("num_questions", 20))
    dist = parse_distribution(distribution if distribution is not None else gen_cfg.get("distribution"))
    provider_name = answer_provider or gen_cfg.get("answer_provider", "llm_baseline")
    src = Path(source) if source else ec.resolve_path(cfg, ec.cfg_get(cfg, "paths.data_source", "sample_data/airport_sample.md"))
    if n < 1:
        raise ec.EvalError("--num-questions must be at least 1")

    docs = load_markdown_documents(src, bool(gen_cfg.get("split_by_headers", False) if split_by_headers is None else split_by_headers))
    records = load_seed_records(seed_file) if seed_file else generate_synthetic_records(docs, cfg, n, dist)
    requested = len(records) if seed_file else n
    if seed_file:                                    # seed records may lack a source; infer it from the documents
        for r in records:
            r["source"] = r.get("source") or _infer_source(r.get("reference_contexts") or [], docs)

    provider = build_answer_provider(provider_name, cfg)
    records = attach_system_outputs(records, provider)
    clean, report = validate_records(
        records,
        strict=bool(val_cfg.get("strict", False) if strict is None else strict),
        near_duplicate_threshold=float(val_cfg.get("near_duplicate_threshold", 0.92)),
        drop_near_duplicates=bool(val_cfg.get("drop_near_duplicates", True)),
        min_question_chars=int(val_cfg.get("min_question_chars", 8)),
    )
    realised: dict[str, int] = {}
    for r in clean:
        realised[r.get("question_type", "simple")] = realised.get(r.get("question_type", "simple"), 0) + 1
    try:
        ragas_version = importlib.metadata.version("ragas")
    except importlib.metadata.PackageNotFoundError:
        ragas_version = None
    gen_llm = cfg.get("generator_llm") or {}
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(src),
        "mode": "seed_file" if seed_file else "ragas_synthetic",
        "ragas_version": ragas_version,
        "generator_llm": f"{gen_llm.get('provider')}/{gen_llm.get('model')}",
        "answer_provider": provider_name,
        "requested_questions": requested,
        "exported_questions": len(clean),
        "requested_distribution": dist,
        "realised_question_types": realised,
        "validation": report.to_dict(),
    }
    if len(clean) < requested:
        logger.warning("Exported %d of %d requested question(s); see metadata.validation.issues", len(clean), requested)
    return clean, metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line interface."""
    p = argparse.ArgumentParser(description="Generate a golden dataset (question, contexts, ground_truth, answer) from Markdown.")
    p.add_argument("--source", "-s", help=".md file or folder of .md files (default: paths.data_source in config)")
    p.add_argument("--output", "-o", help="output JSON path (default: paths.golden_dataset in config)")
    p.add_argument("--num-questions", "-n", type=int, help="number of questions to generate")
    p.add_argument("--distribution", help="e.g. simple=0.5,reasoning=0.25,multi_context=0.25")
    p.add_argument("--answer-provider", choices=["llm_baseline", "kiosk_app"], help="how `answer`/`contexts` are produced")
    p.add_argument("--seed-file", help="JSON list of hand-written questions; skips RAGAS synthesis")
    p.add_argument("--split-headers", action="store_true", default=None, help="split Markdown files at headings first")
    p.add_argument("--strict", action="store_true", default=None, help="fail on any invalid record instead of dropping it")
    p.add_argument("--jsonl", action="store_true", help="also write a JSON Lines copy (Hugging Face datasets format)")
    p.add_argument("--config", "-c", help="config.yaml path")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Exit codes: 0 ok, 2 user-facing error, 1 unexpected failure."""
    args = parse_args(argv)
    ec.setup_logging(args.log_level)
    try:
        cfg = ec.load_config(args.config)
        records, metadata = build_golden_dataset(
            cfg, source=args.source, num_questions=args.num_questions, distribution=args.distribution,
            answer_provider=args.answer_provider, seed_file=args.seed_file,
            split_by_headers=args.split_headers, strict=args.strict,
        )
        if not records:
            raise ec.EvalError("No valid records left after validation; nothing to export")
        out = Path(args.output) if args.output else ec.resolve_path(cfg, ec.cfg_get(cfg, "paths.golden_dataset", "golden_dataset.json"))
        export_dataset(records, out, metadata, jsonl=args.jsonl)
    except (ec.EvalError, ImportError) as exc:
        logger.error("%s", ec.scrub(str(exc)))
        return 2
    except Exception:  # noqa: BLE001
        logger.exception("Golden dataset generation failed")
        return 1
    print(f"\nGolden dataset: {out}  ({len(records)} samples, types: {metadata['realised_question_types']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
