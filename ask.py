"""Ask a question from the command line and see what each retrieval stage returns.

Usage:
    python ask.py "your question"                # answer once and exit
    python ask.py                                # interactive loop (empty line / exit / quit / Ctrl+C to leave)
    python ask.py -k 3 "question"                # change how many final passages are shown
    python ask.py --stage-k 10 "question"        # chunks shown per intermediate stage (default 5)
    python ask.py --full "question"              # print full chunk text instead of a short preview
    python ask.py --no-llm "question"            # retrieval only, skip the LLM answer
    python ask.py --model gpt-4o-mini "question" # override the chat model (default: OPENAI_CHAT_MODEL)

Reuses the app's HybridRetriever and prints the chunks of every stage, in order:
    1. ChromaDB dense results   2. BM25 sparse results
    3. RRF fused results        4. Cohere rerank results (final)
After the stages, the final reranked passages are sent to the configured OpenAI chat model,
which answers from those passages only and cites them as [1], [2], ... (skip with --no-llm).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXIT_WORDS = {"", "exit", "quit"}
DEFAULT_STAGE_K = 5
PREVIEW_CHARS = 200
LLM_TIMEOUT_SECONDS = 60.0
LLM_MAX_ANSWER_TOKENS = 1024
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the numbered context passages "
    "provided by the user. Do not use outside knowledge. Cite the passages that support each "
    "statement with their numbers in square brackets, e.g. [1] or [2][3]. If the context does not "
    "contain the answer, say that you don't know based on the provided context. Be concise."
)
_SECRET_RE = re.compile(r"\b(?:sk|key|co)-[A-Za-z0-9_\-]{8,}")
_secrets: list[str] = []


def _fail(message: str) -> "None":
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def _scrub(text: str) -> str:
    """Mask configured API keys (and key-looking tokens) before anything is printed."""
    for secret in _secrets:
        text = text.replace(secret, "***")
    return _SECRET_RE.sub("***", text)


def _load_retriever():
    # Settings use project-relative paths (.env, .chroma, .bm25), so run from the project root.
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    try:
        from config.settings import get_settings

        settings = get_settings()
    except Exception as exc:
        fields = getattr(exc, "errors", None)
        if callable(fields):
            missing = sorted({str(e["loc"][0]).upper() for e in fields() if e.get("loc")})
            _fail(f"invalid or missing configuration in .env: {', '.join(missing)}")
        _fail(f"could not load configuration: {type(exc).__name__}")

    for name in ("openai_api_key", "cohere_api_key"):
        value = getattr(settings, name, None)
        value = value.get_secret_value() if hasattr(value, "get_secret_value") else value
        if isinstance(value, str) and len(value) >= 8:
            _secrets.append(value)

    if not Path(settings.chroma_dir).is_dir():
        _fail(f"Chroma directory '{settings.chroma_dir}' not found. Run ingestion first.")
    if not Path(settings.bm25_index_path).is_file():
        _fail(f"BM25 index '{settings.bm25_index_path}' not found. Run ingestion first.")

    try:
        from app.retrieval.retrievers import HybridRetriever

        return HybridRetriever.from_settings(settings), settings
    except Exception as exc:
        _fail(f"could not initialise retrieval: {_scrub(str(exc))}")


def _body(text: str, full: bool) -> str:
    if full:
        return text.strip()
    flat = " ".join(text.split())
    return flat if len(flat) <= PREVIEW_CHARS else flat[:PREVIEW_CHARS].rstrip() + "..."


def _print_stage(title, items, trace, limit, full, note=None) -> None:
    print(f"\n=== {title} ({min(len(items), limit)} of {len(items)} shown) ===")
    if note:
        print(f"  note: {note}")
    if not items:
        print("  (no results)")
        return
    for rank, (cid, score) in enumerate(items[:limit], 1):
        source = trace.sources.get(cid, "?")
        text = trace.texts.get(cid)
        head = f"[{rank}] score={score:.4f}  source={source}  id={cid}"
        if text is None:
            print(f"{head}\n    (chunk text unavailable)")
        elif full:
            print(f"{head}\n{text.strip()}\n")
        else:
            print(f"{head}\n    {_body(text, False)}")


def _llm_error_message(exc: Exception) -> str:
    """Turn an OpenAI SDK failure into a short, readable message."""
    import openai

    if isinstance(exc, openai.AuthenticationError):
        return "authentication failed; check OPENAI_API_KEY in .env"
    if isinstance(exc, openai.PermissionDeniedError):
        return "permission denied for this key or model"
    if isinstance(exc, openai.RateLimitError):
        return "rate limit or quota exceeded; wait and retry, or check your OpenAI billing"
    if isinstance(exc, openai.APITimeoutError):
        return f"request timed out after {LLM_TIMEOUT_SECONDS:.0f}s"
    if isinstance(exc, openai.APIConnectionError):
        return "could not reach the OpenAI API; check your network connection"
    if isinstance(exc, openai.NotFoundError):
        return "model not found; check OPENAI_CHAT_MODEL or --model"
    if isinstance(exc, openai.APIStatusError):
        return f"API returned HTTP {exc.status_code}"
    return type(exc).__name__


def _generate_answer(settings, question: str, trace, top_k: int, model: str) -> None:
    """Ask the LLM to answer from the final reranked passages and print the result."""
    print("=== Answer ===")
    passages = []  # (source, text) in final rank order
    for cid, _score in trace.reranked[:top_k]:
        text = trace.texts.get(cid)
        if text and text.strip():
            passages.append((trace.sources.get(cid, "?"), text.strip()))
    if not passages:
        print("No passages available, so the LLM was not called.\n")
        return

    context = "\n\n".join(f"[{i}] (source: {src})\n{text}" for i, (src, text) in enumerate(passages, 1))
    user_message = f"Context passages:\n\n{context}\n\nQuestion: {question}"

    try:
        import openai

        key = settings.openai_api_key
        key = key.get_secret_value() if hasattr(key, "get_secret_value") else key
        if not key:
            print("Error: OPENAI_API_KEY is missing; set it in .env or use --no-llm.\n", file=sys.stderr)
            return
        client = openai.OpenAI(api_key=key, timeout=LLM_TIMEOUT_SECONDS, max_retries=1)
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=min(settings.openai_max_tokens, LLM_MAX_ANSWER_TOKENS),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        answer = (response.choices[0].message.content or "").strip()
    except ImportError:
        print("Error: the 'openai' package is not installed; use --no-llm.\n", file=sys.stderr)
        return
    except Exception as exc:
        print(f"Error: LLM call failed ({model}): {_scrub(_llm_error_message(exc))}\n", file=sys.stderr)
        return

    print(_scrub(answer) if answer else "(the model returned an empty answer)")
    print(f"\nModel: {model}")
    print("Passages used:")
    for i, (src, _text) in enumerate(passages, 1):
        print(f"  [{i}] {src}")
    print()


def _ask(retriever, question: str, top_k: int, stage_k: int, full: bool,
         llm_model: "str | None" = None, settings=None) -> None:
    try:
        trace = retriever.retrieve_trace(question, top_k)
    except Exception as exc:
        print(f"Error: retrieval failed: {_scrub(str(exc))}", file=sys.stderr)
        return

    err = trace.errors
    for stage in ("dense", "sparse", "fuse", "lookup"):
        if stage in err:
            print(f"Warning: {stage} stage failed: {_scrub(err[stage])}", file=sys.stderr)

    _print_stage("1. ChromaDB dense", trace.dense, trace, stage_k, full,
                 note=f"FAILED: {_scrub(err['dense'])}" if "dense" in err else None)
    _print_stage("2. BM25 sparse", trace.sparse, trace, stage_k, full,
                 note=f"FAILED: {_scrub(err['sparse'])}" if "sparse" in err else None)
    _print_stage("3. RRF fused", trace.fused, trace, stage_k, full,
                 note="built from the remaining stage only" if ("dense" in err) != ("sparse" in err) else None)
    if trace.rerank_fallback:
        print(f"Warning: Cohere rerank failed, showing RRF order instead: {_scrub(err['rerank'])}",
              file=sys.stderr)
        rerank_note = "Cohere rerank FAILED; fallback = RRF fused order (scores are RRF scores)"
        title = "4. Final (RRF fallback, no rerank)"
    else:
        rerank_note, title = None, "4. Cohere rerank (final)"
    _print_stage(title, trace.reranked, trace, top_k, full, note=rerank_note)
    if not trace.reranked:
        print("\nNo relevant passages found.")
    print()
    if llm_model:
        _generate_answer(settings, question, trace, top_k, llm_model)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask a question against the knowledge base.")
    parser.add_argument("question", nargs="*", help="question text; omit for interactive mode")
    parser.add_argument("-k", "--top-k", type=int, default=None, help="final passages to show")
    parser.add_argument("--stage-k", type=int, default=DEFAULT_STAGE_K,
                        help=f"chunks to show per intermediate stage (default {DEFAULT_STAGE_K})")
    parser.add_argument("--full", action="store_true",
                        help="print full chunk text instead of a ~200 char preview")
    parser.add_argument("--no-llm", action="store_true",
                        help="skip the LLM call and show retrieval output only")
    parser.add_argument("--model", default=None,
                        help="OpenAI chat model for the answer (default: OPENAI_CHAT_MODEL from settings)")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):  # chunk text may hold characters the console can't encode
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    retriever, settings = _load_retriever()
    top_k = args.top_k if args.top_k and args.top_k > 0 else min(5, settings.reranker_top_k)
    stage_k = args.stage_k if args.stage_k > 0 else DEFAULT_STAGE_K

    llm_model = None if args.no_llm else (args.model or settings.openai_chat_model)

    if args.question:
        _ask(retriever, " ".join(args.question), top_k, stage_k, args.full, llm_model, settings)
        return 0

    print("Ask a question (empty line, 'exit' or 'quit' to leave).")
    while True:
        try:
            question = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if question.lower() in EXIT_WORDS:
            break
        _ask(retriever, question, top_k, stage_k, args.full, llm_model, settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
