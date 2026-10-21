# ADR 0001 — Core architecture decisions

- **Status:** Accepted. The "text first" consequence below is superseded by [ADR 0002](0002-hold-until-video-ready.md).
- **Date:** 2026-10-17

## Context

A kiosk for the fictional Dam International Airport must answer traveler
questions grounded in approved documents, speak the answer, and drive a
talking avatar — on a single RTX 4050 laptop (6 GB VRAM) within a short
timeline.

## Decisions

1. **Pipeline:** Whisper STT → LangGraph orchestration → OpenAI `gpt-4o` →
   Kokoro TTS → Ditto → kiosk.
2. **LangGraph is the orchestrator only.** It owns routing, retry limits,
   timeouts, and error handling. It does not do reasoning itself.
3. **OpenAI models:** `gpt-4o` for answer generation (max 8191 tokens);
   `text-embedding-3-small` (1536 dims) for retrieval.
4. **Vector store:** Chroma, persistent, populated by a separate offline
   ingestion job that indexes every document in the knowledge folder.
5. **Intent routing is deterministic** (rules + explicit policy), with no LLM
   or classifier model. Greetings use predefined responses; sensitive /
   out-of-scope categories are denied by policy. A classifier never authorizes
   a sensitive action on its own.
6. **Retrieval evidence assessment** uses calibrated similarity thresholds
   tuned on a committed test set. Strong evidence → generate; medium or weak →
   safe fallback (no query-rewrite retry, no clarifying question in v1).
7. **Jev (TypeSafe System One) is used exactly once** — at answer validation —
   to check whether every claim is supported by the retrieved passages. If Jev
   is unavailable, a deterministic citation-overlap check is the fallback.
8. **Hardware split:** Ditto owns the GPU; Whisper and Kokoro run on CPU;
   OpenAI and Jev are server-side API calls.
9. **Security:** all keys load server-side only; consent-based photos stay on
   local disk (gitignored) and are never sent to any cloud service.

## Consequences

- Grounding and refusal behavior are predictable and testable before any voice
  or avatar work begins.
- The single Jev dependency is isolated and has a deterministic fallback, so
  its early-access status cannot block the pipeline.
- Latency is dominated by Ditto, so approved answer text is shown immediately
  while audio and video render.
