# LangSmith Setup & Dashboard Guide

Production observability for the Dam AI Talking Avatar Kiosk using
[LangSmith Cloud](https://smith.langchain.com) (free tier).

---

## 1. Prerequisites

| Item | Where |
|---|---|
| LangSmith account | [smith.langchain.com](https://smith.langchain.com) — sign up free |
| API key | LangSmith → Settings → API Keys → Create |
| Project | LangSmith → Projects → New Project → name it `dam-kiosk` |

---

## 2. Enable in `.env`

Add the following to your `.env` (copy from `.env.example`):

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls__your-key-here
LANGCHAIN_PROJECT=dam-kiosk
```

Restart the server. The startup log will confirm:

```
INFO:     app.tracing: LangSmith tracing enabled → project=dam-kiosk endpoint=https://api.smith.langchain.com
```

Verify with `GET /health` — response should include `"langsmith": true`.

---

## 3. Trace structure

Every `POST /api/ask` request produces one parent trace named **`ask_turn`**:

```
ask_turn
├── guardrail          ← LangGraph node (auto-traced)
├── supervisor         ← LLM call: gpt-4o-mini (token count visible)
├── rag                ← embedding + Cohere rerank (auto-traced)
├── answer             ← LLM call: gpt-4o (token count + cost visible)
├── finalize / fallback
├── kokoro_tts         ← custom child span: text_chars, audio_bytes, duration_ms
└── ditto_render       ← custom child span: face_key, status, render_ms
```

Every `POST /api/transcribe` produces an **`transcribe_turn`** trace:

```
transcribe_turn
└── whisper_stt        ← custom child span: audio_bytes, transcript_chars, confidence
```

---

## 4. Recommended dashboard charts

Open **LangSmith → dam-kiosk project → Dashboard**.

### A. End-to-end latency percentiles (p50 / p95 / p99)

- Filter: `name = ask_turn`
- Y-axis: Latency
- Chart type: Percentile over time
- Useful for: spotting turn latency regressions (normal p95 ≈ 3–8 s without video)

### B. Error rate by status

- Filter: `name = ask_turn`
- Group by: `metadata.status`
- Chart type: Stacked bar / percentage over time
- Values to watch: `fallback` (insufficient evidence), `error` (stage failure)
- Alert threshold: `error` rate > 5 %

### C. LLM cost over time

- Filter: `run_type = llm`
- Group by: `name` (separates `gpt-4o` vs `gpt-4o-mini` calls)
- Chart type: Total token cost over time
- Useful for: budget tracking; the `answer` node (gpt-4o) dominates cost

### D. Ditto render latency histogram

- Filter: `name = ditto_render`
- Y-axis: `outputs.render_ms`
- Chart type: Histogram / percentile
- Normal range: 30,000–120,000 ms (30–120 s) warm; first render after restart is longer
- Alert threshold: p95 > 150,000 ms (worker may be degraded)

### E. TTS duration vs text length scatter

- Filter: `name = kokoro_tts`
- X-axis: `inputs.text_chars`
- Y-axis: `outputs.duration_ms`
- Chart type: Scatter plot
- Useful for: confirming linear TTS scaling; outliers indicate CPU contention

### F. Whisper confidence distribution

- Filter: `name = whisper_stt`
- X-axis: `outputs.confidence` (0–1)
- Chart type: Histogram
- Low-confidence cluster (< 0.4) indicates poor microphone input or background noise

---

## 5. Filtering tips

| Goal | Filter |
|---|---|
| Debug one session | `metadata.session_id = <id>` |
| Compare routing paths | `metadata.route = airport_question` vs `greeting` vs `out_of_scope` |
| Isolate voice turns | `inputs.input_type = voice` |
| Find guardrail blocks | `metadata.guardrail_reason` is not null |
| Find answer failures | `metadata.status = fallback` or `error` |

---

## 6. Alert setup

In LangSmith → Projects → dam-kiosk → Alerts:

**Alert 1 — High error rate**
- Metric: Error rate for `name = ask_turn`
- Condition: > 10 % over a 5-minute window
- Channel: Email or Slack webhook

**Alert 2 — High latency**
- Metric: p95 latency for `name = ask_turn`
- Condition: > 30,000 ms (30 s) over a 10-minute window
- Indicates: LLM timeout, Cohere rerank slow, or Ditto queue backup

---

## 7. Privacy & data policy

**Important:** By default, LangSmith captures the full text of all LLM inputs and outputs.
For this kiosk, that includes traveler questions and retrieved passages.

- **Free tier** retains traces for **14 days**.
- Review [LangChain's privacy and data processing agreement](https://www.langchain.com/privacy) before
  production deployment.
- To redact all LLM inputs and outputs globally, add to `.env`:
  ```env
  LANGCHAIN_HIDE_INPUTS=true
  LANGCHAIN_HIDE_OUTPUTS=true
  ```
  This removes question text from traces but keeps timing, routing, and cost data intact.

---

## 8. Disabling tracing

Set `LANGSMITH_TRACING=false` or remove `LANGSMITH_API_KEY` from `.env` and restart.
`GET /health` will show `"langsmith": false`. The kiosk behaves identically; no LangSmith
SDK calls are made when tracing is disabled.
