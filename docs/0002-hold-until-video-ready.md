# ADR 0002 — Hold text and voice until the talking-face video is ready

- **Status:** Accepted (supersedes the "text first" consequence in ADR 0001)
- **Date:** 2026-10-18

## Context

ADR 0001 said that, because Ditto dominates latency, approved answer text would be shown
immediately while audio and video rendered in the background. Stage 3 measured the real
cost on the RTX 4050 (6 GB): a warm Ditto worker renders at about 4x the audio length
(a typical 8 s answer takes about 40 s). Showing the text first and the face much later
means the traveler reads the answer, then sees a face speak words they already read.

## Decision

1. With the avatar on, the page **holds the answer text and the voice** until the video is
   ready, then shows the text and plays the video (its audio track is the Kokoro voice)
   together. While waiting it shows "Preparing the avatar..." with elapsed seconds.
2. **Fallback.** If the video is not ready within `AVATAR_TIMEOUT_SECONDS` (default 180 s),
   or the render fails, the text appears and the Kokoro voice plays over the still face,
   with a short notice. The page keeps polling in the background (up to 3x that limit) and,
   if the video arrives late, offers it through the Play answer button. Nothing the
   traveler needs is lost; only the moving face is.
3. **Fixed phrases are pre-rendered** with the default concierge only (greeting, thanks,
   goodbye, fallback, the refusals and the guardrail messages, 18 distinct clips), so the most
   common replies play with no wait. A traveler's own photo is used only for grounded answers,
   live; no clips are rendered for it (they used to block live turns for up to 90 s). The
   kiosk shows "Warming up" until the concierge's clips exist. *(Updated: this originally
   pre-rendered 14 clips per face.)*
4. Ditto runs as a **warm worker** in its own process so models load once.
5. `AVATAR_ENABLED=false` keeps the Stage 2 behavior (text first, voice after).

## Consequences

- The traveler waits for a typical live answer (tens of seconds) before seeing anything of
  it. That is the chosen trade-off for lips that match what is shown. The elapsed-seconds
  message, the Clear button (cancels the wait) and the timeout fallback keep it bearable.
- Pre-rendering turns the most frequent turns (greeting, refusals) into instant ones.
- Answers over roughly 15 s of speech may exceed the 180 s default (measured at 90 s when this was written) and fall back to voice only.
- Revisit in Stage 4 if the wait is too long on the target hardware: GPU ONNX for Ditto's
  aux models, a shorter answer-length cap, or sentence-by-sentence clips.

## Related decisions made in Stage 3

- **Traveler photos are session-only.** Consent tick required; decoded and re-encoded to a
  plain PNG (metadata dropped); stored under a random name; deleted on reset, Clear, page
  close, after a quiet period (server backstop 60 s, paused while a turn renders), at
  shutdown and at the next start. Never sent to a cloud service.
- **Handover.** The kiosk cannot detect a new traveler. After 30 s idle a "Still there?"
  prompt with a 15 s countdown asks the current traveler to confirm; if nobody does, the photo
  is deleted, the concierge returns and the page is cleared.
- **Not done:** presence detection with the camera and voice-change detection. Both would
  add biometric processing and unreliable signals on short, noisy clips. A voice-change hint
  ("Is this a new traveler?") is a possible Stage 4 addition, memory-only and consent-covered.