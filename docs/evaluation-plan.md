# Evaluation Plan

A repeatable evaluation harness (built in Task 9, extended in Tasks 13 and 18)
runs a committed question set under `evaluation/question_set/` and reports:

- **Retrieval quality** — are the right source passages retrieved?
- **Answer groundedness** — does every claim trace to a retrieved passage?
- **Refusal behavior** — are out-of-scope / unanswerable questions refused,
  not fabricated?
- **Transcription quality** — (Stage 2) spoken-input accuracy.
- **End-to-end latency** — (Stage 4) measured on the RTX 4050.

Each entry in the question set carries: the question, the expected answer
intent, the expected source document(s), and whether it should be refused.

This file is expanded with concrete metric definitions and thresholds in
Task 18.
