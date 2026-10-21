"""Centralized, server-side configuration for Dam AI Talking Avatar Kiosk.

All secrets load here and nowhere else. The browser never receives them.
Required secrets have no default, so a missing value raises a validation
error at startup instead of failing mid-conversation ("fail loudly").
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Required secrets (no default -> missing value fails loudly) ---
    openai_api_key: str = Field(..., description="OpenAI API key (server-side only)")
    # Jev (TypeSafe) is used at the answer-validation step. Required at startup
    # so misconfiguration surfaces immediately, not mid-turn.
    typesafe_api_key: str = Field(..., description="TypeSafe/Jev API key (server-side only)")
    # Cohere reranking has no local fallback, so the key is required at startup.
    cohere_api_key: str = Field(..., description="Cohere API key (reranking, no fallback)")

    # --- OpenAI model configuration ---
    openai_chat_model: str = Field("gpt-4o", description="Chat completion model")
    openai_classifier_model: str = Field(
        "gpt-4o-mini",
        description="Small/cheap model for the supervisor's intent classification (multi-agent graph)",
    )
    openai_embedding_model: str = Field(
        "text-embedding-3-small", description="Embedding model"
    )
    openai_embedding_dims: int = Field(1536, description="Embedding output dimensions")
    openai_max_tokens: int = Field(8191, description="Max tokens per completion")
    answer_max_tokens: int = Field(
        400,
        description="Hard cap on tokens for spoken answer generation (answers are 1-3 short sentences)",
    )

    # --- Paths ---
    knowledge_dir: str = Field(
        "knowledge/dam-source-documents", description="Source documents to ingest"
    )
    chroma_dir: str = Field(
        "chroma_db", description="Local persistent ChromaDB directory (PersistentClient)"
    )
    faces_dir: str = Field("faces", description="Consent-based photos (local, gitignored)")

    # --- Kiosk behavior ---
    default_language: str = Field("en", description="Active language (English for v1)")
    session_timeout_seconds: int = Field(120, description="Visible session timer budget")

    # --- Voice: speech-to-text (Whisper, CPU) ---
    whisper_model: str = Field("small.en", description="faster-whisper model size/name")
    whisper_compute_type: str = Field("int8", description="CTranslate2 compute type")
    whisper_device: str = Field("cpu", description="Ditto owns the GPU; keep Whisper on CPU")
    max_audio_seconds: float = Field(20.0, description="Reject recordings longer than this")
    max_audio_bytes: int = Field(5_000_000, description="Reject uploads larger than this")

    # --- Voice: text-to-speech (Kokoro, CPU) ---
    kokoro_voice: str = Field("af_heart")
    kokoro_lang: str = Field("a", description="Kokoro lang_code (a = American English)")
    sessions_dir: str = Field("sessions", description="Per-turn generated audio (gitignored)")
    audio_ttl_seconds: int = Field(600, description="Generated turn audio is deleted after this")

    # --- Avatar: Ditto talking head (GPU, isolated conda env) ---
    avatar_enabled: bool = Field(False, description="Render a talking-face video for spoken turns")
    ditto_python: str = Field(
        str(Path.home() / "anaconda3" / "envs" / "ditto" / "python.exe"),
        description="Python of the isolated 'ditto' conda env that runs the warm worker",
    )
    ditto_worker_script: str = Field("scripts/ditto_worker.py")
    ditto_repo: str = Field("third_party/ditto-talkinghead")
    ditto_data_root: str = Field("checkpoints/ditto_pytorch")
    ditto_cfg_pkl: str = Field("checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl")
    concierge_dir: str = Field("config/concierge", description="Default concierge face image lives here")
    concierge_image: str = Field("", description="Explicit concierge image path (overrides concierge_dir)")
    avatar_startup_timeout_seconds: float = Field(180.0, description="Max wait for the worker to load models")
    avatar_timeout_seconds: float = Field(
        180.0, description="Give up on a turn's video after this long and fall back to text + audio"
    )
    avatar_max_queue: int = Field(64, description="Max queued avatar jobs (live + background)")
    video_ttl_seconds: int = Field(600, description="Generated turn videos are deleted after this")
    clips_dir: str = Field("cache/clips", description="Pre-rendered phrase clips (gitignored)")
    clips_prerender_on_start: bool = Field(
        True,
        description="Render the default concierge's missing phrase clips at startup (the kiosk shows 'Warming up' meanwhile)",
    )
    max_face_bytes: int = Field(5_000_000, description="Reject traveler photo uploads larger than this")
    face_idle_warn_seconds: int = Field(30, description="Idle time before the 'Still there?' prompt")
    face_idle_countdown_seconds: int = Field(15, description="Countdown on the prompt before the face resets")
    face_ttl_backstop_seconds: int = Field(
        60, description="Server deletes a traveler photo after this long without activity or keep-alive"
    )

    # --- Workflow ---
    policy_path: str = Field("config/policy.json")
    responses_path: str = Field("config/responses.json")
    evidence_thresholds_path: str = Field("config/evidence_thresholds.json")
    answer_passages: int = Field(5, description="Final reranked passages shown to the LLM")
    node_timeout_seconds: float = Field(30.0, description="Per-node timeout for LLM/retrieval")
    max_question_chars: int = Field(500)

    # --- LangSmith observability (optional) ---
    # Tracing is off by default. Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY to enable.
    # When disabled the kiosk is completely unaffected; no LangSmith SDK calls are made.
    langsmith_api_key: str = Field("", description="LangSmith API key; blank disables tracing")
    langsmith_tracing: bool = Field(False, description="Enable LangSmith tracing")
    langsmith_project: str = Field("dam-kiosk", description="LangSmith project name")
    langsmith_endpoint: str = Field(
        "https://api.smith.langchain.com", description="LangSmith endpoint URL"
    )

    @property
    def tracing_enabled(self) -> bool:
        """True when both the tracing flag and a non-blank API key are set."""
        return self.langsmith_tracing and bool(self.langsmith_api_key)

    # --- Ingestion ---
    manifest_path: str = Field("data/manifest.json", description="Source of truth for ingestion")
    document_root: str = Field("data", description="Root for manifest-relative paths")
    ingest_dry_run: bool = Field(False, description="Plan only; write nothing")

    # --- Chunking ---
    chunking_strategy: str = Field("markdown_headers_recursive")
    chunk_size: int = Field(
        300, description="Max characters per complete chunk (incl. headings)"
    )
    chunk_overlap: int = Field(0)
    primary_header_level: int = Field(2, description="Primary split boundary = H2")

    # --- Storage ---
    vector_db: str = Field("chroma")
    chroma_collection: str = Field("knowledge_base")
    bm25_index_path: str = Field(
        ".bm25/knowledge_base.pkl", description="Persisted sparse index"
    )

    # --- Dense retrieval ---
    embedding_provider: str = Field("openai")
    dense_top_k: int = Field(20)

    # --- Sparse retrieval ---
    sparse_retriever: str = Field("bm25")
    sparse_top_k: int = Field(20)

    # --- Fusion ---
    fusion_method: str = Field("rrf")
    rrf_k: int = Field(20)
    hybrid_top_k: int = Field(20)

    # --- Reranking ---
    reranker_provider: str = Field("cohere")
    reranker_model: str = Field("rerank-v3.5")
    reranker_top_k: int = Field(10)
    reranker_max_candidates: int = Field(50, description="Max candidates passed to the Cohere reranker")


@lru_cache
def get_settings() -> Settings:
    """Return cached settings. Raises if required secrets are missing."""
    return Settings()
