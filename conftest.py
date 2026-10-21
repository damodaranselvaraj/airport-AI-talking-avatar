"""Pytest bootstrap.

Sets placeholder secrets so configuration-dependent imports work under test
without a real .env, and ensures the workspace root is importable.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("TYPESAFE_API_KEY", "test-key")
os.environ.setdefault("COHERE_API_KEY", "test-key")
# Explicitly disable LangSmith tracing in tests so no traces are sent to
# LangSmith Cloud during CI and tests never fail due to connectivity issues.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
