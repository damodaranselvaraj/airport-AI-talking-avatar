"""Manifest validation: manifest.json is the sole source of ingestion scope."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ManifestError(ValueError):
    """The manifest is missing, invalid, or references unusable paths."""


class ManifestValidator:
    def __init__(self, manifest_path: str, document_root: str) -> None:
        self._manifest_path = Path(manifest_path)
        self._root = Path(document_root).resolve()
        self._includes: list[Path] | None = None
        self._excluded: set[Path] | None = None

    # -- loading -----------------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        try:
            raw = self._manifest_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ManifestError(
                f"Cannot read manifest at {self._manifest_path}: {exc}"
            ) from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ManifestError(
                f"Invalid JSON in manifest at {self._manifest_path}: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise ManifestError(
                f"Manifest at {self._manifest_path} must be a JSON object"
            )
        return data

    def _resolve(self, entry: str) -> Path:
        """Resolve a manifest-relative path to a normalized absolute path in the root."""
        candidate = (self._root / entry).resolve()
        # Entries such as "data/gates.md" are written relative to the project
        # root, i.e. the parent of a document_root named "data".
        if not candidate.exists() and Path(entry).parts[:1] == (self._root.name,):
            alt = (self._root.parent / entry).resolve()
            if alt.exists():
                candidate = alt
        if not candidate.is_relative_to(self._root):
            raise ManifestError(
                f"Manifest path '{entry}' resolves outside document root {self._root}"
            )
        return candidate

    @staticmethod
    def _entry_path(entry: Any) -> str:
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict) and isinstance(entry.get("source"), str):
            return entry["source"]
        raise ManifestError(f"Unsupported manifest entry: {entry!r}")

    def _ensure_loaded(self) -> None:
        if self._includes is not None and self._excluded is not None:
            return
        data = self._load()
        entries = data.get("documents", data.get("include"))
        if not isinstance(entries, list):
            raise ManifestError(
                f"Manifest at {self._manifest_path} has no 'documents' list"
            )
        excluded_entries = data.get("do_not_ingest", [])
        if not isinstance(excluded_entries, list):
            raise ManifestError("Manifest 'do_not_ingest' must be a list")

        includes = [self._resolve(self._entry_path(e)) for e in entries]
        excluded = {self._resolve(self._entry_path(e)) for e in excluded_entries}

        missing = [p for p in includes if p not in excluded and not p.is_file()]
        if missing:
            raise ManifestError(
                "Manifest include paths not found: " + ", ".join(str(p) for p in missing)
            )
        self._includes = includes
        self._excluded = excluded

    # -- public API --------------------------------------------------------------
    def eligible_files(self) -> list[Path]:
        """All manifest files minus do_not_ingest, as resolved absolute paths."""
        self._ensure_loaded()
        assert self._includes is not None and self._excluded is not None
        seen: set[Path] = set()
        result: list[Path] = []
        for path in self._includes:
            if path in self._excluded or path in seen:
                continue
            seen.add(path)
            result.append(path)
        return result

    def is_excluded(self, path: Path) -> bool:
        """True if path matches any do_not_ingest entry (resolved-path equality)."""
        self._ensure_loaded()
        assert self._excluded is not None
        return Path(path).resolve() in self._excluded
