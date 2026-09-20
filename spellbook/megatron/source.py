"""Shared Megatron source and container-path helpers."""

from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse


def is_megatron_url(value: str) -> bool:
    """Return whether *value* is a supported Git URL."""
    parsed = urlparse(value)
    return (parsed.scheme in {"git", "http", "https", "ssh"} and bool(parsed.netloc)) or (
        value.startswith("git@") and ":" in value
    )


def normalize_container_path(value: str, *, field_name: str = "megatron_container_path") -> str:
    """Validate and normalize the absolute path used inside a container."""
    path = value.rstrip("/")
    if (
        not path.startswith("/")
        or ".." in path.split("/")
        or re.fullmatch(r"/[A-Za-z0-9_./-]+", path) is None
        or len([part for part in path.split("/") if part]) < 2
    ):
        raise ValueError(
            f"{field_name} must be a shell-safe absolute path with at least two components"
        )
    return path


def source_cache_key(source: str) -> str:
    """Return the stable short key used for a cached source checkout."""
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def worktree_cache_key(source: str, commit: str = "") -> str:
    """Return the stable short key for one source and commit pair."""
    return hashlib.sha256(f"{source}\0{commit}".encode()).hexdigest()[:16]
