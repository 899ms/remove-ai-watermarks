"""Shared Hugging Face model-repository publication helper."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


def publish_model_folder(
    stage: Path,
    *,
    repo_id: str,
    token: str,
    message: str,
    allow_patterns: Iterable[str],
) -> str:
    """Upload an explicit file allowlist and return the Hub revision."""
    from huggingface_hub import HfApi, create_repo

    create_repo(repo_id, repo_type="model", exist_ok=True, private=False, token=token)
    commit = HfApi(token=token).upload_folder(
        folder_path=str(stage),
        repo_id=repo_id,
        repo_type="model",
        commit_message=message,
        allow_patterns=list(allow_patterns),
    )
    return getattr(commit, "oid", "") or ""
