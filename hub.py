"""Small, transactional Hugging Face transport for the Actions training jobs."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import requests
import torch
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_url
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def tensor_bytes(value: Any) -> bytes:
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def load_tensors(data: bytes) -> Any:
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


class Hub:
    def __init__(self, repo_id: str, token: str | bool | None = None) -> None:
        self.repo_id = repo_id
        self.token = os.environ.get("HF_TOKEN") if token is None else token
        self.api = HfApi(token=self.token)
        self.session = requests.Session()
        if self.token:
            self.session.headers["Authorization"] = f"Bearer {self.token}"

    @staticmethod
    def _retryable(status: int | None) -> bool:
        return status in (408, 409, 412, 429) or bool(status and status >= 500)

    def head(self) -> str:
        for attempt in range(7):
            try:
                return self.api.repo_info(self.repo_id, repo_type="model").sha
            except requests.RequestException as error:
                status = getattr(error.response, "status_code", None)
                if (status is not None and not self._retryable(status)) or attempt == 6:
                    raise
                time.sleep(min(30, 2 ** attempt) + random.random())
        raise RuntimeError("Unreachable retry state")

    def read(self, path: str, revision: str = "main", optional: bool = False) -> bytes | None:
        # Resolve the immutable commit first. Requests are not cached as negative
        # entries while workers wait for a file another job is about to commit.
        url = hf_hub_url(self.repo_id, path, repo_type="model", revision=revision)
        for attempt in range(7):
            try:
                response = self.session.get(url, timeout=(20, 180))
                if response.status_code == 404 and optional:
                    return None
                if self._retryable(response.status_code) and attempt < 6:
                    time.sleep(min(30, 2 ** attempt) + random.random())
                    continue
                response.raise_for_status()
                return response.content
            except requests.RequestException as error:
                status = getattr(error.response, "status_code", None)
                if attempt == 6 or (status is not None and not self._retryable(status)):
                    raise
                time.sleep(min(30, 2 ** attempt) + random.random())
        raise RuntimeError("Unreachable retry state")

    def read_json(self, path: str, revision: str = "main", optional: bool = False) -> Any:
        data = self.read(path, revision, optional)
        return None if data is None else json.loads(data.decode("utf-8"))

    def commit(self, files: dict[str, bytes | Path], message: str) -> str:
        """Atomically publish related files; retry only concurrent-head conflicts.

        Writers use distinct paths. The coordinator alone owns state/latest.pt.
        Retried writes have identical bytes, so a lost HTTP response is harmless.
        """
        for attempt in range(9):
            parent = self.head()
            operations = [
                CommitOperationAdd(path_in_repo=name, path_or_fileobj=value if isinstance(value, bytes) else str(value))
                for name, value in files.items()
            ]
            try:
                result = self.api.create_commit(
                    self.repo_id, operations=operations, commit_message=message,
                    repo_type="model", parent_commit=parent,
                )
                return result.oid
            except requests.RequestException as error:
                status = getattr(error.response, "status_code", None)
                if (status is not None and not self._retryable(status)) or attempt == 8:
                    raise
                time.sleep(min(30, 2 ** min(attempt, 4)) + random.random())
        raise RuntimeError("Unreachable retry state")

    def create_owned(self, source_repo: str, private: bool) -> None:
        """Create only this run's new project; never adopt an unrelated old model."""
        if not self.repo_id.startswith("RabbitDaisuke/"):
            raise ValueError("The requested publication namespace is RabbitDaisuke")
        marker_path = "project.json"
        expected = {"format": 1, "source_repository": source_repo, "project": "wholebody-compact-a6fde3"}
        try:
            info = self.api.repo_info(self.repo_id, repo_type="model")
        except RepositoryNotFoundError:
            self.api.create_repo(self.repo_id, repo_type="model", private=private, exist_ok=False)
            self.commit({marker_path: json_bytes(expected)}, "Initialize new WholeBody training project")
            return
        marker = self.read_json(marker_path, revision=info.sha, optional=True)
        if marker != expected:
            raise RuntimeError(f"{self.repo_id} already exists and is not this project; refusing to reuse it")
        if bool(info.private) != private:
            raise RuntimeError(f"Unexpected visibility for {self.repo_id}; refusing to change it implicitly")


def wait_json(hub: Hub, path: str, deadline: float, *, abort_path: str | None = None) -> Any:
    while time.monotonic() < deadline:
        revision = hub.head()
        result = hub.read_json(path, revision=revision, optional=True)
        if result is not None:
            return result
        if abort_path:
            aborted = hub.read_json(abort_path, revision=revision, optional=True)
            if aborted:
                raise RuntimeError(f"Worker stopped: {aborted.get('reason', aborted.get('error_type', 'unknown'))}")
        time.sleep(3)
    raise TimeoutError(f"Timed out waiting for {path}")
