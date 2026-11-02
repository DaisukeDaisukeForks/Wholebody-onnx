"""Small, transactional Hugging Face transport for the Actions training jobs."""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
import httpx
import torch
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_url
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from bucket_transport import _pause, _retry_delay


HTTP_ERRORS = (requests.RequestException, HfHubHTTPError, httpx.HTTPError)


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
            except HTTP_ERRORS as error:
                status = getattr(getattr(error, "response", None), "status_code", None)
                if (status is not None and not self._retryable(status)) or attempt == 6:
                    raise
                _pause(_retry_delay(getattr(error, "response", None), attempt), "repository-head", status)
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
                    _pause(_retry_delay(response, attempt), "repository-read", response.status_code)
                    continue
                response.raise_for_status()
                return response.content
            except requests.RequestException as error:
                status = getattr(getattr(error, "response", None), "status_code", None)
                if attempt == 6 or (status is not None and not self._retryable(status)):
                    raise
                _pause(_retry_delay(getattr(error, "response", None), attempt), "repository-read", status)
        raise RuntimeError("Unreachable retry state")

    def read_json(self, path: str, revision: str = "main", optional: bool = False) -> Any:
        data = self.read(path, revision, optional)
        return None if data is None else json.loads(data.decode("utf-8"))

    def commit(self, files: dict[str, bytes | Path], message: str, *, delete_files: list[str] | None = None,
               wait_for_rate_limit: bool = True) -> str:
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
            operations.extend(CommitOperationDelete(path_in_repo=name, is_folder=False)
                              for name in (delete_files or []))
            try:
                result = self.api.create_commit(
                    self.repo_id, operations=operations, commit_message=message,
                    repo_type="model", parent_commit=parent,
                )
                return result.oid
            except HTTP_ERRORS as error:
                status = getattr(getattr(error, "response", None), "status_code", None)
                if status == 429 and not wait_for_rate_limit:
                    raise
                if (status is not None and not self._retryable(status)) or attempt == 8:
                    raise
                response = getattr(error, "response", None)
                delay = _retry_delay(response, attempt)
                if status == 429 and response is not None:
                    # The Hub's commit quota can be hourly, independently of
                    # the five-minute API limit. Never hammer it on short retries.
                    detail = response.text.lower()
                    if "commit" in detail and ("hour" in detail or "128" in detail):
                        delay = max(delay, 3600.0)
                _pause(delay, "repository-commit", status)
        raise RuntimeError("Unreachable retry state")

    def create_owned(self, source_repo: str, private: bool) -> None:
        """Create only this run's new project; never adopt an unrelated old model."""
        if not self.repo_id.startswith("RabbitDaisuke/"):
            raise ValueError("The requested publication namespace is RabbitDaisuke")
        marker_path = "project.json"
        expected = {"format": 1, "source_repository": source_repo, "project": "wholebody-pose73-a6fde3"}
        try:
            info = self.api.repo_info(self.repo_id, repo_type="model")
        except RepositoryNotFoundError:
            self.api.create_repo(self.repo_id, repo_type="model", private=private, exist_ok=False)
            self.commit({marker_path: json_bytes(expected)}, "Initialize new WholeBody training project")
            return
        marker = self.read_json(marker_path, revision=info.sha, optional=True)
        allowed_new_repositories = {
            "RabbitDaisuke/Wholebody-Pose73-20261102-a6fde3",
            "RabbitDaisuke/Wholebody-Pose73-20261102-a6fde3-checkpoints",
        }
        if marker is None and self.repo_id in allowed_new_repositories and bool(info.private) == private:
            files = self.api.list_repo_files(self.repo_id, repo_type="model", revision=info.sha)
            if set(files) <= {".gitattributes"}:
                self.commit({marker_path: json_bytes(expected)}, "Finish this task's interrupted empty-repository initialization")
                return
        if marker != expected:
            raise RuntimeError(f"{self.repo_id} already exists and is not this project; refusing to reuse it")
        if bool(info.private) != private:
            raise RuntimeError(f"Unexpected visibility for {self.repo_id}; refusing to change it implicitly")

    def compact_transport(self, phase_prefix: str) -> None:
        """Retain durable model/optimizer state, discard acknowledged exchanges.

        Applies only to the new task-owned private transport repository. Latest,
        previous-phase and initial checkpoints plus all update evidence remain.
        The caller has received peer acknowledgement before removing revisions.
        """
        if self.repo_id != "RabbitDaisuke/Wholebody-Pose73-20261102-a6fde3-checkpoints":
            raise ValueError("Transport cleanup is restricted to this new private training repository")
        info = self.api.repo_info(self.repo_id, repo_type="model")
        marker = self.read_json("project.json", revision=info.sha)
        if not info.private or marker.get("project") != "wholebody-pose73-a6fde3":
            raise RuntimeError("Unexpected repository ownership during transport cleanup")
        if not phase_prefix.startswith("runs/") or "/phase" not in phase_prefix or ".." in phase_prefix:
            raise ValueError("Invalid completed phase prefix")
        files = self.api.list_repo_files(self.repo_id, repo_type="model", revision=info.sha)
        # Sequential workflow phases guarantee older phases have also stopped.
        # Include abandoned exchanges from recovered phases, so a previous
        # failure cannot grow the retained repository indefinitely.
        temporary = [name for name in files if name.startswith("runs/")
                     and "/phase" in name and ("/weights/" in name or "/gradients/" in name)]
        if temporary:
            self.commit({}, "Remove acknowledged round exchanges; retain complete training state", delete_files=temporary)
        self.api.super_squash_history(
            self.repo_id, repo_type="model", branch="main",
            commit_message="Compact temporary transport history; retain initial, previous and latest optimizer checkpoints",
        )


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
