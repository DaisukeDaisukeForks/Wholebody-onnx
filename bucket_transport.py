"""Mutable, commit-free transport for this task's private training exchanges.

Requires huggingface_hub==1.33.0 (and its hf-xet dependency). The SDK's
batch_bucket_files operation is explicitly NON-transactional. Binary payloads
therefore finish before any control JSON is written; JSON files are published
one at a time, in the caller's insertion order. Put the latest pointer last.

The caller must use immutable checkpoint/exchange paths and verify the SHA256
stored in a last-written pointer. ``head`` and ``revision`` only retain the old
transport interface: a bucket has no Git revision or multi-file read snapshot.
The coordinator alone writes a mutable latest pointer. Cleanup is only valid
after every peer has acknowledged that it no longer needs the phase exchanges.

Official API/source references:
https://huggingface.co/docs/huggingface_hub/guides/buckets
https://github.com/huggingface/huggingface_hub/blob/v1.33.0/src/huggingface_hub/hf_api.py
https://huggingface.co/docs/hub/rate-limits
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.parse import quote

import httpx
import requests
from huggingface_hub import HfApi
from huggingface_hub.errors import BucketBatchError, HfHubHTTPError


BUCKET_ID = "RabbitDaisuke/Wholebody-Pose73-20261102-a6fde3-exchange"
SOURCE_REPOSITORY = "DaisukeDaisuke/Wholebody-onnx"
PROJECT = "wholebody-pose73-a6fde3"
CURRENT = "bucket-current"
_T = TypeVar("_T")


def _path(path: str) -> str:
    """Only relative object paths; keep the exact string used by the caller."""
    if (not isinstance(path, str) or not path or path.startswith("/")
            or "\\" in path or "\x00" in path
            or any(part in ("", ".", "..") for part in path.split("/"))):
        raise ValueError("Bucket object paths must be nonempty relative paths")
    return path


def _retry_delay(response: Any, attempt: int) -> float:
    """Honor server reset times instead of repeatedly exhausting the same quota."""
    delay = min(30.0, float(2 ** min(attempt, 5)))
    headers = getattr(response, "headers", {}) if response is not None else {}
    retry_after = headers.get("Retry-After")
    if retry_after:
        try:
            seconds = float(retry_after)
        except (ValueError, TypeError):
            try:
                reset_at = parsedate_to_datetime(retry_after)
                if reset_at.tzinfo is None:
                    reset_at = reset_at.replace(tzinfo=timezone.utc)
                seconds = (reset_at - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                seconds = 0.0
        if math.isfinite(seconds):
            delay = max(delay, seconds)
    if getattr(response, "status_code", None) == 429:
        for value in re.findall(r"(?:^|[,;])\s*t\s*=\s*(\d+(?:\.\d+)?)",
                                headers.get("RateLimit", ""), flags=re.IGNORECASE):
            seconds = float(value)
            if math.isfinite(seconds):
                delay = max(delay, seconds)
    return max(0.0, delay) + random.uniform(0.1, 0.6)


def _pause(seconds: float, operation: str, status: int | None) -> None:
    print(json.dumps({"event": "storage_retry", "operation": operation,
                      "http_status": status, "wait_seconds": round(seconds, 2)}), flush=True)
    remaining = seconds
    while remaining > 0:
        interval = min(60.0, remaining)
        time.sleep(interval)
        remaining -= interval


class Bucket:
    """Transport confined to the new task-owned bucket, with no public writes."""

    def __init__(self, bucket_id: str, token: str | bool | None = None) -> None:
        if bucket_id != BUCKET_ID:
            raise ValueError("Bucket transport is restricted to this task's new private bucket")
        self.bucket_id = bucket_id
        self.repo_id = bucket_id  # Compatibility for diagnostic callers only.
        self.token = os.environ.get("HF_TOKEN") if token is None else token
        if not isinstance(self.token, str) or not self.token:
            raise RuntimeError("HF_TOKEN is required for private training-bucket access")
        self.api = HfApi(token=self.token)
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.token}",
                                     "Cache-Control": "no-cache", "Pragma": "no-cache"})
        self._owned = False

    @staticmethod
    def _retryable(status: int | None) -> bool:
        # Buckets have no concurrent Git-head conflict. A create409 is handled
        # by create_owned's ownership check rather than retried as a write.
        return status in (408, 429) or bool(status and status >= 500)

    def _call(self, operation: Callable[[], _T], description: str) -> _T:
        for attempt in range(9):
            try:
                return operation()
            except (HfHubHTTPError, httpx.HTTPError, requests.RequestException) as error:
                response = getattr(error, "response", None)
                status = getattr(response, "status_code", None)
                # HTTP200 can still report failed bucket operations. Retrying
                # identical immutable uploads is safe; never advance a pointer
                # until the complete upload call has returned successfully.
                partial_failure = isinstance(error, BucketBatchError)
                if attempt == 8 or (status is not None and not self._retryable(status)
                                    and not partial_failure):
                    raise
                _pause(_retry_delay(response, attempt), description, status)
        raise RuntimeError("Unreachable retry state")

    @staticmethod
    def _marker(source_repo: str = SOURCE_REPOSITORY) -> dict[str, Any]:
        return {"format": 1, "source_repository": source_repo, "project": PROJECT}

    def head(self) -> str:
        """Return an interface sentinel, without an API call or snapshot claim."""
        return CURRENT

    def read(self, path: str, revision: str = "main", optional: bool = False) -> bytes | None:
        """Read the current object. ``revision`` is ignored, never a Git SHA."""
        path = _path(path)
        del revision
        # The official SDK uses this exact resolver route for bucket metadata.
        # Direct GET avoids one bucket_info request per polling iteration and
        # does not retain negative entries in a local Hub/filesystem cache.
        url = f"https://huggingface.co/buckets/{self.bucket_id}/resolve/{quote(path, safe='')}"

        def download() -> bytes | None:
            with self.session.get(url, timeout=(20, 180)) as response:
                if response.status_code == 404 and optional:
                    return None
                response.raise_for_status()
                return response.content

        return self._call(download, f"read:{path}")

    def read_json(self, path: str, revision: str = "main", optional: bool = False) -> Any:
        data = self.read(path, revision=revision, optional=optional)
        return None if data is None else json.loads(data.decode("utf-8"))

    def _verify_owned(self) -> None:
        info = self._call(lambda: self.api.bucket_info(self.bucket_id), "bucket-info")
        if not info.private or self.read_json("project.json", optional=True) != self._marker():
            raise RuntimeError("The bucket is not this task's private training storage")
        self._owned = True

    def create_owned(self, source_repo: str) -> None:
        """Create only the named new project; accept its own interrupted setup."""
        if source_repo != SOURCE_REPOSITORY:
            raise ValueError("Unexpected source repository for this training bucket")
        try:
            info = self._call(lambda: self.api.bucket_info(self.bucket_id), "bucket-info")
        except HfHubHTTPError as error:
            if getattr(error.response, "status_code", None) != 404:
                raise
            try:
                self._call(lambda: self.api.create_bucket(self.bucket_id, private=True, exist_ok=False),
                           "create-private-bucket")
            except HfHubHTTPError as creation_error:
                # The first request can have succeeded before a connection was
                # lost. A conflict only allows the ownership/empty-bucket checks
                # below; it never adopts an existing unrelated bucket.
                if getattr(creation_error.response, "status_code", None) != 409:
                    raise
            info = self._call(lambda: self.api.bucket_info(self.bucket_id), "bucket-info")
        if not info.private:
            raise RuntimeError("Unexpected public bucket; refusing to change its visibility")
        marker = self.read_json("project.json", optional=True)
        if marker is None:
            existing = self._call(lambda: list(self.api.list_bucket_tree(self.bucket_id, recursive=True)),
                                  "list-uninitialized-bucket")
            if existing:
                raise RuntimeError("The named bucket is nonempty and has no project ownership marker")
            content = (json.dumps(self._marker(), sort_keys=True, indent=2) + "\n").encode("utf-8")
            self._call(lambda: self.api.batch_bucket_files(self.bucket_id, add=[(content, "project.json")]),
                       "initialize-project-marker")
        elif marker != self._marker():
            raise RuntimeError("The named bucket belongs to another project; refusing to reuse it")
        self._owned = True

    def commit(self, files: dict[str, bytes | Path], message: str,
               *, delete_files: list[str] | None = None) -> str:
        """Stage payloads, then each JSON in insertion order, then optional deletes.

        This compatibility name does NOT create a Git commit. Immutable payload
        names plus a final SHA256 pointer are the caller's consistency protocol.
        JSON files can be partially published; order any dependent pointers last.
        """
        del message  # Bucket writes have no commit-message/history equivalent.
        if not self._owned:
            self._verify_owned()
        binary: list[tuple[bytes | Path, str]] = []
        controls: list[tuple[bytes | Path, str]] = []
        for name, value in files.items():
            name = _path(name)
            if not isinstance(value, (bytes, Path)):
                raise TypeError("Bucket contents must be bytes or a pathlib.Path")
            if name == "project.json":
                raise ValueError("The project ownership marker cannot be overwritten")
            (controls if name.lower().endswith(".json") else binary).append((value, name))
        deletions = [_path(name) for name in (delete_files or [])]
        if "project.json" in deletions or any(name in files for name in deletions):
            raise ValueError("Cannot delete ownership metadata or a file written by the same operation")
        if binary:
            self._call(lambda: self.api.batch_bucket_files(self.bucket_id, add=binary), "stage-binary-payloads")
        for value, name in controls:
            self._call(lambda value=value, name=name: self.api.batch_bucket_files(
                self.bucket_id, add=[(value, name)]), f"publish-control:{name}")
        if deletions:
            # Resolve remaining paths again on each retry. This makes a lost
            # successful delete response safe even if deleting absent paths is
            # reported as an error by the service.
            def remove_remaining() -> None:
                remaining = [item.path for item in self.api.get_bucket_paths_info(self.bucket_id, deletions)]
                if remaining:
                    self.api.batch_bucket_files(self.bucket_id, delete=remaining)
            self._call(remove_remaining, "remove-retired-objects")
        return CURRENT

    def compact_transport(self, phase_prefix: str) -> None:
        """Delete this acknowledged phase's weights/gradients; retain all evidence.

        The coordinator must call this only after EVERY peer acknowledges.
        Older or abandoned phases are not swept implicitly. There is no Git
        history to squash; checkpoint retention remains the caller's policy.
        """
        phase_prefix = _path(phase_prefix)
        if not re.fullmatch(r"runs/[^/]+/phase[0-9]+", phase_prefix):
            raise ValueError("Cleanup requires one completed runs/<run-key>/phaseN prefix")
        self._verify_owned()
        temporary: list[str] = []
        for directory in ("weights", "gradients"):
            prefix = f"{phase_prefix}/{directory}/"
            entries = self._call(lambda prefix=prefix: list(self.api.list_bucket_tree(
                self.bucket_id, prefix=prefix, recursive=True)), f"list-retired:{directory}")
            temporary.extend(item.path for item in entries
                             if getattr(item, "type", None) == "file" and item.path.startswith(prefix))
        if temporary:
            self.commit({}, "Remove acknowledged phase exchanges", delete_files=temporary)
