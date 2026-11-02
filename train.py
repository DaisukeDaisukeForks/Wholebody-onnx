"""Real multi-runner gradient training, checkpoint continuation, and publication."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import signal
import tarfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import default_collate

from data import ASPsetDataset, CocoWholeBodyDataset, PREPARED_FILES, ensure_images, prepare_data
from hub import Hub, atomic_write, json_bytes, load_tensors, sha256, tensor_bytes, wait_json
from model import WholeBodyModel


SOURCE_REPOSITORY = "DaisukeDaisuke/Wholebody-onnx"
ROOT = Path(__file__).resolve().parent
STOP_REQUESTED = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stop_signal(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def run_key() -> str:
    value = os.environ.get("RUN_KEY")
    if not value or any(character not in "0123456789-" for character in value):
        raise RuntimeError("RUN_KEY must be GitHub run_id-run_attempt; start this from Actions")
    return value


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["workers"] != 2:
        raise ValueError("The workflow matrix and configuration must both have two workers")
    if value["hf_repo_id"] == value["checkpoint_repo_id"]:
        raise ValueError("Public inference files and private training state use separate repositories")
    return value


def training_identity(config: dict[str, Any]) -> str:
    keys = ("workers", "seed", "batch2d", "batch3d", "accumulate", "loss3d_weight",
            "learning_rate", "weight_decay", "clip_grad_norm", "model", "train_limit",
            "val_limit", "max_3d_frames", "frame_stride", "joint_schema")
    return sha256(json_bytes({key: config[key] for key in keys}))


def log(work: Path, name: str, event: dict[str, Any]) -> None:
    path = work / "logs" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"utc": utc_now(), **event}
    line = json.dumps(event, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def optimizer_for(model: WholeBodyModel, config: dict[str, Any]) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )


def prepared_bundle(data_root: Path) -> bytes:
    memory = io.BytesIO()
    with tarfile.open(fileobj=memory, mode="w:gz") as archive:
        for filename in PREPARED_FILES:
            archive.add(data_root / filename, arcname=filename, recursive=False)
    return memory.getvalue()


def unpack_prepared(data: bytes, data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        if {entry.name for entry in members} != set(PREPARED_FILES):
            raise ValueError("Unexpected prepared-data archive members")
        for entry in members:
            if not entry.isfile():
                raise ValueError("Prepared-data members must be regular files")
            source = archive.extractfile(entry)
            if source is None:
                raise ValueError("Missing prepared-data member")
            atomic_write(data_root / entry.name, source.read())


def bootstrap(config: dict[str, Any], work: Path) -> None:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is missing. Add the Hugging Face write token as this repository's Actions secret HF_TOKEN.")
    checkpoint_hub = Hub(config["checkpoint_repo_id"])
    publication_hub = Hub(config["hf_repo_id"])
    checkpoint_hub.create_owned(SOURCE_REPOSITORY, private=True)
    publication_hub.create_owned(SOURCE_REPOSITORY, private=False)
    head = checkpoint_hub.head()
    previous_bytes = checkpoint_hub.read("state/latest.pt", revision=head, optional=True)
    prepared = checkpoint_hub.read("data/prepared.tar.gz", revision=head, optional=True)
    data_root = work / "data"
    if prepared is None and config.get("prepared_source"):
        # Reuse only the actual data prepared by this session's cancelled run.
        # No model or optimizer checkpoint is read from that source repository.
        source = config["prepared_source"]
        source_hub = Hub(source["repo_id"])
        source_revision = source_hub.head()
        source_project = source_hub.read_json("project.json", revision=source_revision)
        expected_source = {"format": 1, "source_repository": SOURCE_REPOSITORY, "project": "wholebody-parallel-a6fde3"}
        if source_project != expected_source:
            raise RuntimeError("Prepared data source is not this session's original preparation")
        source_run = source_hub.read_json(f"runs/{source['run_key']}/bootstrap.json", revision=source_revision)
        if source_run["data_sha256"] != source["sha256"]:
            raise RuntimeError("Prepared-data provenance differs from the recorded preparation")
        prepared = source_hub.read("data/prepared.tar.gz", revision=source_revision)
        if sha256(prepared) != source["sha256"]:
            raise RuntimeError("Reused prepared-data checksum mismatch")
        log(work, "bootstrap", {"event": "reused_current_session_data", "source_revision": source_revision,
                               "source_run": source["run_key"], "sha256": source["sha256"]})
    if prepared is None:
        prepare_data(
            data_root, train_limit=config["train_limit"], val_limit=config["val_limit"],
            download_images=False, workers=8, frame_stride=config["frame_stride"],
            max_3d_frames=config["max_3d_frames"],
        )
        prepared = prepared_bundle(data_root)
    else:
        unpack_prepared(prepared, data_root)
    data_digest = sha256(prepared)
    if previous_bytes is None:
        torch.manual_seed(config["seed"])
        model = WholeBodyModel(config["model"], pretrained=True)
        optimizer = optimizer_for(model, config)
        checkpoint = {
            "format": 1, "source_repository": SOURCE_REPOSITORY,
            "model": model.state_dict(), "model_config": model.config,
            "optimizer": optimizer.state_dict(), "step": 0,
            "training_identity": training_identity(config), "data_sha256": data_digest,
            "seed": config["seed"], "created_at": utc_now(),
            "contribution_counts": {str(rank): 0 for rank in range(config["workers"])},
            "labels2d": 0, "labels3d": 0, "parallel_overlap_steps": 0,
            "history": [], "resume_history": [],
        }
        previous_bytes = tensor_bytes(checkpoint)
    else:
        checkpoint = load_tensors(previous_bytes)
        if checkpoint["training_identity"] != training_identity(config):
            raise RuntimeError("Training/model configuration changed; cannot silently resume incompatible state")
        if checkpoint["data_sha256"] != data_digest:
            raise RuntimeError("Prepared training data differs from the checkpoint")
    manifest = {
        "run_key": run_key(), "source_repository": SOURCE_REPOSITORY,
        "source_sha": os.environ.get("GITHUB_SHA"), "initial_step": checkpoint["step"],
        "target_step": checkpoint["step"] + 2 * config["rounds_per_phase"],
        "data_sha256": data_digest, "training_identity": training_identity(config),
        "created_at": utc_now(), "workers": config["workers"],
    }
    checkpoint_hub.commit({
        "data/prepared.tar.gz": prepared,
        "data/dataset_manifest.json": data_root / "dataset_manifest.json",
        "state/latest.pt": previous_bytes,
        f"runs/{run_key()}/bootstrap.json": json_bytes(manifest),
    }, f"Prepare run {run_key()} at step {checkpoint['step']}")
    atomic_write(work / "status.json", json_bytes(manifest))
    log(work, "bootstrap", {"event": "prepared", **manifest})


class ShardStream:
    """Disjoint deterministic shards; the global update determines each cursor."""

    def __init__(self, dataset: Any, rank: int, world_size: int, seed: int) -> None:
        self.dataset = dataset
        self.indices = torch.arange(rank, len(dataset), world_size)
        self.seed = seed + 97 * rank
        self.epoch = -1
        self.order = self.indices
        if not len(self.indices):
            raise ValueError("A worker has an empty data shard")

    def batch(self, offset: int, size: int) -> dict[str, Any]:
        selected = []
        for position in range(offset, offset + size):
            epoch, index = divmod(position, len(self.indices))
            if epoch != self.epoch:
                generator = torch.Generator().manual_seed(self.seed + epoch * 10007)
                self.order = self.indices[torch.randperm(len(self.indices), generator=generator)]
                self.epoch = epoch
            selected.append(self.dataset[int(self.order[index])])
        return default_collate(selected)


def trainable_parameters(model: WholeBodyModel) -> dict[str, torch.nn.Parameter]:
    return {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}


def parameter_digest(model: WholeBodyModel) -> str:
    digest = hashlib.sha256()
    for name, parameter in trainable_parameters(model).items():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_snapshot(model: WholeBodyModel, snapshot: dict[str, Any]) -> None:
    parameters = trainable_parameters(model)
    if set(snapshot["parameters"]) != set(parameters):
        raise ValueError("Snapshot parameter names differ")
    with torch.no_grad():
        for name, parameter in parameters.items():
            incoming = snapshot["parameters"][name]
            if incoming.shape != parameter.shape or not torch.isfinite(incoming).all():
                raise ValueError(f"Invalid snapshot tensor: {name}")
            parameter.copy_(incoming)
    if parameter_digest(model) != snapshot["parameter_sha256"]:
        raise ValueError("Loaded model differs from the common round weights")


def gradient_packet(
    model: WholeBodyModel, stream2d: ShardStream, stream3d: ShardStream,
    config: dict[str, Any], rank: int, phase: int, step: int,
    snapshot_hash: str, scheduled_start: float,
) -> dict[str, Any]:
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(config["seed"] + step * config["workers"] + rank)
    remaining = scheduled_start - time.time()
    if remaining > 0:
        time.sleep(min(remaining, 20))
    totals: dict[str, float] = {}
    started = time.time()
    for microstep in range(config["accumulate"]):
        offset2d = (step * config["accumulate"] + microstep) * config["batch2d"]
        offset3d = (step * config["accumulate"] + microstep) * config["batch3d"]
        batch2d = stream2d.batch(offset2d, config["batch2d"])
        batch3d = stream3d.batch(offset3d, config["batch3d"])
        loss2d, metrics2d = model.loss_2d(batch2d)
        loss3d, metrics3d = model.loss_3d(batch3d)
        loss = (loss2d + config["loss3d_weight"] * loss3d) / config["accumulate"]
        if not torch.isfinite(loss):
            raise FloatingPointError("A real training loss is non-finite")
        loss.backward()
        for name, value in {**metrics2d, **metrics3d}.items():
            scale = 1 if name.startswith("labeled") else config["accumulate"]
            totals[name] = totals.get(name, 0.0) + float(value) / scale
    finished = time.time()
    gradients = {}
    for name, parameter in trainable_parameters(model).items():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Missing or non-finite gradient: {name}")
        gradients[name] = parameter.grad.detach().clone()
    return {
        "run_key": run_key(), "phase": phase, "step": step, "worker_id": rank,
        "job_id": os.environ.get("GITHUB_JOB"), "runner_name": os.environ.get("RUNNER_NAME"),
        "snapshot_sha256": snapshot_hash, "parameters_sha256": parameter_digest(model),
        "samples2d": config["batch2d"] * config["accumulate"],
        "samples3d": config["batch3d"] * config["accumulate"],
        "compute_started": started, "compute_finished": finished,
        "metrics": totals, "gradients": gradients,
    }


def wait_packets(hub: Hub, prefix: str, step: int, workers: int, deadline: float) -> list[dict[str, Any]]:
    packets: dict[int, Any] = {}
    while len(packets) < workers and time.monotonic() < deadline:
        revision = hub.head()
        for rank in range(workers):
            if rank in packets:
                continue
            data = hub.read(f"{prefix}/gradients/{step:08d}-{rank}.pt", revision=revision, optional=True)
            if data is not None:
                packets[rank] = load_tensors(data)
                continue
            failure = hub.read_json(f"{prefix}/failures/{rank}.json", revision=revision, optional=True)
            if failure:
                raise RuntimeError(f"Worker {rank} stopped: {failure['error_type']}")
        if len(packets) < workers:
            time.sleep(3)
    if len(packets) != workers:
        raise TimeoutError(f"Missing gradient contributions for step {step}: {sorted(set(range(workers)) - set(packets))}")
    return [packets[rank] for rank in range(workers)]


def apply_gradients(
    model: WholeBodyModel, optimizer: torch.optim.AdamW, checkpoint: dict[str, Any],
    packets: list[dict[str, Any]], config: dict[str, Any], phase: int,
    snapshot_hash: str, expected_parameters_hash: str,
) -> dict[str, Any]:
    parameters = trainable_parameters(model)
    seen = set()
    for packet in packets:
        rank = packet["worker_id"]
        if rank in seen or rank not in range(config["workers"]):
            raise ValueError("Duplicate or unknown worker contribution")
        seen.add(rank)
        expected = (run_key(), phase, checkpoint["step"], snapshot_hash, expected_parameters_hash)
        observed = tuple(packet[key] for key in ("run_key", "phase", "step", "snapshot_sha256", "parameters_sha256"))
        if observed != expected:
            raise ValueError("Gradient contribution was computed from a different model/round")
        if packet["samples2d"] != config["batch2d"] * config["accumulate"] or packet["samples3d"] != config["batch3d"] * config["accumulate"]:
            raise ValueError("Worker task sample counts differ; cannot apply equal-worker averaging")
        if packet["metrics"]["labeled2d"] <= 0 or packet["metrics"]["labeled3d"] <= 0:
            raise ValueError("A worker has not contributed actual labels for both tasks")
        if set(packet["gradients"]) != set(parameters):
            raise ValueError("Gradient parameter names differ")
        for name, parameter in parameters.items():
            gradient = packet["gradients"][name]
            if gradient.shape != parameter.shape or not torch.isfinite(gradient).all():
                raise ValueError(f"Malformed gradient: {name}")
    if seen != set(range(config["workers"])):
        raise ValueError("Every worker must contribute to each common update")
    optimizer.zero_grad(set_to_none=True)
    for name, parameter in parameters.items():
        # Fixed, equal task sample counts make this exactly the gradient of
        # mean(global 2D examples) + lambda * mean(global 3D examples).
        parameter.grad = torch.stack([packet["gradients"][name] for packet in packets]).mean(dim=0)
    norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), config["clip_grad_norm"], error_if_nonfinite=True)
    optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in parameters.values()):
        raise FloatingPointError("Optimizer produced non-finite model parameters")
    overlap = max(0.0, min(packet["compute_finished"] for packet in packets) - max(packet["compute_started"] for packet in packets))
    evidence = {
        "step": checkpoint["step"] + 1, "phase": phase, "run_key": run_key(),
        "gradient_norm_before_clip": float(norm), "compute_overlap_seconds": overlap,
        "base_snapshot_sha256": snapshot_hash,
        "contributors": [{key: value for key, value in packet.items() if key != "gradients"} for packet in packets],
    }
    checkpoint["step"] += 1
    checkpoint["model"] = model.state_dict()
    checkpoint["optimizer"] = optimizer.state_dict()
    checkpoint["updated_at"] = utc_now()
    checkpoint["source_sha"] = os.environ.get("GITHUB_SHA")
    checkpoint["parallel_overlap_steps"] += int(overlap > 0)
    checkpoint["history"].append(evidence)
    for packet in packets:
        checkpoint["contribution_counts"][str(packet["worker_id"]) ] += 1
        checkpoint["labels2d"] += int(packet["metrics"]["labeled2d"])
        checkpoint["labels3d"] += int(packet["metrics"]["labeled3d"])
    return evidence


def worker(config: dict[str, Any], work: Path, phase: int, rank: int) -> None:
    hub = Hub(config["checkpoint_repo_id"])
    prefix = f"runs/{run_key()}/phase{phase}"
    deadline = time.monotonic() + 60 * config["worker_minutes"]
    wait_seconds = 60 * config["worker_wait_minutes"]
    bootstrap_manifest = hub.read_json(f"runs/{run_key()}/bootstrap.json")
    if rank == 0:
        revision = hub.head()
        checkpoint = load_tensors(hub.read("state/latest.pt", revision=revision))
        plan = {
            "checkpoint_revision": revision, "start_step": checkpoint["step"],
            "target_step": bootstrap_manifest["initial_step"] + phase * config["rounds_per_phase"],
            "run_key": run_key(), "phase": phase,
        }
        hub.commit({f"{prefix}/plan.json": json_bytes(plan)}, f"Start phase {phase} from saved step {checkpoint['step']}")
    else:
        plan = wait_json(
            hub, f"{prefix}/plan.json", min(deadline, time.monotonic() + wait_seconds),
            abort_path=f"{prefix}/failures/0.json",
        )
        checkpoint = load_tensors(hub.read("state/latest.pt", revision=plan["checkpoint_revision"]))
    if checkpoint["training_identity"] != training_identity(config):
        raise ValueError("Worker configuration differs from the common checkpoint")
    data_bytes = hub.read("data/prepared.tar.gz", revision=plan["checkpoint_revision"])
    if sha256(data_bytes) != checkpoint["data_sha256"]:
        raise ValueError("Worker data differs from the common checkpoint")
    data_root = work / "data"
    unpack_prepared(data_bytes, data_root)
    ensure_images(data_root, split="train", rank=rank, world_size=config["workers"], workers=8)
    model = WholeBodyModel(checkpoint["model_config"], pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer = optimizer_for(model, config)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if rank == 0 and checkpoint["step"] > 0:
        if not optimizer.state:
            raise RuntimeError("A trained checkpoint is missing its optimizer state")
        checkpoint["resume_history"].append({
            "run_key": run_key(), "phase": phase, "restored_step": checkpoint["step"],
            "optimizer_state_entries": len(optimizer.state), "utc": utc_now(),
        })
    stream2d = ShardStream(CocoWholeBodyDataset(data_root, "train"), rank, config["workers"], config["seed"])
    stream3d = ShardStream(ASPsetDataset(data_root, "train"), rank, config["workers"], config["seed"] + 1000003)
    ready = {
        "worker_id": rank, "phase": phase, "restored_step": checkpoint["step"],
        "runner": os.environ.get("RUNNER_NAME"), "cpu": platform.processor(),
        "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
        "shard2d": len(stream2d.indices), "shard3d": len(stream3d.indices), "utc": utc_now(),
    }
    hub.commit({f"{prefix}/ready/{rank}.json": json_bytes(ready)}, f"Worker {rank} ready in phase {phase}")
    log(work, f"worker-{phase}-{rank}", {"event": "ready", **ready})
    if rank == 0:
        for other_rank in range(config["workers"]):
            wait_json(
                hub, f"{prefix}/ready/{other_rank}.json", min(deadline, time.monotonic() + wait_seconds),
                abort_path=f"{prefix}/failures/{other_rank}.json",
            )
    step = plan["start_step"]
    while step < plan["target_step"]:
        if rank == 0:
            if STOP_REQUESTED or time.monotonic() > deadline - 180:
                hub.commit({f"{prefix}/finished.json": json_bytes({"step": step, "target_step": plan["target_step"], "reason": "time_budget"})}, "Preserve checkpoint before runner deadline")
                return
            snapshot = {
                "step": step, "phase": phase, "run_key": run_key(),
                "parameters": {name: parameter.detach().clone() for name, parameter in trainable_parameters(model).items()},
                "parameter_sha256": parameter_digest(model),
            }
            snapshot_data = tensor_bytes(snapshot)
            snapshot_hash = sha256(snapshot_data)
            state = {"step": step, "snapshot_sha256": snapshot_hash, "compute_after": time.time() + 12}
            hub.commit({f"{prefix}/weights/{step:08d}.pt": snapshot_data, f"{prefix}/rounds/{step:08d}.json": json_bytes(state)}, f"Broadcast common weights for step {step}")
        else:
            while time.monotonic() < deadline:
                revision = hub.head()
                state = hub.read_json(f"{prefix}/rounds/{step:08d}.json", revision=revision, optional=True)
                if state is not None:
                    snapshot_data = hub.read(f"{prefix}/weights/{step:08d}.pt", revision=revision)
                    break
                finished = hub.read_json(f"{prefix}/finished.json", revision=revision, optional=True)
                if finished:
                    return
                failure = hub.read_json(f"{prefix}/failures/0.json", revision=revision, optional=True)
                if failure:
                    raise RuntimeError("Coordinator stopped; saved checkpoint remains available")
                time.sleep(3)
            else:
                raise TimeoutError("Coordinator did not broadcast the next round before the worker deadline")
            snapshot_hash = sha256(snapshot_data)
            if snapshot_hash != state["snapshot_sha256"]:
                raise ValueError("Broadcast weights checksum mismatch")
            snapshot = load_tensors(snapshot_data)
            if (snapshot["run_key"], snapshot["phase"], snapshot["step"]) != (run_key(), phase, step):
                raise ValueError("Wrong common round snapshot")
            load_snapshot(model, snapshot)
        packet = gradient_packet(model, stream2d, stream3d, config, rank, phase, step, snapshot_hash, state["compute_after"])
        hub.commit({f"{prefix}/gradients/{step:08d}-{rank}.pt": tensor_bytes(packet)}, f"Worker {rank} gradient for step {step}")
        log(work, f"worker-{phase}-{rank}", {"event": "gradient", **{key: value for key, value in packet.items() if key != "gradients"}})
        if rank == 0:
            packets = wait_packets(hub, prefix, step, config["workers"], min(deadline, time.monotonic() + wait_seconds))
            evidence = apply_gradients(model, optimizer, checkpoint, packets, config, phase, snapshot_hash, snapshot["parameter_sha256"])
            latest = tensor_bytes(checkpoint)
            atomic_write(work / "latest.pt", latest)
            # One commit contains both the complete optimizer checkpoint and its
            # progress record. An update becomes visible only after this commit.
            hub.commit({"state/latest.pt": latest, "state/progress.json": json_bytes({
                "step": checkpoint["step"], "run_key": run_key(), "phase": phase,
                "checkpoint_sha256": sha256(latest), "last_update": evidence,
            })}, f"Apply both worker gradients: common AdamW step {checkpoint['step']}")
            log(work, f"worker-{phase}-{rank}", {"event": "common_update", **evidence})
        step += 1
    if rank == 0:
        hub.commit({f"{prefix}/finished.json": json_bytes({"step": step, "target_step": plan["target_step"], "reason": "target_reached"})}, f"Complete phase {phase} at step {step}")
    else:
        wait_json(hub, f"{prefix}/finished.json", min(deadline, time.monotonic() + wait_seconds), abort_path=f"{prefix}/failures/0.json")


def publish(config: dict[str, Any], work: Path) -> None:
    from export_model import export
    from joint_schema import COMPACT_COCO_IDS, COMPACT_JOINT_NAMES

    checkpoint_hub = Hub(config["checkpoint_repo_id"])
    revision = checkpoint_hub.head()
    checkpoint_data = checkpoint_hub.read("state/latest.pt", revision=revision)
    checkpoint = load_tensors(checkpoint_data)
    manifest = checkpoint_hub.read_json(f"runs/{run_key()}/bootstrap.json", revision=revision)
    if checkpoint["step"] <= manifest["initial_step"]:
        raise RuntimeError("This run has not committed a new training update; the prior checkpoint is retained without a new publication")
    if checkpoint["step"] <= 0 or checkpoint["labels2d"] <= 0 or checkpoint["labels3d"] <= 0:
        raise RuntimeError("No learned 2D and 3D checkpoint is available to publish")
    if any(checkpoint["contribution_counts"].get(str(rank), 0) <= 0 for rank in range(config["workers"])):
        raise RuntimeError("The checkpoint does not contain every worker's contribution")
    if checkpoint["parallel_overlap_steps"] <= 0:
        raise RuntimeError("No overlapping worker computation was recorded; cannot claim parallel training")
    if not checkpoint["resume_history"]:
        raise RuntimeError("No actual resumed training is recorded yet")
    data_root = work / "data"
    prepared = checkpoint_hub.read("data/prepared.tar.gz", revision=revision)
    if sha256(prepared) != checkpoint["data_sha256"]:
        raise ValueError("Publication data differs from training data")
    unpack_prepared(prepared, data_root)
    atomic_write(work / "latest.pt", checkpoint_data)
    ensure_images(data_root, split="val", workers=8)
    output = work / "release"
    metrics = export(work / "latest.pt", data_root, output)
    training_report = {
        "algorithm": "Synchronous mean gradients, one central AdamW update; no checkpoint weight averaging",
        "source_repository": SOURCE_REPOSITORY, "source_sha": os.environ.get("GITHUB_SHA"),
        "actions_run_url": f"https://github.com/{SOURCE_REPOSITORY}/actions/runs/{os.environ.get('GITHUB_RUN_ID')}",
        "run_key": run_key(), "initial_step": manifest["initial_step"], "step": checkpoint["step"],
        "target_step": manifest["target_step"], "target_reached": checkpoint["step"] >= manifest["target_step"],
        "contribution_counts": checkpoint["contribution_counts"], "parallel_overlap_steps": checkpoint["parallel_overlap_steps"],
        "joint_schema": config["joint_schema"], "learned_2d_points": len(COMPACT_COCO_IDS),
        "learned_2d_joint_names": list(COMPACT_JOINT_NAMES),
        "labels2d": checkpoint["labels2d"], "labels3d": checkpoint["labels3d"],
        "resume_history": checkpoint["resume_history"], "updates": checkpoint["history"],
        "checkpoint_repository": config["checkpoint_repo_id"], "checkpoint_revision": revision,
        "checkpoint_sha256": sha256(checkpoint_data), "data_sha256": checkpoint["data_sha256"],
        "resources": "Two standard ubuntu-24.04 GitHub-hosted CPU runners per training phase",
        "limitations": "Initial research model. No accuracy threshold or end-to-end RGB-to-3D accuracy is claimed.",
    }
    atomic_write(output / "training_report.json", json_bytes(training_report))
    published_data_manifest = json.loads((data_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    published_data_manifest["model_supervision"] = {
        "schema": config["joint_schema"], "source_annotation_points": 133,
        "learned_2d_points": len(COMPACT_COCO_IDS), "selected_coco_ids": list(COMPACT_COCO_IDS),
        "joint_names": list(COMPACT_JOINT_NAMES),
        "note": "The source bundle retains all annotations; only selected targets enter this model's 2D loss.",
    }
    atomic_write(output / "dataset_manifest.json", json_bytes(published_data_manifest))
    for name in ("MODEL_LICENSE.md", "THIRD_PARTY.md", "data_sources.json"):
        atomic_write(output / name, (ROOT / name).read_bytes())
    atomic_write(output / "CODE_LICENSE", (ROOT / "LICENSE").read_bytes())
    atomic_write(output / "LICENSE", (ROOT / "MODEL_LICENSE.md").read_bytes())
    card = f"""---
license: other
license_name: wholebody-research
license_link: MODEL_LICENSE.md
library_name: onnx
pipeline_tag: keypoint-detection
tags:
- wholebody
- pose-estimation
- onnx
- research
---
# Compact WholeBody 2D + relative body 3D

An actually trained research model, generated by [GitHub Actions](https://github.com/{SOURCE_REPOSITORY}/actions).
This publication contains {checkpoint['step']} common AdamW updates. Both workers contributed gradients to each update.
The recorded computation intervals overlap in {checkpoint['parallel_overlap_steps']} of these updates.
The target for this run was {manifest['target_step']} updates; target reached: {checkpoint['step'] >= manifest['target_step']}.

## Download

- [wholebody.onnx](https://huggingface.co/{config['hf_repo_id']}/resolve/main/wholebody.onnx)
- [Input/output and all joint IDs](schema.json)
- [Actual training contributions and resume records](training_report.json)
- [Recorded validation and ONNX output differences](metrics.json)

## Input and output

Input `image`: float32 `[batch,3,192,144]`, RGB in [0,1], one person crop with width/height 0.75. ImageNet normalization is internal.
Output `keypoints2d`: `[batch,73,2]`, crop-normalized x-right/y-down.
Output `confidence2d`: `[batch,73]`, heatmap peak probability, not calibrated visibility.
Output `body3d`: `[batch,12,3]`, meters, camera x-right/y-down/z-away, relative to the hip midpoint.
Output `geometry2d`: `[batch,8,2]`, skeleton-derived hand reference points, hand directions and foot axes, with exact definitions in `schema.json`.
Output `geometry_confidence`: `[batch,8]`, minimum concentration score among each geometry row's source joints, not calibrated visibility.
The twelve 3D joints match COCO IDs 5 through 16. Absolute camera distance, global root movement, and 3D face/finger/toe joints are not outputs.

## Actual training data and initial weights

The original COCO-WholeBody annotation has 133 points; only 73 selected targets are learned and emitted: 12 body joints, 40 finger joints, six foot points and 15 selected face points. Wrists are shared between body and hand skeletons.
ASPset-510 calibrated 3D joints supervise the twelve-joint lifting branch.
The truncated backbone starts with official torchvision MobileNetV3-small ImageNet weights; the compact head and lifter are newly trained in this project. No checkpoint from the superseded 133-point model is loaded.
The exact selected data, transformations, source URLs and image license metadata are in [dataset_manifest.json](dataset_manifest.json).
Separate CMU Foot, AIST++ and Human3.6M datasets were not used in this run.

## Inference cost and simultaneous inputs

The network shares one truncated feature extractor, a 32-channel stride-8 location head with subcell offsets, and a 128-wide body lifter.
Different camera/person crops can occupy the dynamic batch dimension in one ONNX invocation. Spatial features and predictions remain independent per crop.
The exported `resource_usage.json` records the actual model's arithmetic and activation sizes. These counts are not a measurement of GPU utilization or device-specific latency.
Face output includes mouth, nose tip, eye corners, eyelids and cheek-outline proxies. It does not invent pupil/gaze or cheek-surface ground truth.

## Recorded results and scope

2D crop-normalized mean error: {metrics['normalized_2d_error']:.6f}; PCK at 0.05 crop distance: {metrics['pck_at_0_05_crop']:.6f}.
Lifter MPJPE from ASPset ground-truth projected 2D: {metrics['lifter_mpjpe_mm_from_ground_truth_2d']:.3f} mm.
The latter is a lifting-only measurement, not end-to-end RGB-to-3D accuracy. These measurements do not establish production tracking quality.

## License, credits and distribution

Model permissions and restrictions: [MODEL_LICENSE.md](MODEL_LICENSE.md). This model is distributed for noncommercial research and education under that custom license.
COCO-WholeBody's upstream CC BY 4.0 annotation notice and separate research/noncommercial terms both apply to the source data; ASPset-510 is CC0 1.0.
COCO image licenses vary. Torchvision code and pretrained weight provenance are documented separately. Full authors, papers, sources and transformations: [THIRD_PARTY.md](THIRD_PARTY.md).
New project source code retains the repository's Unlicense; this does not relicense external data or pretrained weights.
Source distribution: https://github.com/{SOURCE_REPOSITORY}
Model distribution: https://huggingface.co/{config['hf_repo_id']}
"""
    atomic_write(output / "README.md", card.encode("utf-8"))
    public_hub = Hub(config["hf_repo_id"])
    public_hub.create_owned(SOURCE_REPOSITORY, private=False)
    files = {path.name: path for path in output.iterdir() if path.is_file()}
    publication_revision = public_hub.commit(files, f"Publish learned WholeBody ONNX at common step {checkpoint['step']}")
    # This is the requested deliverable download, not an artificial failure test.
    anonymous = Hub(config["hf_repo_id"], token=False)
    downloaded = anonymous.read("wholebody.onnx", revision=publication_revision)
    expected_hash = sha256((output / "wholebody.onnx").read_bytes())
    if sha256(downloaded) != expected_hash:
        raise RuntimeError("Public ONNX bytes differ from the exported model")
    result = {
        "model_url": f"https://huggingface.co/{config['hf_repo_id']}",
        "download_url": f"https://huggingface.co/{config['hf_repo_id']}/resolve/main/wholebody.onnx",
        "revision": publication_revision, "onnx_sha256": expected_hash,
        "anonymous_download_succeeded": True, "common_step": checkpoint["step"],
        "target_reached": training_report["target_reached"], "utc": utc_now(),
    }
    atomic_write(work / "status.json", json_bytes(result))
    atomic_write(output / "publication.json", json_bytes(result))
    public_hub.commit({"publication.json": json_bytes(result)}, "Record successful public ONNX download")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            summary.write(f"## Learned WholeBody model\n\n[Download ONNX]({result['download_url']})\n\nCommon step: {checkpoint['step']}; overlapping updates: {checkpoint['parallel_overlap_steps']}; actual resume records: {len(checkpoint['resume_history'])}.\n")
    log(work, "publish", {"event": "published", **result})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bootstrap", "worker", "publish"))
    parser.add_argument("--config", type=Path, default=Path("train_config.json"))
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1)
    parser.add_argument("--rank", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    try:
        if args.command == "bootstrap":
            bootstrap(config, args.work_dir)
        elif args.command == "worker":
            worker(config, args.work_dir, args.phase, args.rank)
        else:
            publish(config, args.work_dir)
    except Exception as error:
        failure = {"event": "failure", "command": args.command, "phase": args.phase,
                   "worker_id": args.rank, "error_type": type(error).__name__, "utc": utc_now()}
        atomic_write(args.work_dir / "status.json", json_bytes(failure))
        log(args.work_dir, f"{args.command}-{args.phase}-{args.rank}", failure)
        if args.command == "worker":
            try:
                Hub(config["checkpoint_repo_id"]).commit({
                    f"runs/{run_key()}/phase{args.phase}/failures/{args.rank}.json": json_bytes(failure),
                }, f"Record stopped worker {args.rank}; retain last complete checkpoint")
            except Exception:
                traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
