"""Real multi-runner gradient training, checkpoint continuation, and publication."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import signal
import tarfile
import time
import traceback
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from data import (CocoCalibrationDataset, CocoWholeBodyDataset, LEGACY_PREPARED_FILES,
                  PREPARED_FILES, Unified3DDataset, ensure_images, prepare_data)
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


def earlier_attempts() -> list[str]:
    run_id, attempt = run_key().rsplit("-", 1)
    return [f"{run_id}-{number}" for number in range(int(attempt) - 1, 0, -1)]


def load_run_manifest(hub: Hub, config: dict[str, Any], *, persist: bool) -> dict[str, Any]:
    manifest = hub.read_json(f"runs/{run_key()}/bootstrap.json", optional=True)
    if manifest is not None:
        return manifest
    # GitHub's 'Re-run failed jobs' retains a successful bootstrap from the
    # earlier attempt. Only that same Actions run and exact source commit may
    # supply its preparation; unrelated runs/repositories are never adopted.
    for previous_key in earlier_attempts():
        previous = hub.read_json(f"runs/{previous_key}/bootstrap.json", optional=True)
        if previous is None:
            continue
        if (previous["source_sha"] != os.environ.get("GITHUB_SHA")
                or previous["training_identity"] != training_identity(config)):
            raise RuntimeError("The successful bootstrap belongs to different code or training configuration")
        manifest = {**previous, "run_key": run_key(), "bootstrap_reused_from": previous_key}
        if persist:
            hub.commit({f"runs/{run_key()}/bootstrap.json": json_bytes(manifest)},
                       "Reuse the same run's successful preparation during failed-job recovery")
        return manifest
    raise RuntimeError("No successful preparation exists for this Actions run; run the full workflow")


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["workers"] != 2:
        raise ValueError("The workflow matrix and configuration must both have two workers")
    if value["hf_repo_id"] == value["checkpoint_repo_id"]:
        raise ValueError("Public inference files and private training state use separate repositories")
    return value


def training_identity(config: dict[str, Any]) -> str:
    keys = ("workers", "seed", "batch2d", "accumulate", "loss3d_weight",
            "pose_learning_rate", "lifter_learning_rate", "weight_decay", "clip_grad_norm",
            "model", "train_limit", "val_limit", "max_3d_frames", "frame_stride", "joint_schema",
            "lifter_warmup_epochs", "lifter_warmup_batch")
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
        [{"params": list(model.pose.parameters()), "lr": config["pose_learning_rate"], "name": "pose"},
         {"params": list(model.lifter.parameters()), "lr": config["lifter_learning_rate"], "name": "lifter"}],
        weight_decay=config["weight_decay"],
    )


def set_learning_rates(optimizer: torch.optim.AdamW, config: dict[str, Any], step: int, target: int) -> None:
    warmup = max(1, min(100, target // 20))
    fraction = min(1.0, step / max(1, target))
    multiplier = min(1.0, (step + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * fraction)))
    for group in optimizer.param_groups:
        group["lr"] = config[f"{group['name']}_learning_rate"] * multiplier


def prepared_bundle(data_root: Path) -> bytes:
    memory = io.BytesIO()
    with tarfile.open(fileobj=memory, mode="w:gz") as archive:
        for filename in PREPARED_FILES:
            archive.add(data_root / filename, arcname=filename, recursive=False)
    return memory.getvalue()


def unpack_prepared(data: bytes, data_root: Path, *, legacy: bool = False) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        expected = LEGACY_PREPARED_FILES if legacy else PREPARED_FILES
        if {entry.name for entry in members} != set(expected):
            raise ValueError("Unexpected prepared-data archive members")
        for entry in members:
            if not entry.isfile():
                raise ValueError("Prepared-data members must be regular files")
            source = archive.extractfile(entry)
            if source is None:
                raise ValueError("Missing prepared-data member")
            atomic_write(data_root / entry.name, source.read())


def learning_plan(config: dict[str, Any], data_root: Path, previous: dict[str, Any] | None = None) -> dict[str, Any]:
    dataset2d = CocoWholeBodyDataset(data_root, "train")
    dataset3d = Unified3DDataset(data_root, "train")
    global_batch2d = config["workers"] * config["accumulate"] * config["batch2d"]
    target = max(2, math.ceil(len(dataset2d) * config["epochs2d"] / global_batch2d))
    # Equal-sized rank batches traverse disjoint complete shards. Use the larger
    # shard to guarantee the requested number of passes for every 3D example.
    batch3d = math.ceil(math.ceil(len(dataset3d) / config["workers"]) * config["epochs3d"]
                        / (target * config["accumulate"]))
    if previous is not None:
        if (previous["samples2d"] != len(dataset2d) or previous["samples3d"] != len(dataset3d)
                or previous["source_counts3d"] != dataset3d.source_counts
                or previous["global_batch2d"] != global_batch2d):
            raise RuntimeError("Cannot change datasets or sample cursors while extending training")
        if (config["epochs2d"] < previous["requested_epochs2d"]
                or config["epochs3d"] < previous["requested_epochs3d"]):
            raise RuntimeError("A continuation may increase the learning budget but cannot reduce it")
        # Preserve every cursor and the per-update task weighting. Increasing
        # the requested epochs adds updates, never changes historical sampling.
        batch3d = previous["batch3d"]
        target = max(target, previous["target_step"],
                     math.ceil(math.ceil(len(dataset3d) / config["workers"]) * config["epochs3d"]
                               / (config["accumulate"] * batch3d)))
    return {"target_step": target, "batch3d": max(1, batch3d),
            "samples2d": len(dataset2d), "samples3d": len(dataset3d),
            "source_counts3d": dataset3d.source_counts, "global_batch2d": global_batch2d,
            "global_batch3d": config["workers"] * config["accumulate"] * max(1, batch3d),
            "requested_epochs2d": config["epochs2d"], "requested_epochs3d": config["epochs3d"],
            "phase_steps": max(1, math.ceil(target / config["planned_phases"])),
            "sampling": "Continuous deterministic disjoint rank cursors; final batch may wrap into the next epoch."}


def calibrate(model: WholeBodyModel, data_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    dataset = CocoCalibrationDataset(data_root, max_samples=config["calibration_samples"], seed=config["seed"])
    batches = DataLoader(dataset, batch_size=config["batch2d"], shuffle=False, num_workers=0)
    measured = model.calibrate_residuals(batches, max_samples=config["calibration_samples"])
    return {**dataset.calibration_manifest, "retained_residuals": measured}


def publish_progress(config: dict[str, Any], work: Path, value: dict[str, Any]) -> None:
    """A small public progress record; never a release or an accuracy claim."""
    status = {"utc": utc_now(), "source_repository": SOURCE_REPOSITORY,
              "actions_run_url": f"https://github.com/{SOURCE_REPOSITORY}/actions/runs/{os.environ.get('GITHUB_RUN_ID')}",
              "run_key": run_key(), "published_model": False, **value}
    try:
        Hub(config["hf_repo_id"]).commit({"training_status.json": json_bytes(status)},
                                        f"Record real learning progress: {value['stage']}")
    except Exception as error:
        # The durable optimizer checkpoint, not this public display, controls
        # recovery. A status transport problem must not discard learned work.
        log(work, "status", {"event": "progress_display_unavailable", "error_type": type(error).__name__})


def warmup_lifter(model: WholeBodyModel, optimizer: torch.optim.AdamW, checkpoint: dict[str, Any],
                  config: dict[str, Any], data_root: Path, hub: Hub, work: Path) -> None:
    """Real small-batch 3D learning before parallel image/3D fine-tuning.

    This is one bootstrap job, explicitly reported separately from distributed
    updates. Its optimizer moments are retained by the later common optimizer.
    """
    dataset = Unified3DDataset(data_root, "train")
    batch_size = config["lifter_warmup_batch"]
    target = math.ceil(len(dataset) * config["lifter_warmup_epochs"] / batch_size)
    progress = checkpoint.setdefault("lifter_warmup", {
        "step": 0, "target_step": target, "batch_size": batch_size,
        "source_seen": {source: 0 for source in dataset.source_counts}, "labels3d": 0,
        "completed": False, "parallel": False,
    })
    stream = ShardStream(dataset, 0, 1, config["seed"] + 7000001)
    if progress["target_step"] != target:
        raise RuntimeError("Warmup data/budget changed while resuming")
    model.train()
    for step in range(progress["step"], target):
        torch.manual_seed(config["seed"] + 3000001 + step)
        model.set_training_progress(min(1.0, (step + 1) / max(1, target // 3)))
        set_learning_rates(optimizer, config, step, target)
        batch = stream.batch(step * batch_size, batch_size)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = model.loss_3d(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite supervised 3D warmup loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.lifter.parameters(), config["clip_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        progress["step"] = step + 1
        progress["labels3d"] += int(metrics["labeled3d"])
        for source, count in Counter(batch["source"]).items():
            progress["source_seen"][source] += count
        progress["completed"] = progress["step"] >= target
        if (step + 1) % 50 == 0 or step + 1 == target:
            log(work, "bootstrap", {"event": "lifter_warmup", "step": step + 1,
                                   "target_step": target, "metrics": {name: float(value) for name, value in metrics.items()},
                                   "source_seen": progress["source_seen"]})
        if (step + 1) % 250 == 0 or progress["completed"] or STOP_REQUESTED:
            checkpoint["model"] = model.state_dict()
            checkpoint["optimizer"] = optimizer.state_dict()
            checkpoint["model_config"] = model.config
            hub.commit({"state/latest.pt": tensor_bytes(checkpoint),
                        "state/warmup.json": json_bytes(progress)},
                       f"Save actual 3D warmup update {step + 1}/{target}")
            publish_progress(config, work, {"stage": "3d_warmup", "step": step + 1,
                                            "target_step": target, "parallel": False,
                                            "source_seen": progress["source_seen"],
                                            "training_batch_metrics": {name: float(value) for name, value in metrics.items()}})
        if STOP_REQUESTED:
            raise RuntimeError("Warmup interrupted after saving its optimizer checkpoint")


def bootstrap(config: dict[str, Any], work: Path) -> None:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is missing. Add the Hugging Face write token as this repository's Actions secret HF_TOKEN.")
    checkpoint_hub = Hub(config["checkpoint_repo_id"])
    publication_hub = Hub(config["hf_repo_id"])
    checkpoint_hub.create_owned(SOURCE_REPOSITORY, private=True)
    publication_hub.create_owned(SOURCE_REPOSITORY, private=False)
    if publication_hub.read("wholebody.onnx", optional=True) is None:
        notice = ("---\nlicense: other\nlicense_name: wholebody-research\nlicense_link: MODEL_LICENSE.md\n---\n"
                  "# Pose73 learning in progress\n\n"
                  "The actual training workflow is preparing and learning this model. An ONNX release is uploaded only after the declared part-wise accuracy and inference-cost criteria pass.\n\n"
                  "[Actual progress](training_status.json) · [Source and Actions](https://github.com/DaisukeDaisuke/Wholebody-onnx)\n\n"
                  f"Model distribution: https://huggingface.co/{config['hf_repo_id']}\n\n"
                  "[Model license](MODEL_LICENSE.md) · [Credits and external distribution sources](THIRD_PARTY.md)\n")
        initial_public_files = {"README.md": notice.encode("utf-8"),
                                "MODEL_LICENSE.md": ROOT / "MODEL_LICENSE.md", "THIRD_PARTY.md": ROOT / "THIRD_PARTY.md"}
        initial_public_files.update({path.relative_to(ROOT).as_posix(): path
                                     for path in (ROOT / "LICENSES").rglob("*") if path.is_file()})
        publication_hub.commit(initial_public_files, "Describe the new training run, distribution and licenses")
    publish_progress(config, work, {"stage": "preparing_real_data"})
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
        source_bytes = source_hub.read("data/prepared.tar.gz", revision=source_revision)
        if sha256(source_bytes) != source["sha256"]:
            raise RuntimeError("Reused prepared-data checksum mismatch")
        unpack_prepared(source_bytes, data_root, legacy=True)
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
    previous_checkpoint = load_tensors(previous_bytes) if previous_bytes is not None else None
    plan = learning_plan(config, data_root, previous_checkpoint["plan"] if previous_checkpoint else None)
    checkpoint_hub.commit({"data/prepared.tar.gz": prepared,
                           "data/dataset_manifest.json": data_root / "dataset_manifest.json"},
                          "Save actual COCO, ASPset and H3WB training data")
    if previous_bytes is None:
        torch.manual_seed(config["seed"])
        model = WholeBodyModel(config["model"], pretrained=True)
        calibration = calibrate(model, data_root, config)
        optimizer = optimizer_for(model, config)
        checkpoint = {
            "format": 2, "source_repository": SOURCE_REPOSITORY,
            "model": model.state_dict(), "model_config": model.config,
            "optimizer": optimizer.state_dict(), "step": 0,
            "training_identity": training_identity(config), "data_sha256": data_digest,
            "seed": config["seed"], "created_at": utc_now(),
            "contribution_counts": {str(rank): 0 for rank in range(config["workers"])},
            "labels2d": 0, "labels3d": 0, "parallel_overlap_steps": 0,
            "history": [], "resume_history": [], "plan": plan,
            "samples_seen2d": 0,
            "source_seen3d": {source: 0 for source in plan["source_counts3d"]},
            "calibration": calibration,
        }
        previous_bytes = tensor_bytes(checkpoint)
        checkpoint_hub.commit({"state/initial.pt": previous_bytes,
                               "state/latest.pt": previous_bytes,
                               "data/residual_calibration.json": json_bytes(calibration)},
                              "Save pose-pretrained compact73 initialization and training-only residual calibration")
    else:
        checkpoint = previous_checkpoint
        if checkpoint["training_identity"] != training_identity(config):
            raise RuntimeError("Training/model configuration changed; cannot silently resume incompatible state")
        if checkpoint["data_sha256"] != data_digest:
            raise RuntimeError("Prepared training data differs from the checkpoint")
        if checkpoint["plan"] != plan:
            checkpoint.setdefault("budget_extensions", []).append({
                "utc": utc_now(), "run_key": run_key(), "restored_step": checkpoint["step"],
                "previous_plan": checkpoint["plan"], "extended_plan": plan,
                "preserved": "Model, optimizer moments, exact per-task batch sizes and deterministic sample cursors",
            })
            checkpoint["plan"] = plan
        model = WholeBodyModel(checkpoint["model_config"], pretrained=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer = optimizer_for(model, config)
        optimizer.load_state_dict(checkpoint["optimizer"])
    warmup_lifter(model, optimizer, checkpoint, config, data_root, checkpoint_hub, work)
    previous_bytes = tensor_bytes(checkpoint)
    manifest = {
        "run_key": run_key(), "source_repository": SOURCE_REPOSITORY,
        "source_sha": os.environ.get("GITHUB_SHA"), "initial_step": checkpoint["step"],
        "target_step": plan["target_step"], "learning_plan": plan,
        "lifter_warmup": checkpoint["lifter_warmup"],
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
    model.set_training_progress(step / max(1, config["target_step"]))
    model.zero_grad(set_to_none=True)
    torch.manual_seed(config["seed"] + step * config["workers"] + rank)
    remaining = scheduled_start - time.time()
    if remaining > 0:
        time.sleep(min(remaining, 20))
    totals: dict[str, float] = {}
    source_seen3d: Counter[str] = Counter()
    started = time.time()
    for microstep in range(config["accumulate"]):
        offset2d = (step * config["accumulate"] + microstep) * config["batch2d"]
        offset3d = (step * config["accumulate"] + microstep) * config["batch3d"]
        batch2d = stream2d.batch(offset2d, config["batch2d"])
        batch3d = stream3d.batch(offset3d, config["batch3d"])
        source_seen3d.update(batch3d["source"])
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
        "source_seen3d": dict(source_seen3d),
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
    set_learning_rates(optimizer, config, checkpoint["step"], checkpoint["plan"]["target_step"])
    optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in parameters.values()):
        raise FloatingPointError("Optimizer produced non-finite model parameters")
    overlap = max(0.0, min(packet["compute_finished"] for packet in packets) - max(packet["compute_started"] for packet in packets))
    evidence = {
        "step": checkpoint["step"] + 1, "phase": phase, "run_key": run_key(),
        "gradient_norm_before_clip": float(norm), "compute_overlap_seconds": overlap,
        "base_snapshot_sha256": snapshot_hash,
        "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
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
        checkpoint["samples_seen2d"] += packet["samples2d"]
        for source, count in packet["source_seen3d"].items():
            checkpoint["source_seen3d"][source] += count
    evidence["effective_epochs2d"] = checkpoint["samples_seen2d"] / checkpoint["plan"]["samples2d"]
    evidence["effective_epochs3d"] = {
        source: count / checkpoint["plan"]["source_counts3d"][source]
        for source, count in checkpoint["source_seen3d"].items()
    }
    return evidence


def worker(config: dict[str, Any], work: Path, phase: int, rank: int) -> None:
    hub = Hub(config["checkpoint_repo_id"])
    prefix = f"runs/{run_key()}/phase{phase}"
    deadline = time.monotonic() + 60 * config["worker_minutes"]
    wait_seconds = 60 * config["worker_wait_minutes"]
    bootstrap_manifest = load_run_manifest(hub, config, persist=rank == 0)
    config = {**config, "batch3d": bootstrap_manifest["learning_plan"]["batch3d"],
              "target_step": bootstrap_manifest["target_step"]}
    data_root = work / "data"
    for previous_key in earlier_attempts():
        finished = hub.read_json(f"runs/{previous_key}/phase{phase}/finished.json", optional=True)
        if finished is not None:
            current = load_tensors(hub.read("state/latest.pt"))
            if current["step"] < finished["step"]:
                raise RuntimeError("Durable state precedes a previously completed phase")
            write_worker_status(work, current["step"], bootstrap_manifest["target_step"])
            return
    model = optimizer = None
    if rank == 0:
        revision = hub.head()
        checkpoint_data = hub.read("state/latest.pt", revision=revision)
        checkpoint = load_tensors(checkpoint_data)
        if not checkpoint["lifter_warmup"]["completed"]:
            raise RuntimeError("The real 3D warmup budget is incomplete")
        if checkpoint["step"] < bootstrap_manifest["target_step"]:
            data_bytes = hub.read("data/prepared.tar.gz", revision=revision)
            if sha256(data_bytes) != checkpoint["data_sha256"]:
                raise ValueError("Worker data differs from the common checkpoint")
            unpack_prepared(data_bytes, data_root)
            model = WholeBodyModel(checkpoint["model_config"], pretrained=False)
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer = optimizer_for(model, config)
            optimizer.load_state_dict(checkpoint["optimizer"])
            if checkpoint["step"] > 0:
                if not optimizer.state:
                    raise RuntimeError("A trained checkpoint is missing its optimizer state")
                checkpoint["resume_history"].append({
                    "run_key": run_key(), "phase": phase, "restored_step": checkpoint["step"],
                    "optimizer_state_entries": len(optimizer.state), "utc": utc_now(),
                })
                checkpoint["calibration"] = calibrate(model, data_root, config)
                checkpoint["model"] = model.state_dict()
            revision = hub.commit({"state/previous-phase.pt": checkpoint_data,
                                   "state/latest.pt": tensor_bytes(checkpoint),
                                   "data/residual_calibration.json": json_bytes(checkpoint["calibration"])},
                                  f"Restore phase {phase} optimizer and share current training-data residuals")
        plan = {
            "checkpoint_revision": revision, "start_step": checkpoint["step"],
            "target_step": min(bootstrap_manifest["target_step"], checkpoint["step"] + bootstrap_manifest["learning_plan"]["phase_steps"]),
            "final_target_step": bootstrap_manifest["target_step"],
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
    if plan["start_step"] >= plan["final_target_step"]:
        write_worker_status(work, plan["start_step"], plan["final_target_step"])
        return
    if rank != 0:
        data_bytes = hub.read("data/prepared.tar.gz", revision=plan["checkpoint_revision"])
        if sha256(data_bytes) != checkpoint["data_sha256"]:
            raise ValueError("Worker data differs from the common checkpoint")
        unpack_prepared(data_bytes, data_root)
        model = WholeBodyModel(checkpoint["model_config"], pretrained=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer = optimizer_for(model, config)
        optimizer.load_state_dict(checkpoint["optimizer"])
    ensure_images(data_root, split="train", rank=rank, world_size=config["workers"], workers=8)
    # Teacher belongs to training only and is never a registered student module.
    model.get_teacher()
    stream2d = ShardStream(CocoWholeBodyDataset(data_root, "train"), rank, config["workers"], config["seed"])
    stream3d = ShardStream(Unified3DDataset(data_root, "train"), rank, config["workers"], config["seed"] + 1000003)
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
    reason = "phase_budget_reached"
    last_round_seconds = 0.0
    while step < plan["target_step"]:
        round_started = time.monotonic()
        if rank == 0:
            if STOP_REQUESTED or time.monotonic() > deadline - max(600, last_round_seconds * 1.5):
                reason = "time_budget"
                break
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
                    acknowledge_phase(hub, prefix, rank, finished, work, plan["final_target_step"])
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
            hub.commit({"state/latest.pt": latest,
                        f"state/updates/{checkpoint['step']:08d}.json": json_bytes(evidence),
                        "state/progress.json": json_bytes({
                "step": checkpoint["step"], "run_key": run_key(), "phase": phase,
                "checkpoint_sha256": sha256(latest), "last_update": evidence,
            })}, f"Apply both worker gradients: common AdamW step {checkpoint['step']}")
            log(work, f"worker-{phase}-{rank}", {"event": "common_update", **evidence})
            if checkpoint["step"] % 10 == 0 or checkpoint["step"] >= plan["target_step"]:
                publish_progress(config, work, {"stage": "parallel_finetuning", "phase": phase,
                                                "step": checkpoint["step"], "target_step": plan["final_target_step"],
                                                "effective_epochs2d": evidence["effective_epochs2d"],
                                                "effective_epochs3d": evidence["effective_epochs3d"],
                                                "contribution_counts": checkpoint["contribution_counts"],
                                                "parallel_overlap_steps": checkpoint["parallel_overlap_steps"],
                                                "training_batch_metrics": [packet["metrics"] for packet in packets]})
        step += 1
        last_round_seconds = time.monotonic() - round_started
    if rank == 0:
        finished = {"step": step, "target_step": plan["target_step"],
                    "final_target_step": plan["final_target_step"], "reason": reason}
        hub.commit({f"{prefix}/finished.json": json_bytes(finished)}, f"Complete phase {phase} at saved step {step}")
        wait_json(hub, f"{prefix}/ack/1.json", time.monotonic() + min(wait_seconds, 300),
                  abort_path=f"{prefix}/failures/1.json")
        # Both jobs have finished all reads of round snapshots. Only temporary
        # exchanges from this newly-created training project are compacted.
        hub.compact_transport(prefix)
        write_worker_status(work, step, plan["final_target_step"])
    else:
        finished = wait_json(hub, f"{prefix}/finished.json", min(deadline, time.monotonic() + wait_seconds), abort_path=f"{prefix}/failures/0.json")
        acknowledge_phase(hub, prefix, rank, finished, work, plan["final_target_step"])


def write_worker_status(work: Path, step: int, target: int) -> None:
    result = {"step": step, "target_step": target, "remaining": step < target}
    atomic_write(work / "status.json", json_bytes(result))
    if os.environ.get("GITHUB_OUTPUT"):
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
            stream.write(f"remaining={str(step < target).lower()}\n")


def acknowledge_phase(hub: Hub, prefix: str, rank: int, finished: dict[str, Any], work: Path, target: int) -> None:
    hub.commit({f"{prefix}/ack/{rank}.json": json_bytes({"step": finished["step"], "worker_id": rank})},
               f"Worker {rank} released completed phase snapshots")
    write_worker_status(work, finished["step"], target)


def publish(config: dict[str, Any], work: Path) -> None:
    from publication import publish_learned
    publish_learned(config, work)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bootstrap", "worker", "publish"))
    parser.add_argument("--config", type=Path, default=Path("train_config.json"))
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--phase", type=int, choices=range(1, 13), default=1)
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
