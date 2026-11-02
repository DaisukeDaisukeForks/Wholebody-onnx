"""Publish only a fully trained checkpoint that meets the declared quality gate."""

from __future__ import annotations

import json
import os
from pathlib import Path

from data import ensure_images
from export_model import export
from hub import Hub, atomic_write, json_bytes, load_tensors, sha256
from joint_schema import COMPACT_COCO_IDS, COMPACT_JOINT_NAMES


def publish_learned(config: dict, work: Path) -> None:
    from train import ROOT, SOURCE_REPOSITORY, load_run_manifest, log, publish_progress, run_key, training_identity, unpack_prepared, utc_now

    checkpoint_hub = Hub(config["checkpoint_repo_id"])
    revision = checkpoint_hub.head()
    checkpoint_data = checkpoint_hub.read("state/latest.pt", revision=revision)
    checkpoint = load_tensors(checkpoint_data)
    manifest = load_run_manifest(checkpoint_hub, config, persist=True)
    if checkpoint["training_identity"] != training_identity(config):
        raise RuntimeError("Publication configuration differs from the learned checkpoint")
    if checkpoint["step"] < manifest["target_step"]:
        result = {"status": "training_budget_incomplete", "step": checkpoint["step"],
                  "target_step": manifest["target_step"], "published": False,
                  "checkpoint_repository": config["checkpoint_repo_id"]}
        atomic_write(work / "status.json", json_bytes(result))
        checkpoint_hub.commit({f"runs/{run_key()}/publication_status.json": json_bytes(result)},
                              "Retain incomplete training state for the next Actions continuation")
        raise RuntimeError("The epoch-derived training budget is incomplete; resume the durable optimizer state")
    if not checkpoint["lifter_warmup"]["completed"] or checkpoint["labels2d"] <= 0 or checkpoint["labels3d"] <= 0:
        raise RuntimeError("The supervised learning stages have not completed")
    if any(checkpoint["contribution_counts"].get(str(rank), 0) != checkpoint["step"] for rank in range(config["workers"])):
        raise RuntimeError("Every shared update must contain every worker's actual gradient contribution")
    if checkpoint["parallel_overlap_steps"] <= 0 or not checkpoint["resume_history"]:
        raise RuntimeError("Actual overlapping computation and checkpoint continuation have not both been recorded")

    data_root = work / "data"
    prepared = checkpoint_hub.read("data/prepared.tar.gz", revision=revision)
    if sha256(prepared) != checkpoint["data_sha256"]:
        raise ValueError("Publication data differs from training data")
    unpack_prepared(prepared, data_root)
    atomic_write(work / "latest.pt", checkpoint_data)
    atomic_write(work / "initial.pt", checkpoint_hub.read("state/initial.pt", revision=revision))
    ensure_images(data_root, split="val", workers=8)
    output = work / "release"
    publish_progress(config, work, {"stage": "held_out_accuracy_evaluation", "step": checkpoint["step"],
                                    "target_step": manifest["target_step"]})
    metrics = export(work / "latest.pt", data_root, output,
                     quality_gate=config["quality_gate"], baseline_path=work / "initial.pt",
                     validation_samples=config["validation_samples"])
    resources = json.loads((output / "resource_usage.json").read_text(encoding="utf-8"))
    inference_checks = {
        "macs_per_crop": {"value": resources["counted_macs_per_sample"],
                          "maximum": config["inference_budget"]["max_macs_per_crop"]},
        "parameters": {"value": metrics["parameters"],
                       "maximum": config["inference_budget"]["max_parameters"]},
    }
    for check in inference_checks.values():
        check["passed"] = check["value"] <= check["maximum"]
    metrics["inference_budget"] = {"passed": all(check["passed"] for check in inference_checks.values()),
                                   "checks": inference_checks,
                                   "scope": "Arithmetic and parameter bounds, not a measurement of device GPU utilization or latency."}
    atomic_write(output / "metrics.json", json_bytes(metrics))
    source_epochs = {
        source: count / checkpoint["plan"]["source_counts3d"][source]
        for source, count in checkpoint["source_seen3d"].items()
    }
    report = {
        "algorithm": "Synchronous equal-worker mean gradients and one central AdamW update",
        "source_repository": SOURCE_REPOSITORY, "source_sha": os.environ.get("GITHUB_SHA"),
        "actions_run_url": f"https://github.com/{SOURCE_REPOSITORY}/actions/runs/{os.environ.get('GITHUB_RUN_ID')}",
        "run_key": run_key(), "initial_step": manifest["initial_step"], "step": checkpoint["step"],
        "target_step": manifest["target_step"], "target_reached": True,
        "learning_plan": checkpoint["plan"], "lifter_warmup": checkpoint["lifter_warmup"],
        "effective_epochs2d": checkpoint["samples_seen2d"] / checkpoint["plan"]["samples2d"],
        "parallel_effective_epochs3d": source_epochs,
        "source_seen3d": checkpoint["source_seen3d"],
        "contribution_counts": checkpoint["contribution_counts"],
        "parallel_overlap_steps": checkpoint["parallel_overlap_steps"],
        "joint_schema": config["joint_schema"], "learned_2d_points": 73, "learned_3d_points": 73,
        "joint_names": list(COMPACT_JOINT_NAMES),
        "labels2d": checkpoint["labels2d"], "labels3d": checkpoint["labels3d"],
        "resume_history": checkpoint["resume_history"], "updates": checkpoint["history"],
        "budget_extensions": checkpoint.get("budget_extensions", []),
        "residual_calibration": checkpoint["calibration"],
        "checkpoint_repository": config["checkpoint_repo_id"], "checkpoint_revision": revision,
        "checkpoint_sha256": sha256(checkpoint_data), "data_sha256": checkpoint["data_sha256"],
        "resources": "Two standard ubuntu-24.04 CPU runners per parallel phase; one CPU job for 3D warmup",
        "quality_gate": metrics["quality_gate"],
        "inference_budget": metrics["inference_budget"],
        "limitations": "Whole-body 3D uses annotation/noise evaluation; actual held-out RGB-to-3D evaluation covers body12 on one ASP clip. GPU latency/utilization are not measured.",
    }
    atomic_write(output / "training_report.json", json_bytes(report))
    published_data = json.loads((data_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    published_data["model_supervision"] = {
        "schema": config["joint_schema"], "source_annotation_points": 133,
        "learned_2d_points": 73, "learned_3d_points": 73,
        "selected_coco_ids": list(COMPACT_COCO_IDS), "joint_names": list(COMPACT_JOINT_NAMES),
        "3d_sources": {"aspset": "Actual body12 only; all other target masks zero",
                       "h3wb": "Actual compact73 multiview reconstructed/completed annotations"},
    }
    atomic_write(output / "dataset_manifest.json", json_bytes(published_data))
    for name in ("MODEL_LICENSE.md", "THIRD_PARTY.md", "data_sources.json", "train_config.json"):
        atomic_write(output / name, (ROOT / name).read_bytes())
    for source in (ROOT / "LICENSES").rglob("*"):
        if source.is_file():
            atomic_write(output / source.relative_to(ROOT), source.read_bytes())
    atomic_write(output / "CODE_LICENSE", (ROOT / "LICENSE").read_bytes())
    atomic_write(output / "LICENSE", (ROOT / "MODEL_LICENSE.md").read_bytes())

    if not metrics["quality_gate"]["passed"] or not metrics["inference_budget"]["passed"]:
        result = {"status": "quality_gate_not_met", "published": False,
                  "step": checkpoint["step"], "quality_gate": metrics["quality_gate"],
                  "inference_budget": metrics["inference_budget"],
                  "checkpoint_repository": config["checkpoint_repo_id"]}
        atomic_write(work / "status.json", json_bytes(result))
        # Retain the real candidate and measured errors privately for further
        # learning. Never mark a conversion-only success as a quality release.
        private_files = {f"candidates/{run_key()}/{path.relative_to(output).as_posix()}": path
                         for path in output.rglob("*") if path.is_file()}
        private_files[f"runs/{run_key()}/publication_status.json"] = json_bytes(result)
        checkpoint_hub.commit(private_files, "Retain evaluated candidate; declared quality thresholds are unmet")
        publish_progress(config, work, {"stage": "quality_criteria_unmet", **result})
        log(work, "publish", result)
        raise RuntimeError("Declared part-wise quality criteria are unmet; trained checkpoint and candidate were retained privately")

    card = f"""---
license: other
license_name: wholebody-research
license_link: https://huggingface.co/{config['hf_repo_id']}/blob/main/MODEL_LICENSE.md
library_name: onnx
pipeline_tag: keypoint-detection
tags:
- wholebody
- pose-estimation
- onnx
- compact73
---
# Pose73 WholeBody: 73-point 2D and relative 3D

This model was learned in [GitHub Actions]({report['actions_run_url']}).
It contains {checkpoint['step']} shared AdamW updates with both parallel workers contributing to every update.
Their actual computation intervals overlap in {checkpoint['parallel_overlap_steps']} updates.
2D training processed {checkpoint['samples_seen2d']} person crops, or {report['effective_epochs2d']:.3f} complete-dataset equivalents.
Small-batch 3D warmup and subsequent parallel 3D training are counted separately in [training_report.json](training_report.json).

## Distribution

- [Download wholebody.onnx](https://huggingface.co/{config['hf_repo_id']}/resolve/main/wholebody.onnx)
- [Source code](https://github.com/{SOURCE_REPOSITORY})
- [Joint order, coordinates and preprocessing](schema.json)
- [Actual part-wise errors and publication decisions](metrics.json)
- [Model arithmetic and memory sizes](resource_usage.json)

## Model and inputs

One pose-pretrained DWPose-t backbone is shared by the selected73-point head and a small coordinate lifter.
The full133-point frozen teacher is used only during training and evaluation; it is absent from the ONNX graph.
Input image is float32 [batch,3,256,192], RGB in [0,1], with one person crop per batch entry.
The crop preserves width/height0.75 and expands the original bounding box by1.25. Normalization is inside the graph.
Multiple person/camera crops can share the dynamic batch dimension. No detector, image resizing pipeline or camera application is included.

| Output | Shape | Meaning |
|---|---|---|
| keypoints2d | [batch,73,2] | Crop-normalized x-right/y-down |
| confidence2d | [batch,73] | SimCC concentration; not calibrated visibility |
| keypoints3d | [batch,73,3] | Camera x-right/y-down/z-away, meters relative to the hip midpoint |
| geometry2d | [batch,8,2] | Hand references and hand/foot direction vectors |
| geometry_confidence | [batch,8] | Minimum2D source confidence; not3D uncertainty |
| geometry3d | [batch,8,3] | The same references/vectors derived from predicted3D, in meters |

The73 learned locations comprise12body,40finger,6foot and15face points. Body wrists also serve as hand roots.
Geometry vectors are derived from the skeleton, not additional learned landmarks or a surface mesh.
The model does not estimate camera distance, global root translation, gaze, pupils or a dense sole surface.

## Actual initialization and supervision

The pose branch starts from the official DWPose-t WholeBody checkpoint, including its pose backbone and head.
Only the final joint-channel convolution is selected down to73; fine-tuning and teacher distillation adapt the changed cross-joint attention.
All pose weights are trainable while pretrained BatchNorm running statistics stay fixed for small image batches.
COCO-WholeBody supplies the selected2D labels. ASPset supplies measured/calibrated body12 supervision.
H3WB supplies aligned compact73 2D/3D multiview reconstructed/completed annotations, with S1/S5/S6 for training and S7 for validation.
Only available labels contribute to each source's loss. Missing ASP hand/foot/face labels are never fabricated.
The lifter learns from clean and corrupted2D inputs with confidence and missingness. The empirical corruption comes only from the student's predictions on COCO training images.

## Measured quality and scope

All declared publication criteria passed on the recorded held-out evaluation. See each measured value and threshold in [metrics.json](metrics.json).
The criteria separately cover body, both finger groups, feet and face; finger error uses the local palm length.
The final2D model is also compared with the official full teacher's selected joints and the actual initial compact73 model on the same examples.
The3D report separates clean annotation inputs and inputs corrupted by measured training-image prediction errors.
It also measures the actual RGB-to-predicted2D-to-body3D path on a held-out ASPset test clip, with exact source frames and scope recorded in metrics.json.
That body-only clip and the lifting measurements do not establish whole-body RGB-to-3D accuracy or production tracking performance.
Arithmetic/parameter counts are not measured GPU utilization, camera capacity or device-specific frame rate.

## License and attribution

The distribution terms for our new contributions are in [MODEL_LICENSE.md](MODEL_LICENSE.md), for noncommercial research and education.
Upstream DWPose parameters and adapted OpenMMLab/DWPose code retain Apache-2.0; H3WB annotations retain MIT.
COCO-WholeBody's CC BY4.0 notice and separate noncommercial research terms, COCO image terms and ASPset CC0 are recorded separately.
[THIRD_PARTY.md](THIRD_PARTY.md) includes authors, papers, source/distribution links and modifications.
Full upstream notices are in [LICENSES](LICENSES). Original project code retains Unlicense; external code/data/weights are not relicensed as Unlicense.
Model distribution: https://huggingface.co/{config['hf_repo_id']}
Source distribution: https://github.com/{SOURCE_REPOSITORY}
"""
    atomic_write(output / "README.md", card.encode("utf-8"))
    public_hub = Hub(config["hf_repo_id"])
    public_hub.create_owned(SOURCE_REPOSITORY, private=False)
    files = {path.relative_to(output).as_posix(): path for path in output.rglob("*") if path.is_file()}
    publication_revision = public_hub.commit(files, f"Publish learned Pose73 ONNX after declared quality criteria pass at step {checkpoint['step']}")
    anonymous = Hub(config["hf_repo_id"], token=False)
    downloaded = anonymous.read("wholebody.onnx", revision=publication_revision)
    expected_hash = sha256((output / "wholebody.onnx").read_bytes())
    if sha256(downloaded) != expected_hash:
        raise RuntimeError("Public ONNX bytes differ from the exported learned model")
    result = {"model_url": f"https://huggingface.co/{config['hf_repo_id']}",
              "download_url": f"https://huggingface.co/{config['hf_repo_id']}/resolve/main/wholebody.onnx",
              "revision": publication_revision, "onnx_sha256": expected_hash,
              "anonymous_download_succeeded": True, "quality_gate_passed": True,
              "common_step": checkpoint["step"], "target_reached": True, "utc": utc_now()}
    atomic_write(work / "status.json", json_bytes(result))
    public_hub.commit({"publication.json": json_bytes(result),
                       "training_status.json": json_bytes({"stage": "published", "published_model": True, **result})},
                      "Record successful anonymous model download")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as summary:
            summary.write(f"## Learned Pose73 model\n\n[Download ONNX]({result['download_url']})\n\n"
                          f"Shared updates: {checkpoint['step']}; quality criteria: passed; "
                          f"overlapping updates: {checkpoint['parallel_overlap_steps']}; actual resumes: {len(checkpoint['resume_history'])}.\n")
    log(work, "publish", {"event": "published", **result})
