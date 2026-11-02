"""Export the learned compact model and measure its declared held-out quality."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader, Subset

from data import ASPsetRGBEvaluationDataset, CocoWholeBodyDataset, Unified3DDataset
from joint_schema import GROUPS
from model import REGIONS_3D, WholeBodyModel, batch_2d, batch_3d


OUTPUT_NAMES = [
    "keypoints2d", "confidence2d", "keypoints3d", "geometry2d",
    "geometry_confidence", "geometry3d",
]


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _indices(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        raise ValueError("Held-out evaluation requires a nonempty source and a positive sample count")
    return np.linspace(0, length - 1, min(length, count), dtype=np.int64).tolist()


def _2d_counter() -> dict[str, dict[str, float]]:
    return {name: {"error": 0.0, "correct": 0.0, "count": 0,
                   "local_error": 0.0, "local_correct": 0.0, "local_count": 0}
            for name in ("all", *GROUPS)}


def _accumulate_2d(totals: dict[str, dict[str, float]], prediction: torch.Tensor,
                   target: torch.Tensor, visibility: torch.Tensor,
                   crop_threshold: float, hand_threshold: float) -> None:
    valid = ((visibility > 0) & torch.isfinite(target).all(-1)
             & (target >= 0).all(-1) & (target <= 1).all(-1))
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    aspect = target.new_tensor([0.75, 1.0])
    error = ((prediction - target) * aspect).square().sum(-1).sqrt()
    if not torch.isfinite(error).all():
        raise RuntimeError("A learned 2D prediction contains non-finite coordinates")
    for name, ids in {"all": list(range(73)), **GROUPS}.items():
        selected_error, selected_valid = error[:, ids], valid[:, ids]
        total = totals[name]
        total["error"] += float((selected_error * selected_valid).sum())
        total["correct"] += int(((selected_error <= crop_threshold) & selected_valid).sum())
        total["count"] += int(selected_valid.sum())
        if name in ("left_fingers", "right_fingers"):
            wrist, middle_mcp = (4, 20) if name == "left_fingers" else (5, 40)
            palm_length = ((target[:, middle_mcp] - target[:, wrist]) * aspect).square().sum(-1).sqrt()
            local_valid = (selected_valid & valid[:, wrist:wrist+1]
                           & valid[:, middle_mcp:middle_mcp+1] & (palm_length[:, None] > 1e-6))
            local_error = selected_error / palm_length[:, None].clamp_min(1e-6)
            total["local_error"] += float((local_error * local_valid).sum())
            total["local_correct"] += int(((local_error <= hand_threshold) & local_valid).sum())
            total["local_count"] += int(local_valid.sum())


def _finish_2d(totals: dict[str, dict[str, float]]) -> dict[str, Any]:
    result = {}
    for name, value in totals.items():
        count, local_count = int(value["count"]), int(value["local_count"])
        result[name] = {
            "labeled_joint_count": count,
            "normalized_2d_error": value["error"] / count if count else None,
            "pck_crop": value["correct"] / count if count else None,
            "local_labeled_joint_count": local_count,
            "local_normalized_error": value["local_error"] / local_count if local_count else None,
            "pck_local": value["local_correct"] / local_count if local_count else None,
        }
    return result


def evaluate_2d(model: WholeBodyModel, baseline: WholeBodyModel | None,
                data_root: Path, count: int,
                quality_gate: dict[str, float]) -> tuple[dict[str, Any], torch.Tensor]:
    dataset = CocoWholeBodyDataset(data_root, split="val")
    selected = _indices(len(dataset), count)
    totals = {name: _2d_counter() for name in ("student", "teacher", "initial")}
    crop_threshold = float(quality_gate["crop_pck_threshold"])
    hand_threshold = float(quality_gate["hand_local_pck_threshold"])
    example = None
    with torch.inference_mode():
        for batch in DataLoader(Subset(dataset, selected), batch_size=8, shuffle=False):
            rgb = batch["image"]
            if example is None:
                example = rgb[:2].clone()
            target, validity = batch_2d(batch)
            predictions = {
                "student": model.predict_2d(rgb)[0],
                "teacher": model.teacher_predictions(rgb)[0],
            }
            if baseline is not None:
                predictions["initial"] = baseline.predict_2d(rgb)[0]
            for name, prediction in predictions.items():
                _accumulate_2d(totals[name], prediction, target, validity, crop_threshold, hand_threshold)
    if example is None or not totals["student"]["all"]["count"]:
        raise RuntimeError("No real labeled COCO validation landmarks were available")
    final = _finish_2d(totals["student"])
    return {
        "validation2d_samples": len(selected),
        "student_2d": final,
        "teacher_2d": _finish_2d(totals["teacher"]),
        "initial_2d": _finish_2d(totals["initial"]) if baseline is not None else None,
        "group_2d": {name: final[name] for name in GROUPS},
        "normalized_2d_error": final["all"]["normalized_2d_error"],
        "pck_crop": final["all"]["pck_crop"],
        "2d_metric_definitions": {
            "crop_error": "Euclidean x/y distance after x is multiplied by 0.75, in crop-height units",
            "crop_pck_threshold": crop_threshold,
            "hand_local_scale": "Ground-truth wrist-to-middle-MCP distance in the same isotropic crop plane",
            "hand_local_pck_threshold": hand_threshold,
            "labels": "Only annotated, finite landmarks inside the crop; no confidence-based rejection of difficult predictions",
            "baselines": "Final compact73, actual initial compact73, and full frozen133-joint teacher selected to73, on the identical held-out RGB crops",
        },
    }, example


def _3d_counter() -> dict[str, dict[str, float]]:
    return {name: {"error_m": 0.0, "count": 0} for name in REGIONS_3D}


def _accumulate_3d(totals: dict[str, dict[str, float]], prediction: torch.Tensor,
                   target: torch.Tensor, visibility: torch.Tensor) -> None:
    valid = (visibility > 0) & torch.isfinite(target).all(-1)
    safe_target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    if not torch.isfinite(prediction).all():
        raise RuntimeError("A learned 3D prediction contains non-finite coordinates")
    for name, (ids, anchor) in REGIONS_3D.items():
        pred_region, target_region = prediction[:, ids], safe_target[:, ids]
        selected_valid = valid[:, ids]
        if anchor is not None:
            pred_region = pred_region - prediction[:, anchor:anchor+1]
            target_region = target_region - safe_target[:, anchor:anchor+1]
            selected_valid = selected_valid & valid[:, anchor:anchor+1]
        error = (pred_region - target_region).square().sum(-1).sqrt()
        totals[name]["error_m"] += float((error * selected_valid).sum())
        totals[name]["count"] += int(selected_valid.sum())


def _finish_3d(totals: dict[str, dict[str, float]]) -> dict[str, Any]:
    result = {}
    for name, value in totals.items():
        count = int(value["count"])
        anchor = REGIONS_3D[name][1]
        result[name] = {
            "labeled_joint_count": count,
            "mpjpe_mm": 1000.0 * value["error_m"] / count if count else None,
            "reference": "hip midpoint" if anchor is None else f"local joint {anchor}",
        }
    return result


def evaluate_3d(model: WholeBodyModel, data_root: Path, count: int) -> dict[str, Any]:
    dataset = Unified3DDataset(data_root, split="val")
    aggregate = {name: _3d_counter() for name in ("clean", "corrupted")}
    by_source, sample_counts = {}, {}
    generator = torch.Generator().manual_seed(20261102)
    with torch.inference_mode():
        for source_dataset in dataset.datasets:
            source = source_dataset.source
            selected = _indices(len(source_dataset), count)
            source_totals = {name: _3d_counter() for name in ("clean", "corrupted")}
            sample_counts[source] = len(selected)
            for batch in DataLoader(Subset(source_dataset, selected), batch_size=128, shuffle=False):
                xy, validity2d, target, validity3d = batch_3d(batch)
                confidence = validity2d.to(xy.dtype) * torch.isfinite(xy).all(-1)
                xy = torch.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
                noisy_xy, noisy_confidence = model.corrupt_2d(xy, confidence, level=1.0, generator=generator)
                predictions = {
                    "clean": model.lift_body(xy, confidence),
                    "corrupted": model.lift_body(noisy_xy, noisy_confidence),
                }
                for name, prediction in predictions.items():
                    _accumulate_3d(source_totals[name], prediction, target, validity3d)
                    _accumulate_3d(aggregate[name], prediction, target, validity3d)
            by_source[source] = {name: _finish_3d(values) for name, values in source_totals.items()}
    return {
        "validation3d_samples_by_source": sample_counts,
        "3d_clean": _finish_3d(aggregate["clean"]),
        "3d_corrupted": _finish_3d(aggregate["corrupted"]),
        "3d_by_source": by_source,
        "residual_calibration_samples": int(model.residual_count),
        "3d_metric_scope": {
            "clean": "Actual held-out paired2D/3D annotations; input2D is ground truth",
            "corrupted": "The same paired samples with training-RGB empirical residuals, confidence and dropout applied before inference normalization; a robustness surrogate, not RGB inference",
            "coordinates": "Body is hip-relative; hands are wrist-relative; feet are ankle-relative; face is nose-relative, all in millimeters with no Procrustes alignment",
            "sources": "ASPset contributes only body12; H3WB contributes actual annotated compact73, including multiview reconstruction/completion",
            "sampling": "Up to the requested sample count from each source independently; aggregate counts weight each available labeled joint",
        },
    }


def evaluate_rgb_body_3d(model: WholeBodyModel, data_root: Path,
                         max_frames: int = 192) -> dict[str, Any]:
    """Measure the complete RGB->predicted2D/confidence->body3D path."""
    dataset = ASPsetRGBEvaluationDataset(data_root, max_frames=max_frames)
    if not len(dataset):
        raise RuntimeError("Actual held-out ASPset RGB frames are required for the RGB-to-body3D gate")
    error_m, labeled_count, samples = 0.0, 0, 0
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=8, shuffle=False):
            # Do not substitute annotation2D or its visibility for model inputs.
            prediction = model(batch["image"])[2][:, :12]
            target = batch["x3d"][:, :12]
            valid = (batch["valid3d"][:, :12] > 0) & torch.isfinite(target).all(-1)
            if not torch.isfinite(prediction).all():
                raise RuntimeError("RGB-to-body3D produced non-finite coordinates")
            target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
            error = (prediction - target).square().sum(-1).sqrt()
            error_m += float((error * valid).sum())
            labeled_count += int(valid.sum())
            samples += len(prediction)
    metric = 1000.0 * error_m / labeled_count if labeled_count else None
    return {
        "rgb_body_mpjpe_mm": metric,
        "rgb_body_3d": {
            "mpjpe_mm": metric, "samples": samples, "labeled_joint_count": labeled_count,
            "input_path": "Real held-out RGB -> predicted73x2D and predicted confidence -> learned73x3D; only body12 is scored",
            "reference": "Camera-space hip-relative meters converted to mm; no Procrustes alignment",
            "crop": "Official ground-truth person box, aspect0.75 and expansion1.25; person detection is not evaluated",
            "scope": "One held-out ASPset test clip and one camera, with evenly spaced frames. This does not establish RGB-to-hand/foot3D accuracy, unseen-camera robustness or tracking quality.",
            "dataset": dataset.manifest,
        },
    }


def assess_quality(metrics: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, observed: Any, threshold: float, direction: str) -> None:
        finite = isinstance(observed, (int, float)) and math.isfinite(float(observed))
        passed = bool(finite and (observed >= threshold if direction == "min" else observed <= threshold))
        checks.append({"name": name, "observed": observed, "threshold": threshold,
                       "comparison": ">=" if direction == "min" else "<=", "passed": passed})

    final = metrics["student_2d"]
    for group, metric, minimum in (
        ("body", "pck_crop", "body_pck_min"),
        ("left_fingers", "pck_local", "finger_local_pck_min"),
        ("right_fingers", "pck_local", "finger_local_pck_min"),
        ("feet", "pck_crop", "feet_pck_min"),
        ("face", "pck_crop", "face_pck_min"),
    ):
        check(f"2d.{group}.{metric}", final[group][metric], float(thresholds[minimum]), "min")
        for baseline_name in ("teacher_2d", "initial_2d"):
            baseline = metrics.get(baseline_name)
            baseline_value = baseline[group][metric] if baseline else None
            observed = final[group][metric]
            ratio = (observed / baseline_value if observed is not None and baseline_value is not None and baseline_value > 0
                     else None)
            check(f"2d.{group}.relative_to_{baseline_name}", ratio, float(thresholds["relative_2d_min"]), "min")
    for mode in ("clean", "corrupted"):
        values = metrics[f"3d_{mode}"]
        for group, maximum in (
            ("body", "body_mpjpe_mm_max"),
            ("left_hand", "hand_local_mpjpe_mm_max"),
            ("right_hand", "hand_local_mpjpe_mm_max"),
            ("left_foot", "foot_local_mpjpe_mm_max"),
            ("right_foot", "foot_local_mpjpe_mm_max"),
        ):
            check(f"3d.{mode}.{group}.mpjpe_mm", values[group]["mpjpe_mm"], float(thresholds[maximum]), "max")
    check("3d.rgb.body.mpjpe_mm", metrics.get("rgb_body_mpjpe_mm"), float(thresholds["rgb_body_mpjpe_mm_max"]), "max")
    check("residual_calibration_samples", metrics.get("residual_calibration_samples"), 1, "min")
    failures = [value["name"] for value in checks if not value["passed"]]
    return {
        "passed": not failures, "thresholds": thresholds, "checks": checks, "failures": failures,
        "scope": "Declared thresholds apply separately to every named part, both real2D baselines, clean/corrupted lifting, and actual RGB body3D on one held-out clip. Missing metrics fail; thresholds are not adjusted after observing results.",
    }


def inference_with_resource_counts(model: WholeBodyModel, example: torch.Tensor) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Count model shapes and arithmetic during the required real-image inference."""
    records: list[dict[str, Any]] = []
    hooks = []
    batch_size = int(example.shape[0])

    def record(name: str):
        def hook(module, inputs, output):
            values = [output] if isinstance(output, torch.Tensor) else list(output)
            values = [value for value in values if isinstance(value, torch.Tensor)]
            elements = sum(int(value.numel()) for value in values)
            macs = 0
            if isinstance(module, torch.nn.Conv2d):
                macs = int(output.numel()) * (module.in_channels // module.groups) * math.prod(module.kernel_size)
            elif isinstance(module, torch.nn.Linear):
                macs = int(output.numel()) * module.in_features
            records.append({
                "module": name, "type": type(module).__name__,
                "output_shapes": [list(value.shape) for value in values],
                "output_elements_per_sample": elements // batch_size,
                "conv_linear_macs_per_sample": macs // batch_size,
            })
        return hook

    for name, module in model.named_modules():
        if name and not list(module.children()):
            hooks.append(module.register_forward_hook(record(name)))
    try:
        with torch.inference_mode():
            outputs = [value.cpu().numpy() for value in model(example)]
    finally:
        for hook in hooks:
            hook.remove()
    conv_linear_macs = sum(value["conv_linear_macs_per_sample"] for value in records)
    gau = model.pose.head.gau
    attention_macs = model.joint_count ** 2 * (int(gau.s) + int(gau.e))
    geometry_macs = 8 * model.joint_count * (2 + 3)
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    simcc_elements = model.joint_count * (384 + 512)
    counts = {
        "input_shape_used": list(example.shape),
        "per_sample_input_shape": list(example.shape[1:]),
        "conv_linear_macs_per_sample": conv_linear_macs,
        "gau_qk_and_av_macs_per_sample": attention_macs,
        "derived_geometry_matmul_macs_per_sample": geometry_macs,
        "counted_macs_per_sample": conv_linear_macs + attention_macs + geometry_macs,
        "mac_definition": "One multiplication plus accumulation is one MAC. Includes all Conv2d/Linear, GAU QK and attention-times-value matrix products, and both fixed geometry matrices. Excludes pooling, normalization, activations, reductions, softmax, indexing and elementwise arithmetic; actual exported operators are inventoried separately.",
        "gau_matrix_products": {"tokens": model.joint_count, "qk_channels": int(gau.s), "value_channels": int(gau.e),
                                "qk_macs": model.joint_count ** 2 * int(gau.s), "av_macs": model.joint_count ** 2 * int(gau.e)},
        "parameter_elements": sum(int(value.numel()) for value in parameters),
        "trainable_parameter_elements": sum(int(value.numel()) for value in parameters if value.requires_grad),
        "parameter_bytes": sum(int(value.numel() * value.element_size()) for value in parameters),
        "checkpoint_buffer_bytes": sum(int(value.numel() * value.element_size()) for value in buffers),
        "checkpoint_buffer_scope": "Includes training-only residual calibration buffers; ONNX omits buffers that inference never references",
        "head_tensors_per_sample": {
            "simcc_x_shape": [model.joint_count, 384], "simcc_y_shape": [model.joint_count, 512],
            "simcc_logit_elements": simcc_elements, "simcc_logit_float32_bytes": simcc_elements * 4,
            "gau_attention_shape": [model.joint_count, model.joint_count],
            "gau_attention_elements": model.joint_count ** 2,
        },
        "largest_leaf_module_output": max(records, key=lambda value: value["output_elements_per_sample"]),
        "sum_leaf_module_output_elements_per_sample": sum(value["output_elements_per_sample"] for value in records),
        "activation_scope": "Logical intermediate sizes from actual student inference, not peak live memory. Module-output sums can count reused storage, and functional tensors are not all module outputs. No GPU utilization, FPS or latency is inferred.",
        "teacher_included": False,
        "leaf_modules": records,
    }
    return outputs, counts


def _load(path: Path) -> tuple[dict[str, Any], WholeBodyModel]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = WholeBodyModel(checkpoint["model_config"], pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint, model.eval()


def export(checkpoint_path: Path, data_root: Path, output_dir: Path,
           quality_gate: dict[str, float], baseline_path: Path | None = None,
           validation_samples: int = 645) -> dict[str, Any]:
    checkpoint, model = _load(checkpoint_path)
    step = int(checkpoint.get("step", 0))
    if step <= 0:
        raise RuntimeError("Refusing to publish an untrained parallel-stage checkpoint")
    baseline = None
    if baseline_path is not None:
        _, baseline = _load(baseline_path)
        if baseline.pretrained_provenance.get("sha256") != model.pretrained_provenance.get("sha256"):
            raise ValueError("Initial and final models were not initialized from the same pose checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, example = evaluate_2d(model, baseline, data_root, validation_samples, quality_gate)
    metrics.update(evaluate_3d(model, data_root, validation_samples))
    metrics.update(evaluate_rgb_body_3d(model, data_root))
    metrics["quality_gate"] = assess_quality(metrics, quality_gate)
    metrics["checkpoint_step"] = step
    # Preserve measured failures even if a later conversion or upload fails.
    _json(output_dir / "metrics.json", metrics)
    onnx_path = output_dir / "wholebody.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model, (example,), str(onnx_path), input_names=["image"], output_names=OUTPUT_NAMES,
            opset_version=17, dynamo=False, external_data=False,
            dynamic_axes={name: {0: "batch"} for name in ["image", *OUTPUT_NAMES]},
        )
    onnx.checker.check_model(str(onnx_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    reference, resource_usage = inference_with_resource_counts(model, example)
    actual = session.run(OUTPUT_NAMES, {"image": example.numpy()})
    comparison: dict[str, Any] = {}
    for name, expected, observed in zip(OUTPUT_NAMES, reference, actual):
        if not np.isfinite(observed).all():
            raise RuntimeError(f"ONNX output {name} contains non-finite values")
        maximum = float(np.max(np.abs(expected - observed)))
        matches = bool(np.allclose(expected, observed, rtol=1e-3, atol=1e-4))
        comparison[name] = {"shape": list(observed.shape), "max_absolute_error": maximum, "matches": matches}
        if not matches:
            raise RuntimeError(f"Learned PyTorch and ONNX outputs diverge for {name}: max_abs={maximum}")
    metrics.update({
        "onnx_equivalence": comparison,
        "onnx_bytes": onnx_path.stat().st_size,
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "parameters": resource_usage["parameter_elements"],
        "trainable_parameters": resource_usage["trainable_parameter_elements"],
        "counted_macs_per_sample": resource_usage["counted_macs_per_sample"],
        "resource_usage_file": "resource_usage.json",
    })
    schema = model.specification()
    schema.update({"opset": 17, "checkpoint_step": step})
    exported_graph = onnx.load(str(onnx_path))
    resource_usage.update({
        "checkpoint_step": step, "onnx_bytes": onnx_path.stat().st_size,
        "onnx_ir_version": int(exported_graph.ir_version),
        "onnx_operator_inventory": dict(sorted(Counter(node.op_type for node in exported_graph.graph.node).items())),
        "onnx_node_count": len(exported_graph.graph.node),
        "onnx_initializer_elements": sum(math.prod(value.dims) for value in exported_graph.graph.initializer),
        "onnx_initializer_bytes": sum(numpy_helper.to_array(value).nbytes for value in exported_graph.graph.initializer),
        "outputs": [{"name": name, "shape": list(value.shape), "elements_per_sample": int(value.size // example.shape[0])}
                    for name, value in zip(OUTPUT_NAMES, actual)],
    })
    _json(output_dir / "schema.json", schema)
    _json(output_dir / "resource_usage.json", resource_usage)
    _json(output_dir / "metrics.json", metrics)
    print(json.dumps({"checkpoint_step": step, "quality_gate": metrics["quality_gate"],
                      "rgb_body_mpjpe_mm": metrics["rgb_body_mpjpe_mm"],
                      "counted_macs_per_sample": metrics["counted_macs_per_sample"],
                      "onnx_sha256": metrics["onnx_sha256"]}, ensure_ascii=False), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("release"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--validation-samples", type=int, default=645)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    metrics = export(args.checkpoint, args.data_root, args.output_dir, quality_gate=config["quality_gate"],
                     baseline_path=args.baseline, validation_samples=args.validation_samples)
    if not metrics["quality_gate"]["passed"]:
        raise SystemExit("Declared quality criteria are unmet; candidate artifacts remain available for continued training")


if __name__ == "__main__":
    main()
