"""Export the learned checkpoint and record inference on real validation data."""

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
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader, Subset

from data import ASPsetDataset, CocoWholeBodyDataset
from joint_schema import GROUPS
from model import WholeBodyModel


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate(model: WholeBodyModel, data_root: Path, count: int) -> tuple[dict[str, Any], torch.Tensor]:
    dataset2d = CocoWholeBodyDataset(data_root, split="val")
    dataset3d = ASPsetDataset(data_root, split="val")
    n2d, n3d = min(count, len(dataset2d)), min(count, len(dataset3d))
    if n2d <= 0 or n3d <= 0:
        raise RuntimeError("Export requires actual held-out 2D and 3D samples")
    indices2d = np.linspace(0, len(dataset2d) - 1, n2d, dtype=np.int64).tolist()
    indices3d = np.linspace(0, len(dataset3d) - 1, n3d, dtype=np.int64).tolist()
    error2d = correct2d = points2d = error3d = points3d = 0.0
    group_totals = {name: {"error": 0.0, "correct": 0.0, "count": 0} for name in GROUPS}
    example = None
    with torch.inference_mode():
        for batch in DataLoader(Subset(dataset2d, indices2d), batch_size=8, shuffle=False):
            if example is None:
                example = batch["image"][:2].clone()
            predicted = model(batch["image"])[0]
            target = batch["keypoints"]
            mask = (batch["visibility"] > 0) & torch.isfinite(target).all(-1) & (target >= 0).all(-1) & (target <= 1).all(-1)
            delta = (predicted - torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)).square().sum(-1).sqrt()
            error2d += float((delta * mask).sum())
            correct2d += float(((delta <= 0.05) & mask).sum())
            points2d += float(mask.sum())
            for name, ids in GROUPS.items():
                group_totals[name]["error"] += float((delta[:, ids] * mask[:, ids]).sum())
                group_totals[name]["correct"] += float(((delta[:, ids] <= 0.05) & mask[:, ids]).sum())
                group_totals[name]["count"] += int(mask[:, ids].sum())
        for batch in DataLoader(Subset(dataset3d, indices3d), batch_size=256, shuffle=False):
            input_valid = torch.isfinite(batch["keypoints"]).all(-1).all(-1, keepdim=True)
            predicted = model.lift_body(torch.nan_to_num(batch["keypoints"], nan=0.0, posinf=0.0, neginf=0.0))
            target = batch["target3d"]
            mask = (batch["visibility"] > 0) & torch.isfinite(target).all(-1) & input_valid
            delta = (predicted - torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)).square().sum(-1).sqrt()
            error3d += float((delta * mask).sum())
            points3d += float(mask.sum())
    if not points2d or not points3d or example is None:
        raise RuntimeError("Validation samples did not contain real labeled joints")
    return {
        "validation2d_samples": n2d,
        "validation3d_samples": n3d,
        "normalized_2d_error": error2d / points2d,
        "pck_at_0_05_crop": correct2d / points2d,
        "group_2d": {
            name: {
                "labeled_joint_count": total["count"],
                "normalized_2d_error": total["error"] / total["count"] if total["count"] else None,
                "pck_at_0_05_crop": total["correct"] / total["count"] if total["count"] else None,
            } for name, total in group_totals.items()
        },
        "lifter_mpjpe_mm_from_ground_truth_2d": 1000 * error3d / points3d,
        "metric_scope": "2D is evaluated from real COCO validation RGB crops. 3D is evaluated from projected ASPset validation joints, not from RGB video; this is not end-to-end RGB-to-3D accuracy.",
    }, example.clone()


def inference_with_resource_counts(model: WholeBodyModel, example: torch.Tensor) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Count shapes and arithmetic during the required real-image inference.

    There is no timer or synthetic benchmark. Module output sizes are logical
    tensor sizes, not peak live device memory; ONNX can fuse or reuse storage.
    """
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
            outputs = [value.numpy() for value in model(example)]
    finally:
        for hook in hooks:
            hook.remove()
    largest = max(records, key=lambda value: value["output_elements_per_sample"])
    conv_linear_macs = sum(value["conv_linear_macs_per_sample"] for value in records)
    geometry_macs = 8 * model.joint_count * 2
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    heatmap_elements = model.joint_count * model.heatmap_height * model.heatmap_width
    counts = {
        "input_shape_used": list(example.shape),
        "per_sample_input_shape": list(example.shape[1:]),
        "conv_linear_macs_per_sample": conv_linear_macs,
        "derived_geometry_matmul_macs_per_sample": geometry_macs,
        "counted_macs_per_sample": conv_linear_macs + geometry_macs,
        "mac_definition": "One multiplication plus accumulation is one MAC. Counts include Conv2d/Linear and the fixed geometry matrix; exclude interpolation, normalization, activations, reductions, softmax, indexing, and elementwise arithmetic. Those operators are listed separately in the actual ONNX inventory.",
        "parameter_elements": sum(int(value.numel()) for value in parameters),
        "trainable_parameter_elements": sum(int(value.numel()) for value in parameters if value.requires_grad),
        "parameter_bytes": sum(int(value.numel() * value.element_size()) for value in parameters),
        "buffer_bytes": sum(int(value.numel() * value.element_size()) for value in buffers),
        "head_tensors_per_sample": {
            "score_shape": [model.joint_count, model.heatmap_height, model.heatmap_width],
            "score_elements": heatmap_elements,
            "offset_elements": 2 * heatmap_elements,
            "combined_score_offset_elements": 3 * heatmap_elements,
            "combined_score_offset_float32_bytes": 12 * heatmap_elements,
            "shared_feature_elements": int(model.config["head_channels"]) * model.heatmap_height * model.heatmap_width,
        },
        "largest_leaf_module_output": largest,
        "sum_leaf_module_output_elements_per_sample": sum(value["output_elements_per_sample"] for value in records),
        "activation_scope": "Recorded module outputs are logical intermediate sizes from this compact model. Their sum can count reused memory more than once; functional tensors are not all module outputs. This is not measured peak GPU memory, GPU utilization, FPS, or latency.",
        "leaf_modules": records,
    }
    return outputs, counts


def export(checkpoint_path: Path, data_root: Path, output_dir: Path, validation_samples: int = 128) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    step = int(checkpoint.get("step", 0))
    if step <= 0:
        raise RuntimeError("Refusing to publish an untrained checkpoint (step <= 0)")
    model = WholeBodyModel(checkpoint.get("model_config"), pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, example = evaluate(model, data_root, validation_samples)
    onnx_path = output_dir / "wholebody.onnx"
    output_names = ["keypoints2d", "confidence2d", "body3d", "geometry2d", "geometry_confidence"]
    with torch.no_grad():
        torch.onnx.export(
            model, (example,), str(onnx_path), input_names=["image"], output_names=output_names,
            opset_version=17, dynamo=False, external_data=False,
            dynamic_axes={name: {0: "batch"} for name in ["image", *output_names]},
        )
    onnx.checker.check_model(str(onnx_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    reference, resource_usage = inference_with_resource_counts(model, example)
    actual = session.run(output_names, {"image": example.numpy()})
    comparison: dict[str, Any] = {}
    for name, expected, observed in zip(output_names, reference, actual):
        if not np.isfinite(observed).all():
            raise RuntimeError(f"ONNX output {name} contains non-finite values")
        max_abs = float(np.max(np.abs(expected - observed)))
        matches = bool(np.allclose(expected, observed, rtol=1e-3, atol=1e-4))
        comparison[name] = {"shape": list(observed.shape), "max_absolute_error": max_abs, "matches": matches}
        if not matches:
            raise RuntimeError(f"Learned PyTorch and ONNX outputs diverge for {name}: max_abs={max_abs}")
    metrics.update({
        "checkpoint_step": step,
        "onnx_equivalence": comparison,
        "onnx_bytes": onnx_path.stat().st_size,
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    })
    schema = model.specification()
    schema["opset"] = 17
    schema["checkpoint_step"] = step
    exported_graph = onnx.load(str(onnx_path))
    resource_usage.update({
        "checkpoint_step": step,
        "onnx_bytes": onnx_path.stat().st_size,
        "onnx_ir_version": int(exported_graph.ir_version),
        "onnx_operator_inventory": dict(sorted(Counter(node.op_type for node in exported_graph.graph.node).items())),
        "onnx_node_count": len(exported_graph.graph.node),
        "onnx_initializer_elements": sum(math.prod(value.dims) for value in exported_graph.graph.initializer),
        "onnx_initializer_bytes": sum(onnx.numpy_helper.to_array(value).nbytes for value in exported_graph.graph.initializer),
        "outputs": [{"name": name, "shape": list(value.shape), "elements_per_sample": int(value.size // example.shape[0])}
                    for name, value in zip(output_names, actual)],
    })
    metrics["resource_usage_file"] = "resource_usage.json"
    _json(output_dir / "schema.json", schema)
    _json(output_dir / "metrics.json", metrics)
    _json(output_dir / "resource_usage.json", resource_usage)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("release"))
    parser.add_argument("--validation-samples", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    export(args.checkpoint, args.data_root, args.output_dir, args.validation_samples)


if __name__ == "__main__":
    main()
