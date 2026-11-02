"""Export the learned checkpoint and record inference on real validation data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader, Subset

from data import ASPsetDataset, CocoWholeBodyDataset
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
    example = None
    with torch.inference_mode():
        for batch in DataLoader(Subset(dataset2d, indices2d), batch_size=8, shuffle=False):
            if example is None:
                example = batch["image"][:2].clone()
            predicted, _, _ = model(batch["image"])
            target = batch["keypoints"]
            mask = (batch["visibility"] > 0) & torch.isfinite(target).all(-1) & (target >= 0).all(-1) & (target <= 1).all(-1)
            delta = (predicted - torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)).square().sum(-1).sqrt()
            error2d += float((delta * mask).sum())
            correct2d += float(((delta <= 0.05) & mask).sum())
            points2d += float(mask.sum())
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
        "lifter_mpjpe_mm_from_ground_truth_2d": 1000 * error3d / points3d,
        "metric_scope": "2D is evaluated from real COCO validation RGB crops. 3D is evaluated from projected ASPset validation joints, not from RGB video; this is not end-to-end RGB-to-3D accuracy.",
    }, example.clone()


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
    output_names = ["keypoints2d", "confidence2d", "body3d"]
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
    with torch.inference_mode():
        reference = [value.numpy() for value in model(example)]
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
    _json(output_dir / "schema.json", schema)
    _json(output_dir / "metrics.json", metrics)
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
