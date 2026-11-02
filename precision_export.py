"""Optional mixed-FP16 derivative of an already evaluated FP32 ONNX model.

Integration dependency: onnxconverter-common==1.16.0, alongside the existing
onnx==1.17.0 and onnxruntime==1.23.2 pins. This module does not install packages.

Call export_mixed_fp16 only after the primary export. Pass a PRIVATE directory
outside the primary release folder. Promote its explicitly returned files only
when eligible_for_publication is true; a derivative failure never invalidates
or overwrites wholebody.onnx. No model training state is loaded or changed.

Only the extracted backbone/final-convolution segment is converted. RGB
normalization and the entire GAU/SimCC/lifter/geometry remainder stay FP32.
The expected reduction is about one third of parameter-storage bytes, not a
claim about GPU utilization, latency, throughput or total device memory.
"""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import compose, numpy_helper, utils
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader, Subset

from data import CocoWholeBodyDataset
from joint_schema import GROUPS
from model import batch_2d


CONVERTER_VERSION = "1.16.0"
HALF_NAMESPACE = "fp16_trunk/"
FP16_MIN_POSITIVE = 5.960464477539063e-08
FP16_MAX_FINITE = 65504.0
SOURCE_REFERENCES = {
    "conversion_api": "https://onnxruntime.ai/docs/performance/model-optimizations/float16.html",
    "converter_version": "https://pypi.org/project/onnxconverter-common/1.16.0/",
    "converter_source": "https://github.com/microsoft/onnxconverter-common/blob/v1.16.0/onnxconverter_common/float16.py",
    "onnx_extraction": "https://github.com/onnx/onnx/blob/v1.17.0/onnx/utils.py",
    "onnx_composition": "https://github.com/onnx/onnx/blob/v1.17.0/onnx/compose.py",
    "cpu_fp32_fallback": "https://github.com/microsoft/onnxruntime/blob/v1.23.2/onnxruntime/core/optimizer/insert_cast_transformer.cc",
    "precision_limits": "https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _save_report(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    try:
        _write_json(path, report)
    except (OSError, TypeError, ValueError) as error:
        report.update({"status": "failed", "eligible_for_publication": False,
                       "files_for_publication": [], "report_file_error": str(error)})
    return report


def _scope(name: str, scope: str) -> bool:
    return name.startswith(scope.lstrip("/")) or name.startswith("/" + scope.lstrip("/"))


def _extract(extractor: utils.Extractor, inputs: list[str], outputs: list[str]) -> onnx.ModelProto:
    result = extractor.extract_model(inputs, outputs)
    # Extractor promotes intermediate tensors to graph I/O. Avoid duplicate
    # value_info declarations whose type could otherwise disagree with the
    # converter's explicit FP32 boundary types.
    io_names = set(inputs) | set(outputs)
    values = [value for value in result.graph.value_info if value.name not in io_names]
    del result.graph.value_info[:]
    result.graph.value_info.extend(values)
    onnx.checker.check_model(result)
    return result


def _assert_fp32_region(region: onnx.ModelProto) -> None:
    for value in (*region.graph.input, *region.graph.output, *region.graph.value_info):
        if value.type.HasField("tensor_type") and value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            raise ValueError(f"The protected source region already contains FP16: {value.name}")
    for tensor in region.graph.initializer:
        if tensor.data_type == onnx.TensorProto.FLOAT16:
            raise ValueError(f"A protected source initializer is not FP32: {tensor.name}")
    for node in region.graph.node:
        for attribute in node.attribute:
            if attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                raise ValueError("This precision derivative does not rewrite control-flow subgraphs")
            if attribute.type == onnx.AttributeProto.TENSOR and attribute.t.data_type == onnx.TensorProto.FLOAT16:
                raise ValueError(f"A protected source constant is FP16: {node.name}")
            if node.op_type == "Cast" and attribute.name == "to" and attribute.i == onnx.TensorProto.FLOAT16:
                raise ValueError(f"The protected source region casts to FP16: {node.name}")


def _audit_unchanged_region(region: onnx.ModelProto, mixed: onnx.ModelProto,
                            boundary_names: dict[str, str] | None = None) -> dict[str, Any]:
    """Require identical operations/weights, allowing only boundary wiring."""
    boundary_names = boundary_names or {}
    actual_nodes = {node.name: node for node in mixed.graph.node}
    actual_tensors = {tensor.name: tensor for tensor in mixed.graph.initializer}
    digest = hashlib.sha256()
    for node in region.graph.node:
        expected = copy.deepcopy(node)
        for index, name in enumerate(expected.input):
            expected.input[index] = boundary_names.get(name, name)
        serialized = expected.SerializeToString()
        if node.name not in actual_nodes or actual_nodes[node.name].SerializeToString() != serialized:
            raise ValueError(f"Protected FP32 operation changed: {node.name}")
        digest.update(serialized)
    for tensor in region.graph.initializer:
        serialized = tensor.SerializeToString()
        if tensor.name not in actual_tensors or actual_tensors[tensor.name].SerializeToString() != serialized:
            raise ValueError(f"Protected FP32 learned tensor changed: {tensor.name}")
        digest.update(serialized)
    return {"unchanged": True, "nodes": len(region.graph.node),
            "initializers": len(region.graph.initializer), "operations_and_weights_sha256": digest.hexdigest()}


def _initializer_summary(model: onnx.ModelProto) -> dict[str, Any]:
    elements: Counter[str] = Counter()
    byte_counts: Counter[str] = Counter()
    for tensor in model.graph.initializer:
        dtype = onnx.TensorProto.DataType.Name(tensor.data_type)
        elements[dtype] += math.prod(tensor.dims)
        byte_counts[dtype] += numpy_helper.to_array(tensor).nbytes
    return {"elements_by_dtype": dict(elements), "bytes_by_dtype": dict(byte_counts),
            "total_bytes": int(sum(byte_counts.values()))}


def _preserve_public_contract(source_values, candidate_values) -> None:
    """Keep exact public descriptors while permitting compatible shape inference."""
    if len(source_values) != len(candidate_values):
        raise ValueError("The derivative changed the number of public inputs or outputs")
    for original, candidate in zip(source_values, candidate_values):
        original_type, candidate_type = original.type.tensor_type, candidate.type.tensor_type
        if (original.name != candidate.name or original_type.elem_type != onnx.TensorProto.FLOAT
                or candidate_type.elem_type != onnx.TensorProto.FLOAT):
            raise ValueError("The derivative changed a primary FP32 input/output contract")
        original_dims, candidate_dims = original_type.shape.dim, candidate_type.shape.dim
        if len(original_dims) != len(candidate_dims):
            raise ValueError(f"The derivative changed tensor rank: {original.name}")
        for axis, (expected, inferred) in enumerate(zip(original_dims, candidate_dims)):
            if (expected.HasField("dim_value") and inferred.HasField("dim_value")
                    and expected.dim_value != inferred.dim_value):
                raise ValueError(f"Incompatible inferred shape: {original.name}")
            if axis == 0 and expected.HasField("dim_param") and inferred.HasField("dim_value"):
                raise ValueError(f"The derivative made a dynamic dimension static: {original.name}")
        # Extraction may refine internal symbolic names or add annotations.
        # Retain the original public names/dynamic dimensions and descriptions.
        candidate.CopyFrom(original)


def _convert_protected(source: onnx.ModelProto, output_names: list[str]) -> tuple[onnx.ModelProto, dict[str, Any]]:
    from onnxconverter_common import float16

    node_names = [node.name for node in source.graph.node]
    if any(not name for name in node_names) or len(set(node_names)) != len(node_names):
        raise ValueError("Precision conversion requires unique named source nodes")
    if source.functions or source.graph.sparse_initializer:
        raise ValueError("Unexpected local functions or sparse tensors in the primary model")
    convolutions = [node for node in source.graph.node if node.op_type == "Conv"]
    stem = [node for node in convolutions if _scope(node.name, "pose/backbone/")]
    final = [node for node in convolutions if _scope(node.name, "pose/head/final_layer/")]
    if not stem or len(final) != 1:
        raise ValueError("The exported backbone/final-convolution cut is not uniquely identifiable")
    first, last = stem[0], final[0]
    initializers = {tensor.name: tensor for tensor in source.graph.initializer}
    if first.input[1] not in initializers or list(initializers[first.input[1]].dims) != [12, 3, 3, 3]:
        raise ValueError("Unexpected first backbone convolution; refusing a different graph cut")
    if last.input[1] not in initializers or list(initializers[last.input[1]].dims) != [73, 384, 7, 7]:
        raise ValueError("Unexpected selected73 final convolution")
    entry, exit_tensor = first.input[0], last.output[0]
    extractor = utils.Extractor(source)
    prelude = _extract(extractor, ["image"], [entry])
    trunk = _extract(extractor, [entry], [exit_tensor])
    tail = _extract(extractor, [exit_tensor], output_names)
    _assert_fp32_region(prelude)
    _assert_fp32_region(tail)

    # Unexpected dependencies are rejected rather than accidentally converted.
    # Namespacing the middle segment permits shared constant tensors to remain
    # untouched for consumers outside the converted segment.
    for node in trunk.graph.node:
        if not (_scope(node.name, "pose/backbone/") or _scope(node.name, "pose/head/final_layer/")
                or node.op_type in ("Constant", "Identity")):
            raise ValueError(f"Unexpected dependency in the FP16 segment: {node.name} ({node.op_type})")
        if any(attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) for attribute in node.attribute):
            raise ValueError("The converted segment unexpectedly contains a nested graph")
    tiny_values = 0
    for tensor in trunk.graph.initializer:
        if tensor.data_type == onnx.TensorProto.FLOAT:
            values = numpy_helper.to_array(tensor)
            if not np.isfinite(values).all() or np.any(np.abs(values) > FP16_MAX_FINITE):
                raise ValueError(f"The learned tensor cannot be represented as finite FP16: {tensor.name}")
            tiny_values += int(((np.abs(values) > 0) & (np.abs(values) < FP16_MIN_POSITIVE)).sum())
    original_trunk_node_count = len(trunk.graph.node)
    converted = float16.convert_float_to_float16(
        trunk, keep_io_types=True, disable_shape_infer=False,
        min_positive_val=FP16_MIN_POSITIVE, max_finite_val=FP16_MAX_FINITE,
        check_fp16_ready=True,
    )
    onnx.checker.check_model(converted)
    if any(value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
           for value in (*converted.graph.input, *converted.graph.output)):
        raise ValueError("The converted segment did not retain FP32 boundary I/O")
    if not any(tensor.data_type == onnx.TensorProto.FLOAT16 for tensor in converted.graph.initializer):
        raise ValueError("The derivative contains no converted learned weights")

    # The prefix argument renames internal nodes, tensors and initializers and
    # adjusts the connected input name. It does not rename the image input of
    # the unmodified FP32 preprocessing graph.
    joined = compose.merge_models(prelude, converted, [(entry, entry)], prefix2=HALF_NAMESPACE)
    mixed_exit = HALF_NAMESPACE + exit_tensor
    mixed = compose.merge_models(joined, tail, [(mixed_exit, exit_tensor)])
    if [value.name for value in mixed.graph.input] != ["image"]:
        raise ValueError("An unexpected shared dependency added a derivative input")
    if [value.name for value in mixed.graph.output] != output_names:
        raise ValueError("The derivative changed the published output order")
    _preserve_public_contract(source.graph.input, mixed.graph.input)
    _preserve_public_contract(source.graph.output, mixed.graph.output)
    audit = {
        "rgb_preprocessing": _audit_unchanged_region(prelude, mixed),
        "gau_simcc_lifter_geometry": _audit_unchanged_region(tail, mixed, {exit_tensor: mixed_exit}),
        "cut_input": entry, "cut_output": exit_tensor,
        "converted_node_count_before_boundary_casts": original_trunk_node_count,
        "converted_initializer_elements": _initializer_summary(converted)["elements_by_dtype"].get("FLOAT16", 0),
        "tiny_nonzero_initializer_values_mapped_to_min_subnormal": tiny_values,
        "precision": "FP16 backbone and selected final convolution; continuous unchanged FP32 RGB preprocessing, GAU/SimCC decoding, lifter and geometry",
    }
    props = {entry.key: entry.value for entry in source.metadata_props}
    props.update({"precision": "mixed_fp16_backbone_final_conv_fp32_remainder",
                  "converter": f"onnxconverter-common=={CONVERTER_VERSION}",
                  "native_gpu_execution_validated": "false"})
    onnx.helper.set_model_props(mixed, props)
    onnx.checker.check_model(mixed)
    return mixed, audit


class _ORTCandidate:
    def __init__(self, session: ort.InferenceSession, output_names: list[str]) -> None:
        self.session, self.output_names = session, output_names

    def __call__(self, rgb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        inputs = np.ascontiguousarray(rgb.detach().cpu().numpy(), dtype=np.float32)
        outputs = self.session.run(self.output_names, {"image": inputs})
        expected_shapes = [(len(inputs), 73, 2), (len(inputs), 73), (len(inputs), 73, 3),
                           (len(inputs), 8, 2), (len(inputs), 8), (len(inputs), 8, 3)]
        if len(outputs) != len(expected_shapes):
            raise RuntimeError("The derivative did not return all six outputs")
        for name, value, shape in zip(self.output_names, outputs, expected_shapes):
            if value.dtype != np.float32 or tuple(value.shape) != shape or not np.isfinite(value).all():
                raise RuntimeError(f"Invalid derivative output {name}: dtype={value.dtype}, shape={value.shape}")
        return tuple(torch.from_numpy(value) for value in outputs)


def _evaluate_candidate(candidate: _ORTCandidate, data_root: Path,
                        primary: dict[str, Any], thresholds: dict[str, float],
                        count: int, rgb_max_frames: int) -> dict[str, Any]:
    # Deferred import allows export_model to call this helper after completing
    # its own definitions and primary export, without an import cycle.
    from export_model import (
        _2d_counter, _accumulate_2d, _finish_2d, _indices,
        assess_quality, evaluate_rgb_body_3d,
    )

    dataset = CocoWholeBodyDataset(data_root, split="val")
    indices = _indices(len(dataset), count)
    if primary["validation2d_samples"] != len(indices):
        raise ValueError("Primary and derivative COCO evaluation sample counts differ")
    definitions = primary["2d_metric_definitions"]
    if (definitions["crop_pck_threshold"] != thresholds["crop_pck_threshold"]
            or definitions["hand_local_pck_threshold"] != thresholds["hand_local_pck_threshold"]):
        raise ValueError("Primary and derivative PCK definitions differ")
    totals = _2d_counter()
    sample_ids: list[str] = []
    with torch.inference_mode():
        for batch in DataLoader(Subset(dataset, indices), batch_size=8, shuffle=False):
            prediction = candidate(batch["image"])[0]
            target, valid = batch_2d(batch)
            _accumulate_2d(totals, prediction, target, valid,
                           float(thresholds["crop_pck_threshold"]), float(thresholds["hand_local_pck_threshold"]))
            sample_ids.extend(batch["sample_id"])
    student = _finish_2d(totals)
    for group in ("all", *GROUPS):
        for key in ("labeled_joint_count", "local_labeled_joint_count"):
            if student[group][key] != primary["student_2d"][group][key]:
                raise ValueError(f"Primary and derivative annotated COCO cohorts differ: {group}/{key}")
    # Copy only explicit baseline/lifter measurements. In particular, never
    # copy the primary model's ONNX-equivalence result into derivative metrics.
    reused = ("teacher_2d", "initial_2d", "validation3d_samples_by_source", "3d_clean",
              "3d_corrupted", "3d_by_source", "residual_calibration_samples", "3d_metric_scope")
    metrics = {key: copy.deepcopy(primary[key]) for key in reused}
    metrics.update({
        "validation2d_samples": len(indices), "student_2d": student,
        "group_2d": {name: student[name] for name in GROUPS},
        "normalized_2d_error": student["all"]["normalized_2d_error"], "pck_crop": student["all"]["pck_crop"],
        "2d_metric_definitions": copy.deepcopy(definitions),
        "coco_cohort": {
            "sampler": "Same export_model._indices over the same immutable prepared validation dataset",
            "annotation_manifest_sha256": _sha256(data_root / "coco_val.json"),
            "sample_ids_sha256": hashlib.sha256(json.dumps(sample_ids, separators=(",", ":")).encode()).hexdigest(),
            "source_length": len(dataset), "selected_samples": len(indices),
            "primary_label_counts_match": True,
        },
    })
    rgb_metrics = evaluate_rgb_body_3d(candidate, data_root, max_frames=rgb_max_frames)
    expected_rgb = primary["rgb_body_3d"]["dataset"]
    observed_rgb = rgb_metrics["rgb_body_3d"]["dataset"]
    for key in ("subject", "clip", "camera", "source_frame_indices", "crop", "crop_source", "video"):
        if observed_rgb[key] != expected_rgb[key]:
            raise ValueError(f"Primary and derivative ASPset RGB cohorts differ: {key}")
    metrics.update(rgb_metrics)
    metrics["quality_gate"] = assess_quality(metrics, thresholds)
    metrics["change_vs_primary_fp32"] = {
        "2d": {
            group: {
                key: student[group][key] - primary["student_2d"][group][key]
                for key in ("normalized_2d_error", "pck_crop", "local_normalized_error", "pck_local")
                if student[group][key] is not None and primary["student_2d"][group][key] is not None
            } for group in ("all", *GROUPS)
        },
        "rgb_body_mpjpe_mm": (metrics["rgb_body_mpjpe_mm"] - primary["rgb_body_mpjpe_mm"]
                              if metrics["rgb_body_mpjpe_mm"] is not None and primary["rgb_body_mpjpe_mm"] is not None else None),
    }
    metrics["reused_measurement_scope"] = {
        "fp32_primary_sha256": primary["onnx_sha256"],
        "teacher_and_initial_2d": "Shared primary baselines on the identical deterministic held-out COCO cohort",
        "clean_and_corrupted_3d": "Reused only because the lifter's operations, constants and learned weights are byte-for-byte unchanged FP32. Corrupted input still uses the original training-derived residual bank; it is not a new FP16 residual calibration.",
        "actual_recomputed": "All candidate2D part metrics and RGB-to-body3D on the complete same held-out cohorts",
    }
    return metrics


def export_mixed_fp16(fp32_path: Path, data_root: Path, private_output_dir: Path,
                      fp32_metrics: dict[str, Any], quality_gate: dict[str, float],
                      validation_samples: int = 645, rgb_max_frames: int = 192) -> dict[str, Any]:
    """Return a private candidate report; promote files only when eligible.

    The caller must pass metrics from the just-completed primary export using
    the same data_root and sampling settings. Source ONNX SHA, cohort sizes,
    label counts and RGB acquisition identity are checked before approval.
    Conversion/data/runtime/quality failures are recorded and returned, not
    propagated into the primary FP32 publication path.
    """
    fp32_path, data_root = Path(fp32_path), Path(data_root)
    private_output_dir = Path(private_output_dir)
    report: dict[str, Any] = {
        "status": "preparing", "eligible_for_publication": False, "files_for_publication": [],
        "primary_artifact": "wholebody.onnx", "candidate_artifact": "wholebody-fp16.onnx",
        "source_references": SOURCE_REFERENCES,
        "runtime_scope": "Held-out inference uses CPUExecutionProvider, which may promote unsupported FP16 operators to FP32. This establishes artifact/cohort quality under that provider, not native GPU FP16 numerics or GPU performance.",
        "native_gpu_execution_validated": False,
        "expected_storage_reduction": "About33% of FP32 parameter bytes from converting approximately3.72M of5.67M parameters; actual file/initializer sizes are recorded below. MACs are unchanged.",
    }
    public_root, private_root = fp32_path.parent.resolve(), private_output_dir.resolve()
    if private_root == public_root or public_root in private_root.parents:
        report.update({"status": "failed", "stage": "destination", "error_type": "ValueError",
                       "error": "The candidate directory must be outside the primary publication folder"})
        return report
    try:
        private_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        report.update({"status": "failed", "stage": "destination", "error_type": type(error).__name__,
                       "error": str(error)})
        return report
    report_path = private_output_dir / "precision_report.json"
    metrics_path = private_output_dir / "metrics-fp16.json"
    candidate_path = private_output_dir / "wholebody-fp16.onnx"
    try:
        report["stage"] = "primary_identity"
        primary_hash = _sha256(fp32_path)
        if primary_hash != fp32_metrics["onnx_sha256"]:
            raise ValueError("The derivative source does not match the evaluated FP32 ONNX bytes")
        report["fp32_primary_sha256"] = primary_hash
        if quality_gate != fp32_metrics["quality_gate"]["thresholds"]:
            raise ValueError("The derivative quality thresholds differ from the declared primary criteria")
        if not fp32_metrics["quality_gate"]["passed"]:
            report["status"] = "skipped_primary_quality_not_met"
            return _save_report(report_path, report)
        if version("onnxconverter-common") != CONVERTER_VERSION:
            raise RuntimeError(f"Precision export requires onnxconverter-common=={CONVERTER_VERSION}")
        report["versions"] = {"onnxconverter-common": CONVERTER_VERSION, "onnx": onnx.__version__, "onnxruntime": ort.__version__}
        from export_model import OUTPUT_NAMES

        report["stage"] = "convert_protected_segment"
        source = onnx.load(str(fp32_path))
        mixed, audit = _convert_protected(source, OUTPUT_NAMES)
        onnx.save_model(mixed, str(candidate_path), save_as_external_data=False)
        report["precision_audit"] = audit
        report["source_initializers"] = _initializer_summary(source)
        report["candidate_initializers"] = _initializer_summary(mixed)
        report["fp32_bytes"] = fp32_path.stat().st_size
        report["candidate_bytes"] = candidate_path.stat().st_size
        report["file_size_reduction_fraction"] = 1.0 - report["candidate_bytes"] / report["fp32_bytes"]
        report["candidate_sha256"] = _sha256(candidate_path)
        report["operator_inventory"] = dict(sorted(Counter(node.op_type for node in mixed.graph.node).items()))
        report["stage"] = "held_out_inference"
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        session = ort.InferenceSession(str(candidate_path), sess_options=options, providers=["CPUExecutionProvider"])
        candidate = _ORTCandidate(session, OUTPUT_NAMES)
        metrics = _evaluate_candidate(candidate, data_root, fp32_metrics, quality_gate, validation_samples, rgb_max_frames)
        metrics.update({"artifact": candidate_path.name, "onnx_sha256": report["candidate_sha256"],
                        "onnx_bytes": report["candidate_bytes"], "checkpoint_step": fp32_metrics["checkpoint_step"],
                        "evaluation_providers": session.get_providers(), "native_gpu_execution_validated": False,
                        "precision": audit["precision"], "precision_audit": audit})
        _write_json(metrics_path, metrics)
        report["quality_gate"] = metrics["quality_gate"]
        report["evaluation_providers"] = session.get_providers()
        report["eligible_for_publication"] = bool(metrics["quality_gate"]["passed"])
        report["status"] = "quality_criteria_passed" if report["eligible_for_publication"] else "quality_criteria_unmet"
        report["stage"] = "complete"
        if report["eligible_for_publication"]:
            report["files_for_publication"] = [str(candidate_path), str(metrics_path), str(report_path)]
    except Exception as error:
        report.update({"status": "failed", "eligible_for_publication": False,
                       "files_for_publication": [], "error_type": type(error).__name__, "error": str(error)})
    return _save_report(report_path, report)
