"""Pose-pretrained compact whole-body inference and confidence-aware 3D lifting."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from joint_schema import (
    COMPACT_COCO_IDS, COMPACT_JOINT_NAMES, GEOMETRY2D_SCHEMA, GEOMETRY2D_WEIGHTS,
    GEOMETRY3D_SCHEMA, GEOMETRY3D_WEIGHTS, GROUPS, JOINT_SCHEMA,
)
from pose_architecture import PoseNetwork, load_pretrained_pose


DEFAULT_CONFIG: dict[str, Any] = {
    "image_height": 256, "image_width": 192,
    "lifter_width": 512, "lifter_blocks": 2,
    "distillation_weight": 1.0, "distillation_temperature": 2.0,
    "simcc_beta": 10.0, "calibration_samples": 512,
    "corruption_probability": 0.9, "corruption_noise_std": 0.003,
    "corruption_joint_dropout": 0.04, "corruption_region_dropout": 0.12,
}
REGIONS_3D = {
    "body": (list(range(12)), None),
    "left_hand": (list(range(12, 32)), 4),
    "right_hand": (list(range(32, 52)), 5),
    "left_foot": (list(range(52, 55)), 10),
    "right_foot": (list(range(55, 58)), 11),
    "face": (list(range(58, 73)), 60),
}


def batch_2d(batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
    return batch["x2d"] if "x2d" in batch else batch["keypoints"], batch["valid2d"] if "valid2d" in batch else batch["visibility"]


def batch_3d(batch: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    xy, valid2d = batch_2d(batch)
    xyz = batch["x3d"] if "x3d" in batch else batch["target3d"]
    valid3d = batch["valid3d"] if "valid3d" in batch else batch["visibility"]
    return xy, valid2d, xyz, valid3d


class ResidualLinear(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layers(x)


class WholeBodyLifter(nn.Module):
    def __init__(self, width: int, blocks: int) -> None:
        super().__init__()
        self.register_buffer("aspect_scale", torch.tensor([0.75, 1.0]).view(1, 1, 2))
        self.input_layer = nn.Sequential(nn.Linear(73 * 3, width), nn.ReLU())
        self.blocks = nn.Sequential(*(ResidualLinear(width) for _ in range(blocks)))
        self.output_layer = nn.Linear(width, 73 * 3)

    def reference_frame(self, crop_xy: Tensor, confidence: Tensor) -> tuple[Tensor, Tensor]:
        """Return the exact inference center and RMS radius in crop-height units."""
        confidence = confidence.clamp(0, 1)
        xy = crop_xy * self.aspect_scale
        body_weight = confidence[:, :12]
        body_amount = body_weight.sum(dim=1, keepdim=True)
        all_amount = confidence.sum(dim=1, keepdim=True)
        all_center = (xy * confidence[:, :, None]).sum(dim=1, keepdim=True) / all_amount.clamp_min(1e-6)[:, :, None]
        body_center = (xy[:, :12] * body_weight[:, :, None]).sum(dim=1, keepdim=True) / body_amount.clamp_min(1e-6)[:, :, None]
        body_center = torch.where((body_amount > 0)[:, :, None], body_center, all_center)
        hip_weight = confidence[:, 6:8]
        hip_center = (xy[:, 6:8] * hip_weight[:, :, None]).sum(dim=1, keepdim=True) / hip_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)[:, :, None]
        center = torch.where((hip_weight.sum(dim=1, keepdim=True) > 0)[:, :, None], hip_center, body_center)
        centered = xy - center
        squared_radius = centered.square().sum(dim=-1)
        body_radius = (squared_radius[:, :12] * body_weight).sum(dim=1, keepdim=True) / body_amount.clamp_min(1e-6)
        all_radius = (squared_radius * confidence).sum(dim=1, keepdim=True) / all_amount.clamp_min(1e-6)
        scale = torch.where(body_amount > 0, body_radius, all_radius).sqrt().clamp_min(1e-4)
        return center, scale

    def forward(self, crop_xy: Tensor, confidence: Tensor) -> Tensor:
        confidence = confidence.clamp(0, 1)
        center, scale = self.reference_frame(crop_xy, confidence)
        normalized = (crop_xy * self.aspect_scale - center) / scale[:, :, None]
        normalized = normalized * (confidence > 0)[:, :, None]
        features = torch.cat((normalized, confidence[:, :, None]), dim=-1).flatten(1)
        xyz = self.output_layer(self.blocks(self.input_layer(features))).reshape(-1, 73, 3)
        return xyz - (xyz[:, 6:7] + xyz[:, 7:8]) * 0.5


class WholeBodyModel(nn.Module):
    def __init__(self, config: dict[str, Any] | None = None, pretrained: bool = False) -> None:
        super().__init__()
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        if (self.config["image_height"], self.config["image_width"]) != (256, 192):
            raise ValueError("The retained pose-pretrained SimCC head requires native RGB 256x192 input")
        self.joint_count = len(COMPACT_COCO_IDS)
        self.pose = PoseNetwork(num_joints=self.joint_count)
        self.pretrained_provenance = self.config.get("pretrained_provenance", {})
        if pretrained:
            self.pretrained_provenance = load_pretrained_pose(self.pose, coco_ids=COMPACT_COCO_IDS)
            self.config["pretrained_provenance"] = self.pretrained_provenance
        self.lifter = WholeBodyLifter(int(self.config["lifter_width"]), int(self.config["lifter_blocks"]))
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("source_ids", torch.tensor(COMPACT_COCO_IDS, dtype=torch.long))
        self.register_buffer("simcc_x_grid", torch.arange(384, dtype=torch.float32))
        self.register_buffer("simcc_y_grid", torch.arange(512, dtype=torch.float32))
        self.register_buffer("geometry_weights", torch.tensor(GEOMETRY2D_WEIGHTS, dtype=torch.float32))
        self.register_buffer("geometry3d_weights", torch.tensor(GEOMETRY3D_WEIGHTS, dtype=torch.float32))
        self.register_buffer("geometry_source_mask", self.geometry_weights != 0)
        self.register_buffer("geometry_scale", torch.tensor([row["coordinate_scale"] for row in GEOMETRY2D_SCHEMA["rows"]], dtype=torch.float32).unsqueeze(0))
        capacity = int(self.config["calibration_samples"])
        self.register_buffer("residual_bank", torch.zeros(capacity, 73, 2))
        self.register_buffer("residual_confidence", torch.zeros(capacity, 73))
        self.register_buffer("residual_valid", torch.zeros(capacity, 73, dtype=torch.bool))
        self.register_buffer("residual_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("training_progress", torch.zeros(()))
        # Deliberately bypass nn.Module registration: the full teacher is never
        # in optimizer parameters, checkpoints, resource counts, or ONNX.
        object.__setattr__(self, "_teacher", None)
        self.train(True)

    def train(self, mode: bool = True) -> "WholeBodyModel":
        super().train(mode)
        for module in self.pose.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def set_training_progress(self, fraction: float) -> None:
        self.training_progress.fill_(max(0.0, min(1.0, float(fraction))))

    def get_teacher(self) -> PoseNetwork:
        if self._teacher is None:
            with torch.random.fork_rng(devices=[]):
                teacher = PoseNetwork(num_joints=133)
                provenance = load_pretrained_pose(teacher)
            expected = self.pretrained_provenance.get("sha256")
            if not expected or provenance.get("sha256") != expected:
                raise RuntimeError("Teacher checkpoint SHA256 does not match the student's pose initialization")
            teacher.requires_grad_(False).eval()
            object.__setattr__(self, "_teacher", teacher)
        return self._teacher.to(self.image_mean.device)

    def pose_logits(self, rgb: Tensor) -> tuple[Tensor, Tensor]:
        return self.pose((rgb - self.image_mean) / self.image_std)

    def decode_simcc(self, logits_x: Tensor, logits_y: Tensor) -> tuple[Tensor, Tensor]:
        probability_x = (logits_x * float(self.config["simcc_beta"])).softmax(dim=-1)
        probability_y = (logits_y * float(self.config["simcc_beta"])).softmax(dim=-1)
        score_x, position_x = probability_x.max(dim=-1)
        score_y, position_y = probability_y.max(dim=-1)
        xy = torch.stack((position_x.to(logits_x.dtype) / 384.0, position_y.to(logits_y.dtype) / 512.0), dim=-1)
        return xy, torch.minimum(score_x, score_y)

    def predict_2d(self, rgb: Tensor) -> tuple[Tensor, Tensor]:
        return self.decode_simcc(*self.pose_logits(rgb))

    def teacher_predictions(self, rgb: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            x, y = self.get_teacher()((rgb - self.image_mean) / self.image_std)
            return self.decode_simcc(x.index_select(1, self.source_ids), y.index_select(1, self.source_ids))

    def lift_body(self, crop_xy: Tensor, confidence: Tensor | None = None) -> Tensor:
        if confidence is None:
            confidence = torch.ones_like(crop_xy[..., 0])
        return self.lifter(crop_xy, confidence)

    def derived_geometry(self, points: Tensor, confidence: Tensor) -> tuple[Tensor, Tensor]:
        geometry = torch.matmul(self.geometry_weights, points) * self.geometry_scale
        sources = torch.where(self.geometry_source_mask[None], confidence[:, None], torch.ones_like(confidence[:, None]))
        return geometry, sources.amin(dim=-1)

    def forward(self, rgb: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        points, confidence = self.predict_2d(rgb)
        xyz = self.lift_body(points, confidence)
        geometry, geometry_confidence = self.derived_geometry(points, confidence)
        geometry3d = torch.matmul(self.geometry3d_weights, xyz)
        return points, confidence, xyz, geometry, geometry_confidence, geometry3d

    def loss_2d(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        logits_x, logits_y = self.pose_logits(batch["image"])
        target, validity = batch_2d(batch)
        visible = (validity > 0) & torch.isfinite(target).all(-1) & (target >= 0).all(-1) & (target <= 1).all(-1)
        safe_target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 1)
        gaussian_x = torch.exp(-0.5 * ((self.simcc_x_grid[None, None] - safe_target[..., 0, None] * 384) / 4.9).square())
        gaussian_y = torch.exp(-0.5 * ((self.simcc_y_grid[None, None] - safe_target[..., 1, None] * 512) / 5.66).square())
        beta = float(self.config["simcc_beta"])
        gt_loss = 0.5 * (
            F.kl_div((logits_x * beta).log_softmax(-1), (gaussian_x * 10).softmax(-1), reduction="none").sum(-1)
            + F.kl_div((logits_y * beta).log_softmax(-1), (gaussian_y * 10).softmax(-1), reduction="none").sum(-1)
        )
        with torch.no_grad():
            teacher_x, teacher_y = self.get_teacher()((batch["image"] - self.image_mean) / self.image_std)
            teacher_x = teacher_x.index_select(1, self.source_ids)
            teacher_y = teacher_y.index_select(1, self.source_ids)
        temperature = float(self.config["distillation_temperature"])
        kd_loss = 0.5 * temperature ** 2 * (
            F.kl_div((logits_x * beta / temperature).log_softmax(-1), (teacher_x * beta / temperature).softmax(-1), reduction="none").sum(-1)
            + F.kl_div((logits_y * beta / temperature).log_softmax(-1), (teacher_y * beta / temperature).softmax(-1), reduction="none").sum(-1)
        )
        weight = visible.to(logits_x.dtype)
        denominator = weight.sum(-1).clamp_min(1)
        gt = ((gt_loss * weight).sum(-1) / denominator).mean()
        kd = ((kd_loss * weight).sum(-1) / denominator).mean()
        loss = gt + float(self.config["distillation_weight"]) * kd
        with torch.no_grad():
            points, _ = self.decode_simcc(logits_x, logits_y)
            error = (points - safe_target).square().sum(-1).sqrt()
        return loss, {"loss2d": loss.detach(), "supervised2d": gt.detach(), "distillation2d": kd.detach(),
                      "mean_xy_error": (error * weight).sum().detach() / weight.sum().clamp_min(1), "labeled2d": weight.sum().detach()}

    def calibrate_residuals(self, batches: Iterable[dict[str, Any]], max_samples: int = 512) -> dict[str, Any]:
        was_training = self.training
        self.eval()
        capacity = min(int(max_samples), self.residual_bank.shape[0])
        count = 0
        self.residual_bank.zero_()
        self.residual_confidence.zero_()
        self.residual_valid.zero_()
        try:
            with torch.no_grad():
                for batch in batches:
                    declared_split = batch.get("split")
                    if declared_split is not None and any(value != "train" for value in ([declared_split] if isinstance(declared_split, str) else declared_split)):
                        raise ValueError("Residual calibration must use only the training split")
                    target, validity = batch_2d(batch)
                    image = batch["image"].to(self.image_mean.device)
                    target, validity = target.to(image.device), validity.to(image.device)
                    predicted, confidence = self.predict_2d(image)
                    valid = (validity > 0) & torch.isfinite(target).all(-1) & (target >= 0).all(-1) & (target <= 1).all(-1)
                    safe_target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
                    _, body_scale = self.lifter.reference_frame(safe_target, valid.to(target.dtype))
                    # A face-only annotation cannot define body-relative noise.
                    supported_rows = (valid[:, :12].sum(-1) >= 4) & (body_scale[:, 0] > 1e-3)
                    predicted, safe_target = predicted[supported_rows], safe_target[supported_rows]
                    confidence, valid = confidence[supported_rows], valid[supported_rows]
                    body_scale = body_scale[supported_rows]
                    normalized_residual = ((predicted - safe_target) * self.lifter.aspect_scale
                                           / body_scale[:, :, None])
                    n = min(capacity - count, len(predicted))
                    self.residual_bank[count:count+n].copy_(torch.where(valid[:n, :, None], normalized_residual[:n], torch.zeros_like(predicted[:n])))
                    self.residual_confidence[count:count+n].copy_(confidence[:n])
                    self.residual_valid[count:count+n].copy_(valid[:n])
                    count += n
                    if count >= capacity:
                        break
        finally:
            self.train(was_training)
        self.residual_count.fill_(count)
        if count <= 0:
            raise RuntimeError("No training samples were available for residual calibration")
        return {"samples": count, "valid_landmarks": int(self.residual_valid[:count].sum()),
                "source": "current pose model residuals on training RGB only", "whole_row_sampling": True,
                "residual_units": "isotropic2D displacement divided by annotated body RMS; converted to the target crop's body RMS before corruption",
                "minimum_body_labels_per_row": 4}

    def corrupt_2d(self, xy: Tensor, confidence: Tensor, *, level: float = 1.0, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
        level = max(0.0, min(1.0, float(level)))
        size = xy.shape[0]
        valid = confidence > 0
        count = int(self.residual_count)
        if count > 0:
            rows = torch.randint(count, (size,), device=xy.device, generator=generator)
            empirical = self.residual_bank.index_select(0, rows)
            supported = self.residual_valid.index_select(0, rows)
            empirical_confidence = self.residual_confidence.index_select(0, rows)
        else:
            empirical, supported, empirical_confidence = torch.zeros_like(xy), torch.zeros_like(valid), torch.ones_like(confidence) * 0.5
        noise = torch.randn(xy.shape, device=xy.device, dtype=xy.dtype, generator=generator) * float(self.config["corruption_noise_std"])
        displacement = torch.where(supported[:, :, None], empirical, noise)
        displacement = displacement * (0.3 + 0.7 * level) + noise * level
        _, target_scale = self.lifter.reference_frame(xy, confidence)
        displacement = displacement * target_scale[:, :, None] / self.lifter.aspect_scale
        corrupted = (xy + displacement).clamp(0, 1)
        scores = torch.where(supported, empirical_confidence, confidence * 0.5) * valid
        dropped = torch.rand(confidence.shape, device=xy.device, generator=generator) < float(self.config["corruption_joint_dropout"]) * level
        for ids in (GROUPS["left_fingers"], GROUPS["right_fingers"], list(range(52, 55)), list(range(55, 58))):
            whole_region = torch.rand((size, 1), device=xy.device, generator=generator) < float(self.config["corruption_region_dropout"]) * level
            dropped[:, ids] = dropped[:, ids] | whole_region
        scores = scores.masked_fill(dropped, 0)
        corrupted = torch.where((scores > 0)[:, :, None], corrupted, torch.zeros_like(corrupted))
        return corrupted, scores

    def loss_3d(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        xy, valid2d, target, valid3d = batch_3d(batch)
        finite_input = torch.isfinite(xy).all(-1)
        confidence = valid2d.to(xy.dtype) * finite_input
        xy = torch.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
        level = float(self.training_progress)
        noisy_xy, noisy_confidence = self.corrupt_2d(xy, confidence, level=level)
        use_noise = torch.rand((len(xy), 1), device=xy.device) < (0.25 + 0.75 * level) * float(self.config["corruption_probability"])
        inputs = torch.where(use_noise[:, :, None], noisy_xy, xy)
        input_confidence = torch.where(use_noise, noisy_confidence, confidence)
        prediction = self.lift_body(inputs, input_confidence)
        visible = (valid3d > 0) & torch.isfinite(target).all(-1)
        safe_target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        sample_loss = torch.zeros(len(xy), device=xy.device)
        active_groups = torch.zeros_like(sample_loss)
        for name, (ids, anchor) in REGIONS_3D.items():
            pred_region, target_region = prediction[:, ids], safe_target[:, ids]
            weight = visible[:, ids]
            if anchor is not None:
                pred_region = pred_region - prediction[:, anchor:anchor+1]
                target_region = target_region - safe_target[:, anchor:anchor+1]
                weight = weight & visible[:, anchor:anchor+1]
            per_joint = F.smooth_l1_loss(pred_region, target_region, reduction="none", beta=0.02).mean(-1)
            amount = weight.sum(-1)
            group_loss = (per_joint * weight).sum(-1) / amount.clamp_min(1)
            if name == "face":
                # The face's nose anchor is not a body limb: supervise its
                # global position as well as fine local facial geometry.
                group_loss = group_loss + F.smooth_l1_loss(prediction[:, 60], safe_target[:, 60], reduction="none", beta=0.02).mean(-1) * visible[:, 60]
            sample_loss += group_loss
            active_groups += (amount > 0).to(sample_loss.dtype)
        loss = (sample_loss / active_groups.clamp_min(1)).mean()
        error = (prediction - safe_target).square().sum(-1).sqrt()
        return loss, {"loss3d": loss.detach(), "mpjpe_mm": (1000 * (error * visible).sum() / visible.sum().clamp_min(1)).detach(),
                      "labeled3d": visible.sum().detach(), "corrupted_samples": use_noise.sum().detach()}

    def specification(self) -> dict[str, Any]:
        return {
            "format_version": 3, "architecture": "Pose-pretrained CSPNeXt-t + selected 73-token RTMCC head + confidence-aware residual 3D MLP",
            "config": self.config, "joint_schema": JOINT_SCHEMA,
            "input": {"name": "image", "dtype": "float32", "shape": ["batch", 3, 256, 192], "color": "RGB", "range": [0, 1],
                      "normalization": "ImageNet channel mean/std inside the model", "crop": "Single-person aspect-0.75 crop with 1.25 bbox expansion"},
            "outputs": [
                {"name": "keypoints2d", "shape": ["batch", 73, 2], "coordinates": "crop-normalized x right, y down", "joint_names": COMPACT_JOINT_NAMES, "source_coco_ids": COMPACT_COCO_IDS},
                {"name": "confidence2d", "shape": ["batch", 73], "meaning": "Minimum x/y SimCC softmax peak concentration; not calibrated visibility"},
                {"name": "keypoints3d", "shape": ["batch", 73, 3], "coordinates": "camera x right, y down, z away", "unit": "meter", "root": "midpoint of left/right hip", "joint_names": COMPACT_JOINT_NAMES},
                {"name": "geometry2d", **GEOMETRY2D_SCHEMA},
                {"name": "geometry_confidence", "shape": ["batch", 8], "meaning": "Minimum required 2D landmark concentration; not 3D uncertainty or surface confidence"},
                {"name": "geometry3d", **GEOMETRY3D_SCHEMA},
            ],
            "initial_weights": self.pretrained_provenance,
            "teacher": "Full 133-joint pose teacher is used only during training/evaluation and is excluded from model state and ONNX",
            "batching": "Independent person/camera crops; frozen pretrained BN statistics and no temporal state",
            "supervision": {"2d": "Real selected COCO-WholeBody labels plus selected-joint teacher distillation; missing labels excluded",
                            "3d": "H3WB paired whole-body 2D/3D annotations and ASPset body labels; source-specific validity masks",
                            "robustness": "Training-only RGB residuals are stored in annotated body-RMS units, transferred to each target crop's body scale, and applied with correlated errors, confidence and missing inputs before the same inference normalizer",
                            "evaluation_limit": "Clean and corrupted2D-to-3D metrics are lifting measurements. Actual RGB-to-body3D is also scored on one held-out ASPset clip using official annotated person boxes; it does not establish RGB-to-hand/foot3D or broad tracking accuracy. H3WB annotation construction includes triangulation and completion."},
            "absolute_depth": "Absolute camera distance and root translation are not predicted",
        }
