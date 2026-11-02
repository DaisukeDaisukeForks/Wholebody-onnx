"""Compact body, finger, foot and face localization with supervised 3D lifting.

RGB inputs are person crops in [0, 1]. ImageNet normalization is internal.
The backbone's BatchNorm statistics never change on an individual worker.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from joint_schema import (
    BODY_3D_COCO_IDS, BODY_3D_NAMES, COMPACT_COCO_IDS,
    COMPACT_JOINT_NAMES, GEOMETRY2D_NAMES, GEOMETRY2D_SCHEMA,
    GEOMETRY2D_WEIGHTS, JOINT_SCHEMA,
)
BACKBONE_WEIGHTS_URL = "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"
DEFAULT_CONFIG: dict[str, Any] = {
    "image_height": 192,
    "image_width": 144,
    "head_channels": 32,
    "lifter_width": 128,
    "freeze_backbone": True,
    "heatmap_sigma": 0.75,
    "backbone_weights": "none",
}


def _conv_relu(in_channels: int, out_channels: int, kernel: int = 1, groups: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2, groups=groups, bias=True),
        nn.ReLU(),
    )


class ResidualLinear(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layers(x)


class BodyLifter(nn.Module):
    """Twelve body joints; no invented 3D labels for faces, fingers, or toes."""

    def __init__(self, width: int, aspect: float) -> None:
        super().__init__()
        self.register_buffer("aspect_scale", torch.tensor([aspect, 1.0]).view(1, 1, 2))
        self.input_layer = nn.Sequential(nn.Linear(24, width), nn.ReLU())
        self.blocks = nn.Sequential(ResidualLinear(width), ResidualLinear(width))
        self.output_layer = nn.Linear(width, 36)

    def forward(self, crop_xy: Tensor) -> Tensor:
        xy = crop_xy * self.aspect_scale
        root = (xy[:, 6:7] + xy[:, 7:8]) * 0.5
        centered = xy - root
        scale = centered.square().sum(dim=-1).mean(dim=-1, keepdim=True).sqrt()
        normalized = centered / scale.clamp_min(1e-4).unsqueeze(-1)
        prediction = self.output_layer(self.blocks(self.input_layer(normalized.flatten(1))))
        prediction = prediction.reshape(-1, 12, 3)
        return prediction - (prediction[:, 6:7] + prediction[:, 7:8]) * 0.5


class WholeBodyModel(nn.Module):
    def __init__(self, config: dict[str, Any] | None = None, pretrained: bool = False) -> None:
        super().__init__()
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        height, width = int(self.config["image_height"]), int(self.config["image_width"])
        channels = int(self.config["head_channels"])
        if height % 8 or width % 8 or channels <= 0:
            raise ValueError("Image dimensions must be divisible by 8; head_channels must be positive")
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        # Keep the pretrained stride-8 and stride-16 features. Later classification
        # blocks and the 576-channel final expansion are not part of this model.
        self.backbone = mobilenet_v3_small(weights=weights).features[:9]
        if pretrained:
            self.config["backbone_weights"] = "MobileNet_V3_Small_Weights.IMAGENET1K_V1"
        self.freeze_backbone = bool(self.config["freeze_backbone"])
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.low_projection = _conv_relu(24, channels)
        self.high_projection = _conv_relu(48, channels)
        self.refine = nn.Sequential(
            _conv_relu(channels, channels, 3, groups=channels),
            _conv_relu(channels, channels),
        )
        self.joint_count = len(COMPACT_COCO_IDS)
        # One shared convolution produces one location score and two subcell
        # offsets per joint. Finger tips remain independent supervised outputs.
        self.pose_head = nn.Conv2d(channels, self.joint_count * 3, 1)
        nn.init.normal_(self.pose_head.weight, std=0.01)
        nn.init.zeros_(self.pose_head.bias)
        self.lifter = BodyLifter(int(self.config["lifter_width"]), width / height)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.heatmap_height, self.heatmap_width = height // 8, width // 8
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, self.heatmap_height),
            torch.linspace(0.0, 1.0, self.heatmap_width), indexing="ij",
        )
        self.register_buffer("coordinate_grid", torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2))
        self.register_buffer("heatmap_scale", torch.tensor([self.heatmap_width - 1, self.heatmap_height - 1]))
        self.register_buffer("body_ids", torch.arange(12, dtype=torch.long))
        self.register_buffer("geometry_weights", torch.tensor(GEOMETRY2D_WEIGHTS, dtype=torch.float32))
        self.register_buffer("geometry_source_mask", self.geometry_weights != 0)
        self.register_buffer("geometry_scale", torch.tensor(
            [row["coordinate_scale"] for row in GEOMETRY2D_SCHEMA["rows"]], dtype=torch.float32,
        ).unsqueeze(0))

    def train(self, mode: bool = True) -> "WholeBodyModel":
        super().train(mode)
        # Even when fine-tuning weights, worker-local running statistics must not diverge.
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward_pose(self, rgb: Tensor) -> tuple[Tensor, Tensor]:
        x = (rgb - self.image_mean) / self.image_std
        low_feature = x
        for index, layer in enumerate(self.backbone):
            x = layer(x)
            if index == 3:
                low_feature = x
        low = self.low_projection(low_feature)
        high = self.high_projection(x)
        fused = low + F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        raw = self.pose_head(self.refine(fused))
        logits = raw[:, :self.joint_count]
        offsets = raw[:, self.joint_count:].reshape(-1, self.joint_count, 2, self.heatmap_height, self.heatmap_width).tanh()
        return logits, offsets

    def forward_heatmaps(self, rgb: Tensor) -> Tensor:
        return self.forward_pose(rgb)[0]

    def decode_heatmaps(self, logits: Tensor, offsets: Tensor) -> tuple[Tensor, Tensor]:
        probability = logits.flatten(2).softmax(dim=-1)
        # Peak probability is a concentration score, not calibrated visibility.
        confidence, indices = probability.max(dim=-1)
        base = self.coordinate_grid[indices]
        selected = offsets.flatten(3).gather(3, indices[:, :, None, None].expand(-1, -1, 2, 1)).squeeze(-1)
        points = (base + selected / self.heatmap_scale).clamp(0.0, 1.0)
        return points, confidence

    def lift_body(self, crop_xy: Tensor) -> Tensor:
        return self.lifter(crop_xy)

    def derived_geometry(self, points: Tensor, confidence: Tensor) -> tuple[Tensor, Tensor]:
        # Fixed source weights live in joint_schema.py, shared with documentation.
        geometry = torch.matmul(self.geometry_weights, points) * self.geometry_scale
        source_scores = torch.where(self.geometry_source_mask[None, :, :], confidence[:, None, :], torch.ones_like(confidence[:, None, :]))
        return geometry, source_scores.amin(dim=-1)

    def forward(self, rgb: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        points, confidence = self.decode_heatmaps(*self.forward_pose(rgb))
        geometry, geometry_confidence = self.derived_geometry(points, confidence)
        return points, confidence, self.lift_body(points.index_select(1, self.body_ids)), geometry, geometry_confidence

    def loss_2d(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        logits, offsets = self.forward_pose(batch["image"])
        target = batch["keypoints"]
        visible = batch["visibility"] > 0
        visible = visible & torch.isfinite(target).all(dim=-1) & (target >= 0).all(dim=-1) & (target <= 1).all(dim=-1)
        safe_target = torch.nan_to_num(target).clamp(0, 1)
        distance = (self.coordinate_grid[None, None, :, :] - safe_target[:, :, None, :]) * self.heatmap_scale
        target_distribution = torch.exp(-distance.square().sum(dim=-1) / (2 * float(self.config["heatmap_sigma"]) ** 2))
        target_distribution = target_distribution / target_distribution.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        cross_entropy = -(target_distribution * logits.flatten(2).log_softmax(dim=-1)).sum(dim=-1)
        # Local offsets receive real coordinate supervision around each target.
        # This retains subcell precision with a small stride-8 score tensor.
        local_weight = target_distribution * (distance.abs().amax(dim=-1) <= 1.0).to(target_distribution.dtype)
        local_weight = local_weight / local_weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        offset_target = (-distance).clamp(-1, 1).permute(0, 1, 3, 2)
        offset_error = F.smooth_l1_loss(offsets.flatten(3), offset_target, reduction="none", beta=0.2).mean(dim=2)
        offset_loss = (offset_error * local_weight).sum(dim=-1)
        weight = visible.to(cross_entropy.dtype)
        count = weight.sum(dim=-1).clamp_min(1)
        per_sample = ((cross_entropy + offset_loss) * weight).sum(dim=-1) / count
        points, _ = self.decode_heatmaps(logits, offsets)
        distance_xy = (points - safe_target).square().sum(dim=-1).sqrt()
        loss = per_sample.mean()
        return loss, {
            "loss2d": loss.detach(),
            "mean_xy_error": ((distance_xy * weight).sum() / weight.sum().clamp_min(1)).detach(),
            "labeled2d": weight.sum().detach(),
        }

    def loss_3d(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        target = batch["target3d"]
        input_valid = torch.isfinite(batch["keypoints"]).all(dim=-1).all(dim=-1, keepdim=True)
        safe_input = torch.nan_to_num(batch["keypoints"], nan=0.0, posinf=0.0, neginf=0.0)
        safe_target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        prediction = self.lift_body(safe_input)
        visible = (batch["visibility"] > 0) & torch.isfinite(target).all(dim=-1) & input_valid
        weight = visible.to(prediction.dtype)
        difference = prediction - safe_target
        per_joint = F.smooth_l1_loss(prediction, safe_target, reduction="none", beta=0.1).mean(dim=-1)
        per_sample = (per_joint * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1)
        loss = per_sample.mean()
        return loss, {
            "loss3d": loss.detach(),
            "mpjpe_mm": (1000 * (difference.square().sum(dim=-1).sqrt() * weight).sum() / weight.sum().clamp_min(1)).detach(),
            "labeled3d": weight.sum().detach(),
        }

    def specification(self) -> dict[str, Any]:
        return {
            "format_version": 2,
            "architecture": "Truncated MobileNetV3-small blocks 0..8, shared stride-8 compact location/offset head, supervised 12-joint 3D lifter",
            "config": self.config,
            "input": {"name": "image", "dtype": "float32", "shape": ["batch", 3, self.config["image_height"], self.config["image_width"]],
                      "color": "RGB", "range": [0, 1], "normalization": "ImageNet mean/std are applied inside the model",
                      "crop": f"Single-person crop, width/height={self.config['image_width'] / self.config['image_height']}; resize without changing aspect ratio"},
            "outputs": [
                {"name": "keypoints2d", "shape": ["batch", self.joint_count, 2], "coordinates": "x right, y down; normalized to crop [0,1]",
                 "joint_names": COMPACT_JOINT_NAMES, "source_coco_ids": COMPACT_COCO_IDS},
                {"name": "confidence2d", "shape": ["batch", self.joint_count], "meaning": "Maximum spatial softmax probability; concentration score, not calibrated visibility"},
                {"name": "body3d", "shape": ["batch", 12, 3], "unit": "meter", "coordinates": "camera x right, y down, z away",
                 "root": "midpoint of left_hip and right_hip", "joint_names": BODY_3D_NAMES,
                 "source_coco_ids": BODY_3D_COCO_IDS,
                 "scope": "Relative articulated body pose only; absolute camera distance and root translation are not predicted"},
                {"name": "geometry2d", "shape": ["batch", 8, 2], "names": GEOMETRY2D_NAMES,
                 "definitions": GEOMETRY2D_SCHEMA,
                 "scope": "Derived hand reference centers and hand/foot axes; no independently labeled hand-back surface, sole surface, or 3D sole normal"},
                {"name": "geometry_confidence", "shape": ["batch", 8],
                 "meaning": "Minimum concentration score of the joints used in each derived quantity; not calibrated visibility. Geometry requires reliable source joints."},
            ],
            "initial_weights": {
                "name": self.config["backbone_weights"],
                "url": BACKBONE_WEIGHTS_URL if self.config["backbone_weights"] != "none" else None,
                "code_license": "torchvision BSD-3-Clause",
                "terms": "Pretrained weights have separate data-derived terms; see THIRD_PARTY.md and MODEL_LICENSE.md",
            },
            "joint_schema": JOINT_SCHEMA,
            "batching": "Each batch entry is an independent person/camera crop; no cross-entry normalization or temporal state",
            "supervision": {"2d": "Selected 73 COCO-WholeBody joints; absent labels excluded",
                            "3d": "ASPset-510 real 3D joints projected through calibrated cameras; only 12 anatomically matching joints",
                            "missing3d": "No synthetic 3D supervision for face, fingers, toe tips, or heels"},
        }
