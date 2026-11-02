"""WholeBody heatmaps and supervised, hip-relative 3D lifting.

RGB inputs are person crops in [0, 1]. ImageNet normalization is internal.
The backbone's BatchNorm statistics never change on an individual worker.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


BODY_2D_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
FOOT_NAMES = [
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
]
HAND_NAMES = ["wrist"] + [
    f"{finger}_{joint}" for finger in ("thumb", "index", "middle", "ring", "pinky")
    for joint in range(1, 5)
]
WHOLEBODY_NAMES = (
    BODY_2D_NAMES + FOOT_NAMES + [f"face_{i:02d}" for i in range(68)]
    + [f"left_hand_{name}" for name in HAND_NAMES]
    + [f"right_hand_{name}" for name in HAND_NAMES]
)
BODY_3D_COCO_IDS = list(range(5, 17))
BODY_3D_NAMES = [BODY_2D_NAMES[i] for i in BODY_3D_COCO_IDS]
BACKBONE_WEIGHTS_URL = "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"
DEFAULT_CONFIG: dict[str, Any] = {
    "image_height": 192,
    "image_width": 144,
    "head_channels": 64,
    "lifter_width": 256,
    "freeze_backbone": True,
    "heatmap_sigma": 1.5,
    "backbone_weights": "none",
}


def _conv_norm(in_channels: int, out_channels: int, kernel: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2, bias=False),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(),
    )


class ResidualLinear(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width), nn.LayerNorm(width), nn.ReLU(),
            nn.Linear(width, width), nn.LayerNorm(width), nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layers(x)


class BodyLifter(nn.Module):
    """Twelve body joints; no invented 3D labels for faces, fingers, or toes."""

    def __init__(self, width: int, aspect: float) -> None:
        super().__init__()
        self.register_buffer("aspect_scale", torch.tensor([aspect, 1.0]).view(1, 1, 2))
        self.input_layer = nn.Sequential(nn.Linear(24, width), nn.LayerNorm(width), nn.ReLU())
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
        if height % 4 or width % 4 or channels % 8:
            raise ValueError("Image dimensions must be divisible by 4; head_channels by 8")
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights).features
        if pretrained:
            self.config["backbone_weights"] = "MobileNet_V3_Small_Weights.IMAGENET1K_V1"
        self.freeze_backbone = bool(self.config["freeze_backbone"])
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
        self.backbone.eval()
        # Official torchvision feature indices: strides 4, 8, 16, 32.
        self.laterals = nn.ModuleList([_conv_norm(c, channels) for c in (16, 24, 48, 576)])
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(8, channels), nn.SiLU(),
            _conv_norm(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(8, channels), nn.SiLU(),
            _conv_norm(channels, channels),
        )
        self.heatmap_head = nn.Conv2d(channels, 133, 1)
        nn.init.normal_(self.heatmap_head.weight, std=0.01)
        nn.init.zeros_(self.heatmap_head.bias)
        self.lifter = BodyLifter(int(self.config["lifter_width"]), width / height)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.heatmap_height, self.heatmap_width = height // 4, width // 4
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, self.heatmap_height),
            torch.linspace(0.0, 1.0, self.heatmap_width), indexing="ij",
        )
        self.register_buffer("coordinate_grid", torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2))
        self.register_buffer("heatmap_scale", torch.tensor([self.heatmap_width - 1, self.heatmap_height - 1]))
        self.register_buffer("body_ids", torch.tensor(BODY_3D_COCO_IDS, dtype=torch.long))

    def train(self, mode: bool = True) -> "WholeBodyModel":
        super().train(mode)
        # Even when fine-tuning weights, worker-local running statistics must not diverge.
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward_heatmaps(self, rgb: Tensor) -> Tensor:
        x = (rgb - self.image_mean) / self.image_std
        features: list[Tensor] = []
        for index, layer in enumerate(self.backbone):
            x = layer(x)
            if index in (1, 3, 8, 12):
                features.append(x)
        pyramid = self.laterals[3](features[3])
        for index in (2, 1, 0):
            lateral = self.laterals[index](features[index])
            pyramid = lateral + F.interpolate(pyramid, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
        return self.heatmap_head(self.refine(pyramid))

    def decode_heatmaps(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        probability = logits.flatten(2).softmax(dim=-1)
        keypoints = torch.matmul(probability, self.coordinate_grid)
        # Peak probability is a concentration score, not calibrated visibility.
        confidence = probability.max(dim=-1).values
        return keypoints, confidence

    def lift_body(self, crop_xy: Tensor) -> Tensor:
        return self.lifter(crop_xy)

    def forward(self, rgb: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        points, confidence = self.decode_heatmaps(self.forward_heatmaps(rgb))
        return points, confidence, self.lift_body(points.index_select(1, self.body_ids))

    def loss_2d(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        logits = self.forward_heatmaps(batch["image"])
        target = batch["keypoints"]
        visible = batch["visibility"] > 0
        visible = visible & torch.isfinite(target).all(dim=-1) & (target >= 0).all(dim=-1) & (target <= 1).all(dim=-1)
        safe_target = torch.nan_to_num(target).clamp(0, 1)
        distance = (self.coordinate_grid[None, None, :, :] - safe_target[:, :, None, :]) * self.heatmap_scale
        target_distribution = torch.exp(-distance.square().sum(dim=-1) / (2 * float(self.config["heatmap_sigma"]) ** 2))
        target_distribution = target_distribution / target_distribution.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        cross_entropy = -(target_distribution * logits.flatten(2).log_softmax(dim=-1)).sum(dim=-1)
        weight = visible.to(cross_entropy.dtype)
        count = weight.sum(dim=-1).clamp_min(1)
        per_sample = (cross_entropy * weight).sum(dim=-1) / count
        points, _ = self.decode_heatmaps(logits)
        distance_xy = (points - safe_target).square().sum(dim=-1).sqrt()
        coordinate_loss = ((points - safe_target).abs().sum(dim=-1) * weight).sum(dim=-1) / count
        loss = (per_sample + coordinate_loss).mean()
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
            "format_version": 1,
            "architecture": "MobileNetV3-small feature pyramid + 133 heatmaps + supervised 12-joint 3D lifter",
            "config": self.config,
            "input": {"name": "image", "dtype": "float32", "shape": ["batch", 3, self.config["image_height"], self.config["image_width"]],
                      "color": "RGB", "range": [0, 1], "normalization": "ImageNet mean/std are applied inside the model",
                      "crop": f"Single-person crop, width/height={self.config['image_width'] / self.config['image_height']}; resize without changing aspect ratio"},
            "outputs": [
                {"name": "keypoints2d", "shape": ["batch", 133, 2], "coordinates": "x right, y down; normalized to crop [0,1]",
                 "joint_names": WHOLEBODY_NAMES},
                {"name": "confidence2d", "shape": ["batch", 133], "meaning": "Maximum spatial softmax probability; concentration score, not calibrated visibility"},
                {"name": "body3d", "shape": ["batch", 12, 3], "unit": "meter", "coordinates": "camera x right, y down, z away",
                 "root": "midpoint of left_hip and right_hip", "joint_names": BODY_3D_NAMES,
                 "source_coco_ids": BODY_3D_COCO_IDS,
                 "scope": "Relative articulated body pose only; absolute camera distance and root translation are not predicted"},
            ],
            "initial_weights": {
                "name": self.config["backbone_weights"],
                "url": BACKBONE_WEIGHTS_URL if self.config["backbone_weights"] != "none" else None,
                "code_license": "torchvision BSD-3-Clause",
                "terms": "Pretrained weights have separate data-derived terms; see THIRD_PARTY.md and MODEL_LICENSE.md",
            },
            "supervision": {"2d": "COCO-WholeBody 133 joints; absent labels excluded",
                            "3d": "ASPset-510 real 3D joints projected through calibrated cameras; only 12 anatomically matching joints",
                            "missing3d": "No synthetic 3D supervision for face, fingers, toe tips, or heels"},
        }
