# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
# Copyright 2018-2020 Open-MMLab.
# DWPose: Copyright 2023 IDEA. All Rights Reserved.
#
# Adapted in 2026 for Wholebody-onnx: pure-PyTorch layer wrappers, a fixed
# CSPNeXt-t / RTMCC configuration, and explicit selected-joint initialization.
# See LICENSES/Apache-2.0.txt and LICENSES/DWPose-NOTICE.txt.
#
# Sources:
# https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/backbones/cspnext.py
# https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/backbones/csp_darknet.py
# https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/layers/csp_layer.py
# https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/layers/se_layer.py
# https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/heads/coord_cls_heads/rtmcc_head.py
# https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/rtmcc_block.py
# https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/transformer.py
# https://github.com/open-mmlab/mmcv/tree/main/mmcv/cnn/bricks
"""Official DWPose-t compatible pose layers without MMCV runtime dependencies.

PoseNetwork receives already standardized RGB, not raw 0..1 pixels. Its wrapper
owns image preprocessing. Outputs are the original raw SimCC logits.

Constructing a network does not initialize it from an external model.
Production bootstrap must call load_pretrained_pose; resumed workers load the
complete shared checkpoint. The initializer never substitutes random or
ImageNet-only weights when a download or strict state load fails.
"""

from collections.abc import Mapping
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmpose-t_simcc-ucoco_dw-ucoco_270e-256x192-dcf277bf_20230728.pth"
)
CHECKPOINT_SHA256 = "dcf277bf6ba4563d5ce7b1bede0f78fd5fcd518128b8f38cdf3d74e100addc51"
CONFIG_URL = (
    "https://github.com/open-mmlab/mmpose/blob/main/configs/"
    "wholebody_2d_keypoint/rtmpose/ubody/"
    "rtmpose-t_8xb64-270e_coco-ubody-wholebody-256x192.py"
)
SOURCE_JOINTS = 133


class ConvModule(nn.Module):
    """Conv/BN/SiLU with the official MMCV parameter names.

    The pose configuration supplies norm_cfg={'type': 'SyncBN'}, overriding
    the detector backbone's default norm_cfg. MMCV therefore uses eps=1e-5
    and PyTorch's default momentum=0.1. Ordinary BN has the same saved tensors.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, groups=groups, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.activate = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0):
        super().__init__()
        self.depthwise_conv = ConvModule(
            in_channels, in_channels, kernel_size, stride=stride,
            padding=padding, groups=in_channels,
        )
        self.pointwise_conv = ConvModule(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        return x * self.act(self.fc(self.global_avgpool(x)))


class CSPNeXtBlock(nn.Module):
    def __init__(self, channels, add_identity=True):
        super().__init__()
        # CSPLayer passes expansion=1.0 to each internal CSPNeXtBlock.
        self.conv1 = ConvModule(channels, channels, 3, padding=1)
        self.conv2 = DepthwiseSeparableConvModule(
            channels, channels, 5, padding=2,
        )
        self.add_identity = add_identity

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return out + x if self.add_identity else out


class CSPLayer(nn.Module):
    def __init__(self, channels, num_blocks=1, add_identity=True):
        super().__init__()
        mid_channels = int(channels * 0.5)
        self.main_conv = ConvModule(channels, mid_channels, 1)
        self.short_conv = ConvModule(channels, mid_channels, 1)
        self.final_conv = ConvModule(2 * mid_channels, channels, 1)
        self.blocks = nn.Sequential(*[
            CSPNeXtBlock(mid_channels, add_identity)
            for _ in range(num_blocks)
        ])
        self.channel_attention = True
        self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x):
        short = self.short_conv(x)
        main = self.blocks(self.main_conv(x))
        return self.final_conv(self.attention(torch.cat((main, short), dim=1)))


class SPPBottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        mid_channels = channels // 2
        self.conv1 = ConvModule(channels, mid_channels, 1)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(size, stride=1, padding=size // 2)
            for size in (5, 9, 13)
        ])
        self.conv2 = ConvModule(mid_channels * 4, channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [pool(x) for pool in self.poolings],
                                    dim=1))


class CSPNeXtTiny(nn.Module):
    """P5, width .375, depth .167, channel attention, output stage 4."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvModule(3, 12, 3, stride=2, padding=1),
            ConvModule(12, 12, 3, padding=1),
            ConvModule(12, 24, 3, padding=1),
        )
        settings = (
            (24, 48, 3, True, False),
            (48, 96, 6, True, False),
            (96, 192, 6, True, False),
            (192, 384, 3, False, True),
        )
        self.layers = ["stem"]
        self.out_channels = 384
        self.out_indices = (4,)
        for stage_id, (cin, cout, depth, identity, use_spp) in enumerate(
                settings, start=1):
            modules = [ConvModule(cin, cout, 3, stride=2, padding=1)]
            if use_spp:
                modules.append(SPPBottleneck(cout))
            modules.append(CSPLayer(
                cout, num_blocks=max(round(depth * 0.167), 1),
                add_identity=identity,
            ))
            name = f"stage{stage_id}"
            self.add_module(name, nn.Sequential(*modules))
            self.layers.append(name)

    def forward(self, x):
        for name in self.layers:
            x = getattr(self, name)(x)
        return x


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class Scale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.scale


class RTMCCBlock(nn.Module):
    """The exact self-attention GAU configuration used by the tiny checkpoint."""

    def __init__(self, num_token):
        super().__init__()
        self.num_token = num_token
        self.s = 128
        self.e = 512
        self.use_rel_bias = False
        self.pos_enc = False
        self.attn_type = "self-attn"
        self.o = nn.Linear(self.e, 256, bias=False)
        self.uv = nn.Linear(256, 2 * self.e + self.s, bias=False)
        self.gamma = nn.Parameter(torch.rand(2, self.s))
        self.beta = nn.Parameter(torch.rand(2, self.s))
        self.ln = ScaleNorm(256)
        self.act_fn = nn.SiLU(inplace=True)
        self.res_scale = Scale(256)
        self.drop_path = nn.Identity()
        self.sqrt_s = math.sqrt(self.s)

    def forward(self, x):
        uv = self.act_fn(self.uv(self.ln(x)))
        u, v, base = torch.split(uv, (self.e, self.e, self.s), dim=2)
        base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta
        q, k = torch.unbind(base, dim=2)
        qk = torch.bmm(q, k.transpose(1, 2))
        kernel = torch.square(F.relu(qk / self.sqrt_s))
        main = self.o(u * torch.bmm(kernel, v))
        return self.res_scale(x) + self.drop_path(main)


class RTMCCHead(nn.Module):
    def __init__(self, num_joints):
        super().__init__()
        self.in_channels = 384
        self.out_channels = num_joints
        self.input_size = (192, 256)
        self.in_featuremap_size = (6, 8)
        self.simcc_split_ratio = 2.0
        self.final_layer = nn.Conv2d(384, num_joints, 7, padding=3, bias=True)
        self.mlp = nn.Sequential(
            ScaleNorm(48),
            nn.Linear(48, 256, bias=False),
        )
        self.gau = RTMCCBlock(num_joints)
        self.cls_x = nn.Linear(256, 384, bias=False)
        self.cls_y = nn.Linear(256, 512, bias=False)

    def forward(self, features):
        tokens = self.final_layer(features).flatten(2)
        tokens = self.gau(self.mlp(tokens))
        return self.cls_x(tokens), self.cls_y(tokens)


class PoseNetwork(nn.Module):
    def __init__(self, num_joints=SOURCE_JOINTS):
        super().__init__()
        if not isinstance(num_joints, int) or not 1 <= num_joints <= SOURCE_JOINTS:
            raise ValueError("num_joints must be an integer from 1 to 133")
        self.num_joints = num_joints
        self.backbone = CSPNeXtTiny()
        self.head = RTMCCHead(num_joints)

    def forward(self, rgb_normalized):
        return self.head(self.backbone(rgb_normalized))


def _numpy_metadata_globals():
    """Allow only ordinary NumPy metadata while retaining weights_only=True."""
    multiarray = np._core.multiarray
    allowed = [
        np.ndarray,
        np.dtype,
        multiarray._reconstruct,
        multiarray.scalar,
        (multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        (multiarray.scalar, "numpy.core.multiarray.scalar"),
    ]
    for dtype in ("bool", "int8", "uint8", "int16", "uint16", "int32",
                  "uint32", "int64", "uint64", "float16", "float32",
                  "float64", "complex64", "complex128", "S", "U"):
        allowed.append(type(np.dtype(dtype)))
    return allowed


def _checkpoint_state(payload):
    if not isinstance(payload, Mapping):
        raise TypeError("Pretrained checkpoint must contain a state dictionary")
    if isinstance(payload.get("state_dict"), Mapping):
        source = payload["state_dict"]
    elif isinstance(payload.get("model"), Mapping):
        source = payload["model"]
    else:
        source = payload

    result = {}
    prefixes = ("module.", "student.")
    for original_key, value in source.items():
        if not isinstance(original_key, str) or not isinstance(value, torch.Tensor):
            raise TypeError("Pretrained state entries must be named tensors")
        key = original_key
        while True:
            matched = next((prefix for prefix in prefixes
                            if key.startswith(prefix)), None)
            if matched is None:
                break
            key = key[len(matched):]
        if key in result:
            raise ValueError(f"Duplicate pretrained parameter after prefix removal: {key}")
        result[key] = value
    return result


def load_pretrained_pose(model: PoseNetwork, coco_ids: list | None = None) -> dict:
    """Download the official pose checkpoint and load every model tensor.

    Only the final keypoint convolution is indexed when a subset is requested.
    Missing/unexpected tensors or shape mismatches are fatal initialization
    errors; they are never hidden by strict=False or a random-weight fallback.
    The reported SHA-256 is computed from the exact cached/downloaded bytes.
    """
    if not isinstance(model, PoseNetwork):
        raise TypeError("model must be a PoseNetwork")

    selected = list(range(SOURCE_JOINTS)) if coco_ids is None else list(coco_ids)
    if (len(selected) != model.num_joints
            or len(set(selected)) != len(selected)
            or any(not isinstance(index, int) or not 0 <= index < SOURCE_JOINTS
                   for index in selected)):
        raise ValueError("coco_ids must uniquely identify every output joint")

    checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / CHECKPOINT_URL.rsplit("/", 1)[-1]
    if not checkpoint_path.is_file():
        torch.hub.download_url_to_file(
            CHECKPOINT_URL, str(checkpoint_path),
            hash_prefix=CHECKPOINT_SHA256, progress=True,
        )

    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != CHECKPOINT_SHA256:
        raise ValueError(
            "Official pose checkpoint SHA-256 mismatch: "
            f"expected {CHECKPOINT_SHA256}, received {actual_sha256}"
        )

    with torch.serialization.safe_globals(_numpy_metadata_globals()):
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True,
        )
    state = _checkpoint_state(payload)
    original_weight = state.get("head.final_layer.weight")
    original_bias = state.get("head.final_layer.bias")
    if (original_weight is None or original_bias is None
            or tuple(original_weight.shape) != (SOURCE_JOINTS, 384, 7, 7)
            or tuple(original_bias.shape) != (SOURCE_JOINTS,)):
        raise ValueError("Checkpoint is not the configured 133-point DWPose-t")

    if coco_ids is not None:
        indices = torch.tensor(selected, dtype=torch.long)
        state["head.final_layer.weight"] = original_weight.index_select(0, indices)
        state["head.final_layer.bias"] = original_bias.index_select(0, indices)
    model.load_state_dict(state, strict=True)

    return {
        "url": CHECKPOINT_URL,
        "checkpoint_url": CHECKPOINT_URL,
        "sha256": actual_sha256,
        "expected_sha256": CHECKPOINT_SHA256,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "configuration_url": CONFIG_URL,
        "architecture": "CSPNeXt-t P5 + RTMCC SimCC, 256x192",
        "source_point_count": SOURCE_JOINTS,
        "output_point_count": model.num_joints,
        "selected_coco_ids": selected,
        "selection_applied": coco_ids is not None,
        "strict_state_load": True,
        "loaded_state_tensor_count": len(state),
        "license": "Apache-2.0 (upstream DWPose parameters and implementation)",
        "license_source": "https://huggingface.co/yzd-v/DWPose/blob/main/README.md",
        "adaptation": (
            "Final keypoint convolution selected; shared MLP, GAU and SimCC "
            "classifiers retain source weights. Removing tokens changes GAU "
            "context, so unchanged predictions or accuracy are not assumed."
        ),
    }
