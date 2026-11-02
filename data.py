"""Real COCO-WholeBody, ASPset-510 and H3WB training annotations.

The compact prepared files are safe to bundle separately from downloads/ and
aspset_raw/. Images remain on the runner, with their original attribution recorded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

from joint_schema import COMPACT_COCO_IDS, COMPACT_JOINT_NAMES, JOINT_SCHEMA

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 192
CROP_EXPANSION = 1.25
COCO_BODY_IDS = tuple(range(5, 17))
BODY_NAMES = (
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
COCO_SOURCE = "https://github.com/jin-s13/COCO-WholeBody"
COCO_DISTRIBUTION = "https://huggingface.co/datasets/msdkhairi/coco2017"
ASPSOURCE = "https://github.com/anibali/aspset-510"
ASPBASE = "https://archive.org/download/aspset510"
ASP_ARCHIVES = {
    "aspset510_v1_common-splits.tar.gz": "ca539550f46762741d8239ed4ac16b8e",
    "aspset510_v1_trainval-cameras.tar.gz": "1dca82253524fa6f379616851c8e9356",
    "aspset510_v1_trainval-joints_3d.tar.gz": "f848dc688c3d8536063515fff5349b98",
}
ASP_RGB_VIDEO_URL = f"{ASPBASE}/aspset510_v1_test-videos.tar.gz"
ASP_RGB_VIDEO_MEMBER = "ASPset-510/test/videos/1e28/1e28-0001-left.mkv"
ASP_RGB_VIDEO_SHA256 = "4dba3cd1ba096390da15a5afbfea9aef23b51b57ca1460ec08e21a3b6c1c55ed"
ASP_RGB_ARCHIVES = {
    "aspset510_v1_common-splits.tar.gz": "ca539550f46762741d8239ed4ac16b8e",
    "aspset510_v1_test-cameras.tar.gz": "b93bd9d41206172361cec8ae0d3434d5",
    "aspset510_v1_test-joints_3d.tar.gz": "cc725af311f45d6b203cce05c5b94208",
    "aspset510_v1_test-boxes.tar.gz": "006c7f709e3e6b5fe820a5e0aa560dd7",
}
H3WB_SOURCE = "https://github.com/wholebody3d/wholebody3d"
H3WB_FILE_ID = "1LZh4Jsg3_ZKBF0iEPiexzoGHE4srLgfC"
H3WB_DISTRIBUTION = f"https://drive.google.com/file/d/{H3WB_FILE_ID}/view"
H3WB_SOURCE_SHA256 = "d423224a9e3c47aa8320687a8fcb58e732aa8d3aabe589e01ce390030d123b1d"
H3WB_SUBJECTS = {"train": ("S1", "S5", "S6"), "val": ("S7",)}
H3WB_CAMERAS = ("54138969", "55011271", "58860488", "60457274")
LEGACY_PREPARED_FILES = (
    "coco_train.json", "coco_val.json", "aspset_train.npz", "aspset_val.npz",
    "dataset_manifest.json",
)
PREPARED_FILES = (
    "coco_train.json", "coco_val.json", "aspset_train.npz", "aspset_val.npz",
    "h3wb_train.npz", "h3wb_val.npz",
    "dataset_manifest.json",
)


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    temporary.replace(path)


def _write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _cached_preparation(root, name, manifest_key):
    """Prepared bundles carry source provenance inside dataset_manifest.json."""
    path = Path(root) / f"{name}_preparation.json"
    if path.exists():
        return _read_json(path)
    manifest_path = Path(root) / "dataset_manifest.json"
    if manifest_path.exists():
        return _read_json(manifest_path).get(manifest_key, {})
    return {}


def _digest(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, destination, *, md5=None, attempts=4):
    """Download atomically; failures never turn an HTML error into a cached file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        if md5 is None or _digest(destination, "md5") == md5:
            return destination
    temporary = destination.with_name(destination.name + ".part")
    last_error = None
    for attempt in range(attempts):
        try:
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", "").lower():
                    raise ValueError(f"HTML response instead of data from {url}")
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if not temporary.stat().st_size:
                raise ValueError(f"Empty download from {url}")
            if md5 and _digest(temporary, "md5") != md5:
                raise ValueError(f"Upstream archive checksum mismatch: {destination.name}")
            temporary.replace(destination)
            return destination
        except (requests.RequestException, OSError, ValueError) as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Could not download {url}: {last_error}") from last_error


def _annotation_path(root, split):
    filename = f"coco_wholebody_{split}_v1.0.json"
    path = Path(root) / "downloads" / filename
    url = f"{COCO_DISTRIBUTION}/resolve/main/{filename}"
    download(url, path)
    return path, url


def _wholebody_annotation(annotation):
    parts = []
    for key, count, validity_key in (
        ("keypoints", 17, None), ("foot_kpts", 6, "foot_valid"),
        ("face_kpts", 68, "face_valid"),
        ("lefthand_kpts", 21, "lefthand_valid"),
        ("righthand_kpts", 21, "righthand_valid"),
    ):
        values = annotation.get(key, [0.0] * (count * 3))
        if len(values) != count * 3:
            raise ValueError(f"Malformed {key} in annotation {annotation.get('id')}")
        points = np.asarray(values, dtype=np.float32).reshape(count, 3)
        if validity_key and not annotation.get(validity_key, False):
            points[:, 2] = 0
        finite = np.isfinite(points).all(axis=1)
        points[~finite] = 0
        parts.append(points)
    return np.concatenate(parts, axis=0)


def _coco_records(path):
    """Read each COCO array in one pass, independent of top-level key order."""
    import ijson
    from ijson.common import ObjectBuilder

    groups = {"images", "annotations", "licenses"}
    finished = set()
    builder = None
    active_group = None
    with Path(path).open("rb") as handle:
        for prefix, event, value in ijson.parse(handle, use_float=True):
            if builder is None and event == "start_map" and prefix in {
                "images.item", "annotations.item", "licenses.item"
            }:
                active_group = prefix.split(".", 1)[0]
                builder = ObjectBuilder()
            if builder is not None:
                builder.event(event, value)
                if prefix == f"{active_group}.item" and event == "end_map":
                    yield active_group, builder.value
                    builder, active_group = None, None
            if prefix in groups and event == "end_array":
                finished.add(prefix)
                if finished == groups:
                    return


def prepare_coco(root, split, image_limit, seed=20261102):
    """Select real person annotations while streaming the large source JSON."""
    if split not in ("train", "val") or image_limit < 1:
        raise ValueError("COCO split must be train/val and image_limit positive")
    root = Path(root)
    output = root / f"coco_{split}.json"
    if output.exists():
        cached = _read_json(output)
        if (cached.get("format") == 2 and cached.get("requested_image_limit") == image_limit
                and cached.get("seed") == seed):
            return cached
    path, source_url = _annotation_path(root, split)
    images, licenses = {}, []
    rows_by_image = {}
    priorities, heap = {}, []
    for group, record in _coco_records(path):
        if group == "images":
            images[int(record["id"])] = record
            continue
        if group == "licenses":
            licenses.append(record)
            continue
        annotation = record
        image_id = int(annotation["image_id"])
        if annotation.get("iscrowd", 0):
            continue
        new_image = image_id not in rows_by_image
        if new_image:
            priority = int.from_bytes(hashlib.sha256(f"{seed}:{split}:{image_id}".encode()).digest()[:8], "big")
            heap_entry = (-priority, -image_id)
            if len(heap) >= image_limit and heap_entry <= heap[0]:
                continue
        box = [float(value) for value in annotation.get("bbox", [])]
        if len(box) != 4 or min(box[2:]) < 24 or not np.isfinite(box).all():
            continue
        points = _wholebody_annotation(annotation)
        if np.count_nonzero(points[:17, 2] > 0) < 6:
            continue
        if new_image:
            if len(heap) >= image_limit:
                _, removed_id = heapq.heappop(heap)
                del rows_by_image[-removed_id]
                del priorities[-removed_id]
            heapq.heappush(heap, heap_entry)
            priorities[image_id] = priority
            rows_by_image[image_id] = []
        rows_by_image[image_id].append({
            "annotation_id": int(annotation["id"]), "image_id": image_id,
            "bbox": box, "keypoints": points.tolist(),
        })
    # The worst retained priority only decreases, so a final selected image has
    # been retained since its first eligible person; none of its labels are lost.
    selected_ids = sorted(rows_by_image, key=lambda image_id: (priorities[image_id], image_id))
    if not selected_ids:
        raise RuntimeError(f"No usable COCO-WholeBody persons in {source_url}")
    missing_images = set(selected_ids) - images.keys()
    if missing_images:
        raise ValueError(f"COCO annotations refer to missing image metadata: {sorted(missing_images)[:5]}")
    rows = []
    image_records = {}
    for image_id in selected_ids:
        original = images[image_id]
        file_name = f"{image_id:012d}.jpg"
        # Image IDs, annotation IDs and source split stay paired throughout training.
        declared = Path(str(original.get("file_name", file_name))).name
        if declared != file_name:
            raise ValueError(f"COCO image filename/id mismatch: {declared} / {image_id}")
        image_records[str(image_id)] = {
            "image_id": image_id, "file_name": file_name,
            "width": int(original["width"]), "height": int(original["height"]),
            "license": original.get("license"),
            "flickr_url": original.get("flickr_url"),
            "coco_url": original.get("coco_url"),
            "awsurl": original.get("awsurl"),
            "download_url": f"https://s3.amazonaws.com/images.cocodataset.org/{split}2017/{file_name}",
        }
        rows.extend(sorted(rows_by_image[image_id], key=lambda row: row["annotation_id"]))
    prepared = {
        "format": 2, "dataset": "COCO-WholeBody v1.0", "split": split,
        "requested_image_limit": image_limit, "seed": seed,
        "annotation_source": source_url, "annotation_sha256": _digest(path),
        "source": COCO_SOURCE, "distribution": COCO_DISTRIBUTION,
        "license": "CC-BY-4.0 with upstream research/non-commercial restriction",
        "image_licenses": licenses, "images": image_records, "samples": rows,
    }
    _write_json(output, prepared)
    print(json.dumps({"event": "prepared_coco", "split": split,
                      "images": len(image_records), "persons": len(rows)}), flush=True)
    return prepared


def _image_path(root, split, record):
    return Path(root) / "images" / f"{split}2017" / record["file_name"]


def _ensure_image(root, split, record):
    path = _image_path(root, split, record)
    if path.is_file() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    # Each process has its own temporary path if two annotations share an image.
    temporary = path.with_name(f"{path.name}.{os.getpid()}.download")
    download(record["download_url"], temporary)
    try:
        with Image.open(temporary) as image:
            if image.size != (record["width"], record["height"]):
                raise ValueError(f"Image dimensions disagree with COCO metadata: {record['image_id']}")
            image.verify()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def ensure_images(root, split="train", rank=0, world_size=1, workers=8):
    """Fetch only images referenced by this worker's rank::world_size samples."""
    if not 0 <= rank < world_size:
        raise ValueError("Invalid image shard")
    manifest = _read_json(Path(root) / f"coco_{split}.json")
    ids = sorted({row["image_id"] for row in manifest["samples"][rank::world_size]})
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(_ensure_image, root, split, manifest["images"][str(image_id)]): image_id
            for image_id in ids
        }
        for future in as_completed(pending):
            try:
                future.result()
            except Exception as error:
                failures.append({"image_id": pending[future], "error": str(error)})
    _write_json(Path(root) / f"image_fetch_{split}_{rank}.json", {
        "split": split, "rank": rank, "world_size": world_size,
        "requested": len(ids), "downloaded": len(ids) - len(failures), "failures": failures,
    })
    if failures:
        # Never silently substitute a different person or a dummy image.
        raise RuntimeError(f"COCO image fetch failed for {len(failures)} images; see image_fetch_{split}_{rank}.json")
    return len(ids)


def _crop_box(box):
    x, y, width, height = box
    center_x, center_y = x + width / 2, y + height / 2
    height = max(height, width * IMAGE_HEIGHT / IMAGE_WIDTH) * CROP_EXPANSION
    width = height * IMAGE_WIDTH / IMAGE_HEIGHT
    return np.asarray((center_x - width / 2, center_y - height / 2,
                       center_x + width / 2, center_y + height / 2), dtype=np.float32)


class CocoWholeBodyDataset(Dataset):
    """RGB crops with 73 selected real labels from the unchanged 133-point cache."""
    def __init__(self, root, split="train"):
        self.root, self.split = Path(root), split
        self.manifest = _read_json(self.root / f"coco_{split}.json")
        self.samples = self.manifest["samples"]
        self.joint_ids = np.asarray(COMPACT_COCO_IDS, dtype=np.int64)
        self.joint_names = list(COMPACT_JOINT_NAMES)
        self.joint_schema = JOINT_SCHEMA

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        record = self.manifest["images"][str(sample["image_id"])]
        path = _ensure_image(self.root, self.split, record)
        box = _crop_box(sample["bbox"])
        with Image.open(path) as source:
            image = source.convert("RGB").transform(
                (IMAGE_WIDTH, IMAGE_HEIGHT), Image.Transform.EXTENT,
                tuple(float(value) for value in box), resample=Image.Resampling.BILINEAR,
            )
            rgb = np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0
        points = np.asarray(sample["keypoints"], dtype=np.float32)[self.joint_ids]
        xy = (points[:, :2] - box[:2]) / (box[2:] - box[:2])
        valid = (points[:, 2] > 0) & (xy >= 0).all(axis=-1) & (xy <= 1).all(axis=-1)
        # Invalid labels are numerically benign and their loss contribution is zero.
        xy[~valid] = 0
        x2d = torch.from_numpy(xy)
        valid2d = torch.from_numpy(valid.astype(np.float32))
        return {"image": torch.from_numpy(rgb), "x2d": x2d, "valid2d": valid2d,
                "keypoints": x2d, "visibility": valid2d,
                "source": "coco_wholebody", "split": self.split,
                "sample_id": f"coco2017/{self.split}/{sample['image_id']}/{sample['annotation_id']}",
                "image_id": sample["image_id"], "annotation_id": sample["annotation_id"]}


class CocoCalibrationDataset(CocoWholeBodyDataset):
    """Deterministic real COCO TRAIN crops for student-residual calibration.

    This class cannot select COCO validation/test examples. The model owns
    residual estimation and corruption; this dataset only supplies labeled RGB.
    """
    def __init__(self, root, max_samples=512, seed=20261102):
        if max_samples < 1:
            raise ValueError("Calibration max_samples must be positive")
        super().__init__(root, split="train")
        ordered = sorted(self.samples, key=lambda row: hashlib.sha256(
            f"calibration:{seed}:{row['image_id']}:{row['annotation_id']}".encode()
        ).digest())
        self.samples = ordered[:max_samples]
        self.calibration_manifest = {
            "source": "COCO-WholeBody train2017", "split": "train",
            "seed": seed, "requested_samples": max_samples,
            "actual_samples": len(self.samples),
            "annotation_ids": [row["annotation_id"] for row in self.samples],
            "image_ids": [row["image_id"] for row in self.samples],
            "annotation_sha256": self.manifest["annotation_sha256"],
            "purpose": "Estimate current student prediction residuals from training data only; not a validation metric.",
        }


def _extract_aspset(archive, destination):
    destination = Path(destination)
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            if not member.isfile():
                continue
            relative = Path(member.name)
            if relative.parts and relative.parts[0] == "ASPset-510":
                relative = Path(*relative.parts[1:])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unexpected archive member: {member.name}")
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise ValueError(f"Unreadable archive member: {member.name}")
            with source, path.open("wb") as target:
                shutil.copyfileobj(source, target)


def _label_indices(c3d):
    labels = [str(value).strip().lower().replace(" ", "_")
              for value in c3d["parameters"]["POINT"]["LABELS"]["value"]]
    missing = [name for name in BODY_NAMES if name not in labels]
    if missing:
        raise ValueError(f"ASPset labels lack common anatomical joints {missing}; labels={labels}")
    return [labels.index(name) for name in BODY_NAMES]


def prepare_aspset(root, frame_stride=5, max_frames=60000):
    """Project measured C3D joints with the corresponding calibrated cameras."""
    import ezc3d

    if frame_stride < 1 or max_frames < 1:
        raise ValueError("ASPset frame_stride and max_frames must be positive")
    root = Path(root)
    result_paths = [root / f"aspset_{split}.npz" for split in ("train", "val")]
    info_path = root / "aspset_preparation.json"
    if all(path.exists() for path in result_paths):
        info = _cached_preparation(root, "aspset", "aspset510")
        if (info.get("format") == 2 and info.get("frame_stride") == frame_stride
                and info.get("max_frames") == max_frames):
            return info
    raw = root / "aspset_raw"
    for name, expected_md5 in ASP_ARCHIVES.items():
        archive = download(f"{ASPBASE}/{name}", root / "downloads" / name, md5=expected_md5)
        _extract_aspset(archive, raw)
    with (raw / "splits.csv").open("r", encoding="utf-8", newline="") as handle:
        split_rows = list(csv.reader(handle))
    gathered = {split: {key: [] for key in ("keypoints", "target3d", "sample_ids")}
                for split in ("train", "val")}
    coordinate_transforms = {}
    clip_splits, seen_observations = {}, set()
    for row in split_rows:
        if len(row) != 4:
            raise ValueError(f"Malformed ASPset split row: {row}")
        subject, clip, split, camera_selector = row
        clip_key = (subject, clip)
        if clip_key in clip_splits and clip_splits[clip_key] != split:
            raise ValueError(f"ASPset clip appears in multiple data splits: {subject}-{clip}")
        clip_splits[clip_key] = split
        if split not in gathered:
            continue
        mocap_path = raw / "trainval" / "joints_3d" / subject / f"{subject}-{clip}.c3d"
        c3d = ezc3d.c3d(str(mocap_path))
        indices = _label_indices(c3d)
        point_data = np.asarray(c3d["data"]["points"], dtype=np.float64)
        world = point_data[:3, indices, ::frame_stride].transpose(2, 1, 0)
        unit = str(c3d["parameters"]["POINT"]["UNITS"]["value"][0]).strip().lower()
        if unit not in {"mm", "cm", "m"}:
            raise ValueError(f"Unrecognized ASPset coordinate unit: {unit}")
        meters_per_unit = {"mm": 0.001, "cm": 0.01, "m": 1.0}[unit]
        valid_world = np.isfinite(world).all(axis=(1, 2))
        if point_data.shape[0] > 3:
            valid_world &= (point_data[3, indices, ::frame_stride] >= 0).all(axis=0)
        cameras = ("left", "mid", "right") if camera_selector == "all" else (camera_selector,)
        for camera_id in cameras:
            observation_key = (subject, clip, camera_id)
            if observation_key in seen_observations:
                continue
            seen_observations.add(observation_key)
            camera_file = raw / "trainval" / "cameras" / subject / f"{subject}-{camera_id}.json"
            camera = _read_json(camera_file)
            intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float64).reshape(3, 4)
            extrinsic = np.asarray(camera["extrinsic_matrix"], dtype=np.float64).reshape(4, 4)
            homogeneous = np.concatenate((world, np.ones((*world.shape[:-1], 1))), axis=-1)
            camera_h = homogeneous @ extrinsic.T
            with np.errstate(divide="ignore", invalid="ignore"):
                camera_xyz = camera_h[..., :3] / camera_h[..., 3:4]
                image_h = camera_h @ intrinsic.T
                xy = image_h[..., :2] / image_h[..., 2:3]
            valid = valid_world & np.isfinite(camera_xyz).all(axis=(1, 2)) & np.isfinite(xy).all(axis=(1, 2))
            valid &= (np.abs(image_h[..., 2]) > 1e-6).all(axis=1)
            if not valid.any():
                raise ValueError(f"No projectable ASPset frames for {subject}-{clip}-{camera_id}")
            # Canonicalize the calibrated camera axes using projection signs.
            if (not np.allclose(intrinsic[2, :2], (0, 0))
                    or not np.isclose(intrinsic[2, 3], 0)
                    or abs(intrinsic[2, 2]) < 1e-12):
                raise ValueError("ASPset calibration has an unsupported projective depth axis")
            depth_sign = 1.0 if np.median(camera_xyz[valid, :, 2]) > 0 else -1.0
            valid &= (camera_xyz[..., 2] * depth_sign > 1e-6).all(axis=1)
            axes = np.asarray((np.sign(intrinsic[0, 0] / intrinsic[2, 2]) * depth_sign,
                               np.sign(intrinsic[1, 1] / intrinsic[2, 2]) * depth_sign, depth_sign))
            if not np.all(np.abs(axes) == 1):
                raise ValueError("Invalid focal length in ASPset calibration")
            canonical = camera_xyz * axes * meters_per_unit
            root_xyz = (canonical[:, 6:7] + canonical[:, 7:8]) / 2
            target = canonical - root_xyz
            lower, upper = xy.min(axis=1), xy.max(axis=1)
            size = upper - lower
            height = np.maximum(size[:, 1], size[:, 0] * IMAGE_HEIGHT / IMAGE_WIDTH) * CROP_EXPANSION
            valid &= height > 1e-6
            if not valid.any():
                continue
            crop_size = np.stack((height * IMAGE_WIDTH / IMAGE_HEIGHT, height), axis=-1)
            origin = (lower + upper - crop_size) / 2
            normalized = (xy - origin[:, None]) / np.maximum(crop_size[:, None], 1e-6)
            indices_valid = np.flatnonzero(valid)
            gathered[split]["keypoints"].append(normalized[valid].astype(np.float32))
            gathered[split]["target3d"].append(target[valid].astype(np.float32))
            gathered[split]["sample_ids"].extend(
                f"{subject}/{clip}/{camera_id}/{int(frame_index * frame_stride)}"
                for frame_index in indices_valid
            )
            coordinate_transforms[f"{subject}/{camera_id}"] = {
                "source_units": unit, "camera_axis_multipliers": axes.tolist(),
            }
    counts = {}
    for split, values in gathered.items():
        if not values["keypoints"]:
            raise RuntimeError(f"No ASPset-510 {split} frames")
        points = np.concatenate(values["keypoints"])
        targets = np.concatenate(values["target3d"])
        sample_ids = np.asarray(values["sample_ids"])
        if len(points) == 0:
            raise RuntimeError(f"No nondegenerate ASPset-510 {split} frames")
        # Uniform deterministic sampling spans all clips instead of taking one subject.
        limit = max_frames if split == "train" else max(1, min(max_frames, 6000))
        if len(points) > limit:
            chosen = np.random.default_rng(20261102 + (split == "val")).choice(len(points), limit, replace=False)
            chosen.sort()
            points, targets, sample_ids = points[chosen], targets[chosen], sample_ids[chosen]
        _write_npz(root / f"aspset_{split}.npz", keypoints=points,
                   target3d=targets, sample_ids=sample_ids)
        counts[split] = len(points)
    info = {
        "format": 2,
        "dataset": "ASPset-510 v1", "source": ASPSOURCE,
        "distribution": "https://archive.org/details/aspset510", "license": "CC0-1.0",
        "archives": [{"url": f"{ASPBASE}/{name}", "md5": md5} for name, md5 in ASP_ARCHIVES.items()],
        "frame_stride": frame_stride, "max_frames": max_frames, "counts": counts,
        "split_clips": {split: sorted(f"{subject}/{clip}" for (subject, clip), value in clip_splits.items() if value == split)
                        for split in ("train", "val")},
        "joint_names": list(BODY_NAMES), "coco_body_ids": list(COCO_BODY_IDS),
        "frame_id_semantics": "Zero-based index within the C3D point sequence; no RGB-frame alignment is asserted by this annotation-only loader.",
        "input_2d": "Projected body-joint bounding box, aspect 0.75, expansion 1.25, crop-normalized XY.",
        "coordinate_system": "camera x right, y down, z away; hip midpoint origin; meters",
        "modifications": "Calibrated 2D projection; anatomical joint selection; root subtraction; meter conversion; frame subsampling. No RGB videos used.",
        "coordinate_transforms": coordinate_transforms,
    }
    _write_json(info_path, info)
    return info


class ASPsetDataset(Dataset):
    """Legacy real body12 cache exposed as masked compact73 supervision."""
    def __init__(self, root, split="train"):
        with np.load(Path(root) / f"aspset_{split}.npz", allow_pickle=False) as archive:
            self.keypoints = torch.from_numpy(archive["keypoints"].astype(np.float32))
            self.target3d = torch.from_numpy(archive["target3d"].astype(np.float32))
            self.sample_ids = archive["sample_ids"].copy()
        if self.keypoints.shape[1:] != (12, 2) or self.target3d.shape[1:] != (12, 3):
            raise ValueError("ASPset prepared cache must contain the 12 declared body joints")
        if not len(self.keypoints) == len(self.target3d) == len(self.sample_ids):
            raise ValueError("ASPset prepared arrays have inconsistent sample counts")
        self.source, self.split = "aspset", split
        self.source_counts = {self.source: len(self.keypoints)}

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, index):
        x2d = torch.zeros((73, 2), dtype=torch.float32)
        x3d = torch.zeros((73, 3), dtype=torch.float32)
        valid2d = torch.zeros(73, dtype=torch.float32)
        valid3d = torch.zeros(73, dtype=torch.float32)
        x2d[:12], x3d[:12] = self.keypoints[index], self.target3d[index]
        valid2d[:12], valid3d[:12] = 1, 1
        return {"x2d": x2d, "valid2d": valid2d, "x3d": x3d, "valid3d": valid3d,
                "keypoints": x2d, "visibility": valid2d, "target3d": x3d,
                "source": self.source, "split": self.split,
                "sample_id": str(self.sample_ids[index])}


def _validate_h3wb_archive(path):
    """Verify the acquired official artifact before any object-array unpickling."""
    if _digest(path) != H3WB_SOURCE_SHA256:
        raise ValueError("Official H3WB NPZ SHA256 differs from the inspected source release")
    if not zipfile.is_zipfile(path):
        raise ValueError("H3WB download is not an NPZ archive")
    with zipfile.ZipFile(path) as archive:
        required = {"train_data.npy", "metadata.npy"}
        if not required.issubset(archive.namelist()):
            raise ValueError("Official H3WB NPZ lacks train_data or metadata")


def _download_h3wb(root, attempts=3):
    """Acquire the author-published NPZ; Google Drive HTML is never cached as data."""
    import gdown

    destination = Path(root) / "downloads" / "h3wb_official_train.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _validate_h3wb_archive(destination)
        return destination
    temporary = destination.with_name(destination.name + ".part")
    last_error = None
    for attempt in range(attempts):
        try:
            result = gdown.download(id=H3WB_FILE_ID, output=str(temporary),
                                    quiet=False, use_cookies=False)
            if result is None or not temporary.is_file() or not temporary.stat().st_size:
                raise RuntimeError("The official H3WB Google Drive download did not produce a file")
            _validate_h3wb_archive(temporary)
            temporary.replace(destination)
            return destination
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Could not acquire official H3WB annotations: {last_error}") from last_error


def _h3wb_points(values, frame_count, dimensions, label):
    points = np.asarray(values, dtype=np.float32)
    expected = frame_count * 133 * dimensions
    if points.size != expected or points.shape[-1] != dimensions:
        raise ValueError(f"Unexpected H3WB {label} shape: {points.shape}")
    if points.ndim > 2 and points.shape[0] != frame_count:
        raise ValueError(f"H3WB {label} frame dimension does not match frame_id")
    return points.reshape(frame_count, 133, dimensions)


def prepare_h3wb(root):
    """Prepare official camera-space whole-body labels with a subject-held-out split."""
    root = Path(root)
    outputs = [root / f"h3wb_{split}.npz" for split in ("train", "val")]
    cached = _cached_preparation(root, "h3wb", "h3wb")
    if (all(path.exists() for path in outputs) and cached.get("format") == 2
            and cached.get("source_file_id") == H3WB_FILE_ID
            and cached.get("source_sha256") == H3WB_SOURCE_SHA256
            and cached.get("compact_coco_ids") == COMPACT_COCO_IDS
            and cached.get("subjects") == {key: list(value) for key, value in H3WB_SUBJECTS.items()}
            and cached.get("crop") == {"aspect": IMAGE_WIDTH / IMAGE_HEIGHT,
                                        "expansion": CROP_EXPANSION}):
        return cached

    path = _download_h3wb(root)
    source_sha256 = _digest(path)
    # Only the author-published source uses object dictionaries. All prepared
    # worker files below are plain numeric/string arrays and need no pickle.
    with np.load(path, allow_pickle=True) as archive:
        training = archive["train_data"].item()
        metadata = archive["metadata"].item()
    expected_subjects = set(H3WB_SUBJECTS["train"] + H3WB_SUBJECTS["val"])
    missing_subjects = expected_subjects - set(training)
    if missing_subjects:
        raise ValueError(f"H3WB source lacks required subjects: {sorted(missing_subjects)}")
    selected_ids = np.asarray(COMPACT_COCO_IDS, dtype=np.int64)
    gathered = {split: {name: [] for name in
                       ("x2d", "valid2d", "x3d", "valid3d", "crop_boxes", "sample_ids")}
                for split in H3WB_SUBJECTS}
    counts_by_subject, source_counts, excluded_counts = {}, {}, {}
    for split, subjects in H3WB_SUBJECTS.items():
        counts_by_subject[split], source_counts[split], excluded_counts[split] = {}, 0, 0
        for subject in subjects:
            counts_by_subject[split][subject] = 0
            for action_name in sorted(training[subject]):
                action = training[subject][action_name]
                frame_ids = np.asarray(action["frame_id"]).reshape(-1)
                if not len(frame_ids):
                    continue
                if len(np.unique(frame_ids)) != len(frame_ids):
                    raise ValueError(f"Duplicate H3WB source frame IDs in {subject}/{action_name}")
                for camera in H3WB_CAMERAS:
                    if camera not in action:
                        continue
                    view = action[camera]
                    all_xy = _h3wb_points(view["pose_2d"], len(frame_ids), 2, "pose_2d")
                    xy = all_xy[:, selected_ids]
                    xyz = _h3wb_points(view["camera_3d"], len(frame_ids), 3, "camera_3d")[:, selected_ids] / 1000.0
                    source_counts[split] += len(frame_ids)
                    # The inspected official NPZ has no bbox dictionary. Use the
                    # full source133 2D extent, not the selected73 subset. The
                    # model's hip/RMS normalization is invariant to this common
                    # crop translation and isotropic scale after aspect correction.
                    finite_xy = np.isfinite(all_xy).all(axis=-1)
                    lower = np.min(np.where(finite_xy[..., None], all_xy, np.inf), axis=1)
                    upper = np.max(np.where(finite_xy[..., None], all_xy, -np.inf), axis=1)
                    boxes = np.concatenate((lower, upper), axis=-1)
                    size = boxes[:, 2:] - boxes[:, :2]
                    usable_box = np.isfinite(boxes).all(axis=1) & (size > 0).all(axis=1)
                    height = np.maximum(size[:, 1], size[:, 0] * IMAGE_HEIGHT / IMAGE_WIDTH) * CROP_EXPANSION
                    crop_size = np.stack((height * IMAGE_WIDTH / IMAGE_HEIGHT, height), axis=-1)
                    origin = (boxes[:, :2] + boxes[:, 2:] - crop_size) / 2
                    with np.errstate(divide="ignore", invalid="ignore"):
                        normalized = (xy - origin[:, None]) / crop_size[:, None]
                    valid2d = np.isfinite(normalized).all(axis=-1)
                    valid2d &= (normalized >= 0).all(axis=-1) & (normalized <= 1).all(axis=-1)
                    valid3d = np.isfinite(xyz).all(axis=-1)
                    usable = usable_box & valid3d[:, 6] & valid3d[:, 7]
                    usable &= valid2d[:, :12].sum(axis=1) >= 6
                    root_xyz = (xyz[:, 6:7] + xyz[:, 7:8]) / 2
                    target = xyz - root_xyz
                    normalized = np.where(valid2d[..., None], normalized, 0).astype(np.float32)
                    target = np.where(valid3d[..., None], target, 0).astype(np.float32)
                    accepted = np.flatnonzero(usable)
                    excluded_counts[split] += len(frame_ids) - len(accepted)
                    counts_by_subject[split][subject] += len(accepted)
                    if not len(accepted):
                        continue
                    chunk = gathered[split]
                    chunk["x2d"].append(normalized[usable])
                    chunk["valid2d"].append(valid2d[usable].astype(np.uint8))
                    chunk["x3d"].append(target[usable])
                    chunk["valid3d"].append(valid3d[usable].astype(np.uint8))
                    chunk["crop_boxes"].append(np.concatenate((origin, origin + crop_size), axis=-1)[usable])
                    chunk["sample_ids"].extend(
                        f"{subject}/{action_name}/{camera}/{frame_ids[index]}" for index in accepted
                    )
    counts, labeled_3d_points = {}, {}
    for split, values in gathered.items():
        if not values["x2d"]:
            raise RuntimeError(f"No usable H3WB {split} samples")
        arrays = {key: np.concatenate(parts) for key, parts in values.items() if key != "sample_ids"}
        sample_ids = values["sample_ids"]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError(f"H3WB {split} contains duplicate subject/action/camera/frame IDs")
        arrays["sample_ids"] = np.asarray(sample_ids)
        arrays["compact_coco_ids"] = selected_ids
        _write_npz(root / f"h3wb_{split}.npz", **arrays)
        counts[split] = len(sample_ids)
        labeled_3d_points[split] = int(arrays["valid3d"].sum())
        print(json.dumps({"event": "prepared_h3wb", "split": split,
                          "samples": counts[split], "subjects": list(H3WB_SUBJECTS[split])}), flush=True)
    info = {
        "format": 2, "dataset": "H3WB official whole-body annotations",
        "source": H3WB_SOURCE, "distribution": H3WB_DISTRIBUTION,
        "source_file_id": H3WB_FILE_ID, "source_sha256": source_sha256,
        "source_bytes": path.stat().st_size, "source_npz_keys": ["metadata", "train_data"],
        "license": "MIT", "license_url": f"{H3WB_SOURCE}/blob/main/LICENSE.md",
        "copyright": "Copyright (c) 2022 H3WB 3D WholeBody Dataset Annotations",
        "subjects": {key: list(value) for key, value in H3WB_SUBJECTS.items()},
        "counts": counts, "counts_by_subject": counts_by_subject,
        "source_counts_before_filtering": source_counts, "excluded_counts": excluded_counts,
        "labeled_3d_points": labeled_3d_points,
        "compact_coco_ids": COMPACT_COCO_IDS, "joint_names": COMPACT_JOINT_NAMES,
        "source_joint_count": 133, "crop": {"aspect": IMAGE_WIDTH / IMAGE_HEIGHT,
                                              "expansion": CROP_EXPANSION},
        "bbox_source": "Extent of all133 finite source pose_2d landmarks; the official NPZ contains no bbox dictionary.",
        "input_2d": "Official pose_2d pixel coordinates mapped through the full133-keypoint extent bbox with aspect0.75 and expansion1.25; XY in [0,1], availability mask separate.",
        "coordinate_system": "Camera x right, y down, z away; midpoint of COCO hips 11 and 12 as origin; meters (camera_3d / 1000).",
        "source_metadata_keys": sorted(str(key) for key in metadata),
        "split_policy": "S1/S5/S6 training; S7 validation. All actions, frames and camera views for each subject remain in one split. S8 test data is not used.",
        "label_provenance": "Author-released multiview reconstructed/completed whole-body annotations; not optical-marker measurements for every face/finger landmark.",
        "modifications": "Select declared compact73 joints, transform official 2D coordinates using their full133-point extent and the common crop rule, convert 3D millimeters to meters, subtract hip midpoint, retain availability masks and unchanged zero-padded source frame IDs.",
        "rgb": "No Human3.6M RGB images downloaded or redistributed. Their separate license is not replaced by the annotation MIT license.",
        "implementation_reference": "https://github.com/open-mmlab/mmpose/blob/main/mmpose/datasets/datasets/wholebody3d/h3wb_dataset.py",
    }
    _write_json(root / "h3wb_preparation.json", info)
    return info


class H3WBDataset(Dataset):
    def __init__(self, root, split="train"):
        with np.load(Path(root) / f"h3wb_{split}.npz", allow_pickle=False) as archive:
            if archive["compact_coco_ids"].tolist() != COMPACT_COCO_IDS:
                raise ValueError("H3WB prepared joint ordering does not match the model schema")
            self.x2d = torch.from_numpy(archive["x2d"].astype(np.float32))
            self.valid2d = torch.from_numpy(archive["valid2d"].astype(np.float32))
            self.x3d = torch.from_numpy(archive["x3d"].astype(np.float32))
            self.valid3d = torch.from_numpy(archive["valid3d"].astype(np.float32))
            self.sample_ids = archive["sample_ids"].copy()
        count = len(self.sample_ids)
        if (self.x2d.shape != (count, 73, 2) or self.x3d.shape != (count, 73, 3)
                or self.valid2d.shape != (count, 73) or self.valid3d.shape != (count, 73)):
            raise ValueError("H3WB prepared sample shapes do not match compact73")
        self.source, self.split = "h3wb", split
        self.source_counts = {self.source: count}

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        x2d, valid2d = self.x2d[index], self.valid2d[index]
        x3d, valid3d = self.x3d[index], self.valid3d[index]
        return {"x2d": x2d, "valid2d": valid2d, "x3d": x3d, "valid3d": valid3d,
                "keypoints": x2d, "visibility": valid2d, "target3d": x3d,
                "source": self.source, "split": self.split,
                "sample_id": str(self.sample_ids[index])}


class Unified3DDataset(ConcatDataset):
    """Whole-body H3WB and body-only ASPset with explicit source/label masks."""
    def __init__(self, root, split="train"):
        datasets = [ASPsetDataset(root, split), H3WBDataset(root, split)]
        super().__init__(datasets)
        self.source_counts = {dataset.source: len(dataset) for dataset in datasets}
        self.split = split
        self.joint_names = list(COMPACT_JOINT_NAMES)
        self.joint_schema = JOINT_SCHEMA


def _download_aspset_rgb_member(directory, max_download_bytes=256 * 1024 * 1024):
    """Stream only the inspected first test video from the official gzip tar."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "1e28-0001-left.mkv"
    acquisition_path = directory / "video_acquisition.json"
    if destination.exists() and _digest(destination) == ASP_RGB_VIDEO_SHA256:
        if acquisition_path.exists():
            return destination, _read_json(acquisition_path)
        return destination, {"source_url": ASP_RGB_VIDEO_URL, "archive_member": ASP_RGB_VIDEO_MEMBER,
                             "sha256": ASP_RGB_VIDEO_SHA256, "member_bytes": destination.stat().st_size,
                             "compressed_bytes_transferred": None, "reused_local_file": True}

    class CappedReader:
        def __init__(self, raw):
            self.raw, self.bytes_read = raw, 0

        def read(self, size=-1):
            if size < 0:
                size = 64 * 1024
            if self.bytes_read + size > max_download_bytes:
                raise RuntimeError("ASPset video-member acquisition exceeded its256MiB transfer cap")
            chunk = self.raw.read(size)
            self.bytes_read += len(chunk)
            return chunk

    temporary = destination.with_name(destination.name + ".part")
    last_error = None
    for attempt in range(3):
        try:
            with requests.get(ASP_RGB_VIDEO_URL, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                reader = CappedReader(response.raw)
                found = False
                with tarfile.open(fileobj=reader, mode="r|gz") as archive:
                    for member in archive:
                        if not member.isfile() or member.name != ASP_RGB_VIDEO_MEMBER:
                            continue
                        source = archive.extractfile(member)
                        if source is None:
                            raise ValueError("ASPset evaluation video member could not be read")
                        with source, temporary.open("wb") as target:
                            shutil.copyfileobj(source, target, length=1024 * 1024)
                        if _digest(temporary) != ASP_RGB_VIDEO_SHA256:
                            raise ValueError("ASPset evaluation video differs from the inspected official member")
                        found = True
                        break
                if not found:
                    raise ValueError("The inspected ASPset evaluation video is missing from the official archive")
                acquisition = {
                    "source_url": ASP_RGB_VIDEO_URL, "archive_member": ASP_RGB_VIDEO_MEMBER,
                    "sha256": ASP_RGB_VIDEO_SHA256, "member_bytes": temporary.stat().st_size,
                    "compressed_bytes_transferred": reader.bytes_read,
                    "whole_archive_downloaded": False,
                }
            temporary.replace(destination)
            _write_json(acquisition_path, acquisition)
            return destination, acquisition
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not acquire the official ASPset evaluation clip: {last_error}") from last_error


def _decode_aspset_rgb_crops(video, frame_indices, crop_boxes, output_directory):
    """Decode actual selected frames with ffmpeg and retain only person crops."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", str(video),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    # A fixed region covering all selected person crops avoids piping the entire
    # 4K frame while retaining every pixel needed for the per-frame crop.
    left = max(0, int(np.floor(crop_boxes[:, 0].min())))
    top = max(0, int(np.floor(crop_boxes[:, 1].min())))
    right = min(width, int(np.ceil(crop_boxes[:, 2].max())))
    bottom = min(height, int(np.ceil(crop_boxes[:, 3].max())))
    if right <= left or bottom <= top:
        raise ValueError("ASPset evaluation boxes do not intersect the source video")
    region_width, region_height = right - left, bottom - top
    select = "+".join(f"eq(n\\,{int(index)})" for index in frame_indices)
    filters = f"select={select},format=rgb24,crop={region_width}:{region_height}:{left}:{top}"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
               "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", filters,
               "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    image_names = []
    frame_bytes = region_width * region_height * 3
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            for frame_index, box in zip(frame_indices, crop_boxes):
                chunks, remaining = [], frame_bytes
                while remaining:
                    chunk = process.stdout.read(remaining)
                    if not chunk:
                        errors.seek(0)
                        detail = errors.read().decode("utf-8", errors="replace")
                        raise RuntimeError(f"ffmpeg ended before ASPset frame{int(frame_index)}: {detail}")
                    chunks.append(chunk)
                    remaining -= len(chunk)
                region = Image.frombytes("RGB", (region_width, region_height), b"".join(chunks))
                local_box = box - np.asarray((left, top, left, top), dtype=np.float32)
                crop = region.transform((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Transform.EXTENT,
                                        tuple(float(value) for value in local_box),
                                        resample=Image.Resampling.BILINEAR)
                image_name = f"frame_{int(frame_index):06d}.png"
                image_path = output_directory / image_name
                temporary = image_path.with_name(image_name + ".tmp")
                crop.save(temporary, format="PNG")
                temporary.replace(image_path)
                image_names.append(image_name)
            process.stdout.close()
            returncode = process.wait(timeout=120)
            if returncode:
                errors.seek(0)
                raise RuntimeError(f"ffmpeg decode failed: {errors.read().decode('utf-8', errors='replace')}")
        finally:
            if process.stdout is not None and not process.stdout.closed:
                process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()
    return image_names, {"source_width": width, "source_height": height,
                         "source_frame_rate": stream["r_frame_rate"],
                         "decoded_region": [left, top, right, bottom],
                         "decoder": "ffmpeg selected zero-based frames, RGB24 region, Pillow bilinear person crops"}


def prepare_aspset_rgb_evaluation(root, max_frames=192):
    """Prepare a real held-out RGB/body3D clip, never used for training or calibration."""
    import ezc3d

    if max_frames < 1:
        raise ValueError("ASPset RGB max_frames must be positive")
    directory = Path(root) / "aspset_rgb_test"
    manifest_path, labels_path = directory / "manifest.json", directory / "labels.npz"
    if manifest_path.exists() and labels_path.exists():
        cached = _read_json(manifest_path)
        if (cached.get("format") == 1 and cached.get("requested_max_frames") == max_frames
                and cached.get("video", {}).get("sha256") == ASP_RGB_VIDEO_SHA256
                and cached.get("crop") == {"aspect": IMAGE_WIDTH / IMAGE_HEIGHT,
                                            "expansion": CROP_EXPANSION,
                                            "height": IMAGE_HEIGHT, "width": IMAGE_WIDTH}
                and all((directory / "frames" / name).exists() for name in cached.get("images", []))):
            return cached
    raw = directory / "raw"
    for name, md5 in ASP_RGB_ARCHIVES.items():
        archive = download(f"{ASPBASE}/{name}", Path(root) / "downloads" / name, md5=md5)
        _extract_aspset(archive, raw)
    subject, clip, camera_id = "1e28", "0001", "left"
    with (raw / "splits.csv").open("r", encoding="utf-8", newline="") as handle:
        matching = [row for row in csv.reader(handle) if row[:2] == [subject, clip]]
    if matching != [[subject, clip, "test", camera_id]]:
        raise ValueError("ASPset RGB source clip is not the declared held-out test camera")
    camera = _read_json(raw / "test" / "cameras" / subject / f"{subject}-{camera_id}.json")
    c3d = ezc3d.c3d(str(raw / "test" / "joints_3d" / subject / f"{subject}-{clip}.c3d"))
    indices = _label_indices(c3d)
    point_data = np.asarray(c3d["data"]["points"], dtype=np.float64)
    world = point_data[:3, indices, :].transpose(2, 1, 0)
    units = str(c3d["parameters"]["POINT"]["UNITS"]["value"][0]).strip().lower()
    if units != "mm":
        raise ValueError("The inspected ASPset RGB evaluation clip must use millimeter C3D units")
    with (raw / "test" / "boxes" / subject / f"{subject}-{clip}-{camera_id}.csv").open(
            "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader) != ["x1", "y1", "x2", "y2"]:
            raise ValueError("Unexpected ASPset evaluation bbox columns")
        boxes = np.asarray([[float(value) for value in row] for row in reader], dtype=np.float32)
    if len(boxes) != len(world):
        raise ValueError("ASPset RGB bbox and C3D frame counts disagree")
    intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float64).reshape(3, 4)
    extrinsic = np.asarray(camera["extrinsic_matrix"], dtype=np.float64).reshape(4, 4)
    homogeneous = np.concatenate((world, np.ones((*world.shape[:-1], 1))), axis=-1)
    camera_h = homogeneous @ extrinsic.T
    with np.errstate(divide="ignore", invalid="ignore"):
        camera_xyz = camera_h[..., :3] / camera_h[..., 3:4]
        projected = camera_h @ intrinsic.T
        xy = projected[..., :2] / projected[..., 2:3]
    valid = np.isfinite(camera_xyz).all(axis=(1, 2)) & np.isfinite(xy).all(axis=(1, 2))
    valid &= (point_data[3, indices, :] >= 0).all(axis=0)
    valid &= (camera_xyz[..., 2] > 0).all(axis=1)
    valid &= np.isfinite(boxes).all(axis=1) & (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
    # This inspected camera has positive focal lengths and identity extrinsics;
    # its camera frame is already x-right, y-down and z-away.
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0 or not np.allclose(intrinsic[2], (0, 0, 1, 0)):
        raise ValueError("Unexpected ASPset evaluation camera-axis convention")
    available = np.flatnonzero(valid)
    if not len(available):
        raise ValueError("ASPset evaluation clip contains no valid body3D frames")
    chosen = np.linspace(0, len(available) - 1, min(max_frames, len(available)), dtype=np.int64)
    frame_indices = available[chosen]
    selected_boxes = boxes[frame_indices]
    crop_boxes = np.stack([_crop_box((box[0], box[1], box[2] - box[0], box[3] - box[1]))
                           for box in selected_boxes])
    normalized = (xy[frame_indices] - crop_boxes[:, None, :2]) / (crop_boxes[:, None, 2:] - crop_boxes[:, None, :2])
    target = camera_xyz[frame_indices].astype(np.float32) / 1000
    target -= (target[:, 6:7] + target[:, 7:8]) / 2
    count = len(frame_indices)
    x2d, x3d = np.zeros((count, 73, 2), np.float32), np.zeros((count, 73, 3), np.float32)
    valid2d, valid3d = np.zeros((count, 73), np.float32), np.zeros((count, 73), np.float32)
    body_valid2d = (normalized >= 0).all(axis=-1) & (normalized <= 1).all(axis=-1)
    x2d[:, :12] = np.where(body_valid2d[..., None], normalized, 0)
    x3d[:, :12] = target
    valid2d[:, :12], valid3d[:, :12] = body_valid2d, 1
    video, acquisition = _download_aspset_rgb_member(directory)
    image_names, video_info = _decode_aspset_rgb_crops(video, frame_indices, crop_boxes, directory / "frames")
    sample_ids = np.asarray([f"{subject}/{clip}/{camera_id}/{int(index)}" for index in frame_indices])
    _write_npz(labels_path, x2d=x2d, valid2d=valid2d, x3d=x3d, valid3d=valid3d,
               frame_indices=frame_indices, crop_boxes=crop_boxes, sample_ids=sample_ids)
    manifest = {
        "format": 1, "dataset": "ASPset-510 actual RGB/body3D held-out clip", "split": "test",
        "subject": subject, "clip": clip, "camera": camera_id,
        "requested_max_frames": max_frames, "count": count,
        "source_frame_count": len(world), "source_frame_indices": frame_indices.tolist(),
        "source_joint_count": int(point_data.shape[1]), "source_units": units,
        "video": acquisition, "video_metadata": video_info, "images": image_names,
        "source": ASPSOURCE, "distribution": "https://archive.org/details/aspset510",
        "license": "CC0-1.0", "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "archives": [{"url": f"{ASPBASE}/{name}", "md5": md5} for name, md5 in ASP_RGB_ARCHIVES.items()],
        "crop": {"aspect": IMAGE_WIDTH / IMAGE_HEIGHT, "expansion": CROP_EXPANSION,
                 "height": IMAGE_HEIGHT, "width": IMAGE_WIDTH},
        "crop_source": "Official frame-aligned ground-truth person bbox CSV. The detector is outside this evaluation's scope.",
        "frame_alignment": "The official ASPset loader indexes mocap.joint_positions[i], bounding_boxes[i] and VideoLoader.select_frame(i) with the same zero-based i. This source clip has C3D first_frame0 and50fps.",
        "alignment_reference": f"{ASPSOURCE}/blob/master/src/aspset510/torch/__init__.py",
        "coordinates": "Camera x right, y down, z away; hip-midpoint-relative meters; only compact0..11 body labels valid.",
        "evaluation_scope": "One real held-out test clip, actual RGB-to-body3D under ground-truth person crops. It does not measure whole-body hand/face/foot3D end-to-end accuracy or person detection.",
        "training_usage": "Evaluation only; no training, fitting, residual-bank calibration or dataset selection uses these test frames.",
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({"event": "prepared_aspset_rgb_evaluation", "frames": count,
                      "clip": f"{subject}/{clip}/{camera_id}", "split": "test"}), flush=True)
    return manifest


class ASPsetRGBEvaluationDataset(Dataset):
    """Actual held-out RGB inputs paired with real camera-space body3D labels."""
    def __init__(self, root, max_frames=192):
        self.manifest = prepare_aspset_rgb_evaluation(root, max_frames=max_frames)
        self.directory = Path(root) / "aspset_rgb_test"
        with np.load(self.directory / "labels.npz", allow_pickle=False) as archive:
            self.x2d = torch.from_numpy(archive["x2d"].copy())
            self.valid2d = torch.from_numpy(archive["valid2d"].copy())
            self.x3d = torch.from_numpy(archive["x3d"].copy())
            self.valid3d = torch.from_numpy(archive["valid3d"].copy())
            self.crop_boxes = torch.from_numpy(archive["crop_boxes"].copy())
            self.frame_indices = archive["frame_indices"].copy()
            self.sample_ids = archive["sample_ids"].copy()

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        with Image.open(self.directory / "frames" / self.manifest["images"][index]) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.float32).transpose(2, 0, 1).copy() / 255
        x2d, valid2d, x3d, valid3d = self.x2d[index], self.valid2d[index], self.x3d[index], self.valid3d[index]
        return {"image": torch.from_numpy(rgb), "x2d": x2d, "valid2d": valid2d,
                "x3d": x3d, "valid3d": valid3d, "keypoints": x2d,
                "visibility": valid2d, "target3d": x3d,
                "source": "aspset_rgb_test", "split": "test", "sample_id": str(self.sample_ids[index]),
                "subject": self.manifest["subject"], "clip": self.manifest["clip"],
                "camera": self.manifest["camera"], "frame_index": int(self.frame_indices[index]),
                "crop_box": self.crop_boxes[index]}


def prepare_data(root, train_limit=8192, val_limit=256, download_images=False,
                 workers=8, frame_stride=5, max_3d_frames=60000):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    coco = {split: prepare_coco(root, split, limit)
            for split, limit in (("train", train_limit), ("val", val_limit))}
    overlap = set(coco["train"]["images"]) & set(coco["val"]["images"])
    if overlap:
        raise ValueError("COCO train/val image IDs overlap")
    aspset = prepare_aspset(root, frame_stride=frame_stride, max_frames=max_3d_frames)
    h3wb = prepare_h3wb(root)
    manifest = {
        "format": 2,
        "prepared_files": list(PREPARED_FILES),
        "joint_schema": JOINT_SCHEMA,
        "rgb_input": {"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH,
                      "color": "RGB", "range": [0.0, 1.0],
                      "crop_expansion": CROP_EXPANSION, "crop_aspect": IMAGE_WIDTH / IMAGE_HEIGHT},
        "source_counts": {
            split: {"coco_wholebody": len(coco[split]["samples"]),
                    "aspset": int(aspset["counts"][split]),
                    "h3wb": int(h3wb["counts"][split])}
            for split in ("train", "val")
        },
        "coco_wholebody": {
            "source": COCO_SOURCE, "distribution": COCO_DISTRIBUTION,
            "annotation_license": "CC-BY-4.0 plus upstream research/non-commercial restriction",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "terms_url": f"{COCO_SOURCE}#terms-of-use",
            "images": "Individual source image licenses; COCO/Flickr terms apply. Original images are not republished.",
            "modifications": "Preserve source 133-joint prepared annotations; select compact73 at runtime; aspect-preserving 1.25-expanded person crops resized to 256x192 RGB; invalid 2D labels excluded from loss.",
            "splits": {split: {"images": len(value["images"]), "persons": len(value["samples"]),
                               "annotation_source": value["annotation_source"],
                               "annotation_sha256": value["annotation_sha256"],
                               "image_licenses": value["image_licenses"],
                               "image_attribution": list(value["images"].values())}
                       for split, value in coco.items()},
        },
        "aspset510": aspset,
        "h3wb": h3wb,
        "supervision": {
            "coco_wholebody": "73 selected 2D points with source annotation validity; no 3D targets.",
            "aspset": "Only compact0..11 body targets are available in 2D and 3D. All other masks are zero; missing targets are not invented.",
            "h3wb": "73 selected source 2D/3D points with separate availability masks. Dropping an input landmark during training does not remove its valid 3D target.",
        },
        "residual_calibration": {
            "dataset_class": "CocoCalibrationDataset", "split": "train",
            "selection": "Deterministic hash of seed, image_id and annotation_id; limited only for residual-bank computation.",
            "model_api": "model.calibrate_residuals(iterable_of_batches, max_samples=512)",
            "purpose": "Measure deployed student 2D errors using labeled COCO training crops. Validation/test examples are never calibration inputs.",
        },
        "unused_candidates": ["CMU Human Foot Keypoint (separate download)", "AIST++", "Human3.6M RGB images", "FreiHAND", "InterHand2.6M"],
    }
    _write_json(root / "dataset_manifest.json", manifest)
    if download_images:
        for split in ("train", "val"):
            ensure_images(root, split, workers=workers)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "images", "rgb-eval"))
    parser.add_argument("--root", default="data")
    parser.add_argument("--train-images", type=int, default=8192)
    parser.add_argument("--val-images", type=int, default=256)
    parser.add_argument("--max-3d-frames", type=int, default=60000)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--rgb-eval-frames", type=int, default=192)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_data(args.root, train_limit=args.train_images, val_limit=args.val_images,
                     download_images=args.download_images, workers=args.io_workers,
                     frame_stride=args.frame_stride, max_3d_frames=args.max_3d_frames)
    elif args.command == "images":
        ensure_images(args.root, args.split, rank=args.rank,
                      world_size=args.world_size, workers=args.io_workers)
    else:
        prepare_aspset_rgb_evaluation(args.root, max_frames=args.rgb_eval_frames)


if __name__ == "__main__":
    main()
