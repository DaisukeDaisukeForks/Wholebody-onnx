"""Real COCO-WholeBody and ASPset-510 data, fetched without video archives.

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
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 144
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
PREPARED_FILES = (
    "coco_train.json", "coco_val.json", "aspset_train.npz", "aspset_val.npz",
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
    height = max(height, width * IMAGE_HEIGHT / IMAGE_WIDTH) * 1.25
    width = height * IMAGE_WIDTH / IMAGE_HEIGHT
    return np.asarray((center_x - width / 2, center_y - height / 2,
                       center_x + width / 2, center_y + height / 2), dtype=np.float32)


class CocoWholeBodyDataset(Dataset):
    """RGB crops and only the valid 133-point 2D labels (no invented 3D labels)."""
    def __init__(self, root, split="train"):
        self.root, self.split = Path(root), split
        self.manifest = _read_json(self.root / f"coco_{split}.json")
        self.samples = self.manifest["samples"]

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
        points = np.asarray(sample["keypoints"], dtype=np.float32)
        xy = (points[:, :2] - box[:2]) / (box[2:] - box[:2])
        valid = (points[:, 2] > 0) & (xy >= 0).all(axis=-1) & (xy <= 1).all(axis=-1)
        # Invalid labels are numerically benign and their loss contribution is zero.
        xy[~valid] = 0
        return {"image": torch.from_numpy(rgb), "keypoints": torch.from_numpy(xy),
                "visibility": torch.from_numpy(valid.astype(np.float32)),
                "image_id": sample["image_id"], "annotation_id": sample["annotation_id"]}


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
    if all(path.exists() for path in result_paths) and info_path.exists():
        info = _read_json(info_path)
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
            axes = np.asarray((np.sign(intrinsic[0, 0] / intrinsic[2, 2]) * depth_sign,
                               np.sign(intrinsic[1, 1] / intrinsic[2, 2]) * depth_sign, depth_sign))
            if not np.all(np.abs(axes) == 1):
                raise ValueError("Invalid focal length in ASPset calibration")
            canonical = camera_xyz * axes * meters_per_unit
            root_xyz = (canonical[:, 6:7] + canonical[:, 7:8]) / 2
            target = canonical - root_xyz
            lower, upper = xy.min(axis=1), xy.max(axis=1)
            size = upper - lower
            height = np.maximum(size[:, 1], size[:, 0] * IMAGE_HEIGHT / IMAGE_WIDTH) * 1.25
            valid &= height > 1e-6
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
        # Uniform deterministic sampling spans all clips instead of taking one subject.
        limit = max_frames if split == "train" else max(1, min(max_frames, 6000))
        if len(points) > limit:
            chosen = np.random.default_rng(20261102 + (split == "val")).choice(len(points), limit, replace=False)
            chosen.sort()
            points, targets, sample_ids = points[chosen], targets[chosen], sample_ids[chosen]
        np.savez_compressed(root / f"aspset_{split}.npz", keypoints=points,
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
        "coordinate_system": "camera x right, y down, z away; hip midpoint origin; meters",
        "modifications": "Calibrated 2D projection; anatomical joint selection; root subtraction; meter conversion; frame subsampling. No RGB videos used.",
        "coordinate_transforms": coordinate_transforms,
    }
    _write_json(info_path, info)
    return info


class ASPsetDataset(Dataset):
    def __init__(self, root, split="train"):
        with np.load(Path(root) / f"aspset_{split}.npz", allow_pickle=False) as archive:
            self.keypoints = torch.from_numpy(archive["keypoints"].copy())
            self.target3d = torch.from_numpy(archive["target3d"].copy())
            self.sample_ids = archive["sample_ids"].copy()

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, index):
        return {"keypoints": self.keypoints[index], "target3d": self.target3d[index],
                "visibility": torch.ones(12, dtype=torch.float32),
                "sample_id": str(self.sample_ids[index])}


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
    manifest = {
        "format": 1,
        "prepared_files": list(PREPARED_FILES),
        "coco_wholebody": {
            "source": COCO_SOURCE, "distribution": COCO_DISTRIBUTION,
            "annotation_license": "CC-BY-4.0 plus upstream research/non-commercial restriction",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "terms_url": f"{COCO_SOURCE}#terms-of-use",
            "images": "Individual source image licenses; COCO/Flickr terms apply. Original images are not republished.",
            "modifications": "Person selection; aspect-preserving expanded crops; RGB resizing; 133-joint concatenation; invalid labels excluded from loss.",
            "splits": {split: {"images": len(value["images"]), "persons": len(value["samples"]),
                               "annotation_source": value["annotation_source"],
                               "annotation_sha256": value["annotation_sha256"],
                               "image_licenses": value["image_licenses"],
                               "image_attribution": list(value["images"].values())}
                       for split, value in coco.items()},
        },
        "aspset510": aspset,
        "unused_candidates": ["CMU Human Foot Keypoint (separate download)", "AIST++", "Human3.6M"],
    }
    _write_json(root / "dataset_manifest.json", manifest)
    if download_images:
        for split in ("train", "val"):
            ensure_images(root, split, workers=workers)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "images"))
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
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_data(args.root, train_limit=args.train_images, val_limit=args.val_images,
                     download_images=args.download_images, workers=args.io_workers,
                     frame_stride=args.frame_stride, max_3d_frames=args.max_3d_frames)
    else:
        ensure_images(args.root, args.split, rank=args.rank,
                      world_size=args.world_size, workers=args.io_workers)


if __name__ == "__main__":
    main()
