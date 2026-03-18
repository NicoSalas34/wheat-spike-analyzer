#!/usr/bin/env python3
"""
Build a YOLO-pose rachis dataset from the existing YOLO-seg rachis dataset.

Input dataset format (YOLO-seg):
    training_rachis/
      images/train, images/val
      labels/train, labels/val

Output dataset format (YOLO-pose):
    training_rachis_pose/
    images/train, images/val  (copied by default)
      labels/train, labels/val
      data.yaml

Each segmentation polygon is converted into 5 keypoints along the main axis,
with a pose bbox computed from the polygon extents.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def parse_seg_line(line: str) -> tuple[int, np.ndarray] | None:
    tokens = line.strip().split()
    if len(tokens) < 7:
        return None

    try:
        class_id = int(float(tokens[0]))
        coords = np.asarray([float(x) for x in tokens[1:]], dtype=np.float64)
    except ValueError:
        return None

    if coords.size < 6 or coords.size % 2 != 0:
        return None

    pts = coords.reshape(-1, 2)
    pts = np.clip(pts, 0.0, 1.0)
    return class_id, pts


def polygon_centerline(points: np.ndarray) -> np.ndarray:
    """
    Extract the centerline from a thin rachis polygon.

    The segmentation labels are annotated as thin polygons where the first
    half of the vertices traces one side and the second half traces the
    return side. Averaging paired vertices (zip of both halves) gives the
    true centerline, regardless of how many points the polygon has.
    """
    n = len(points)
    half = n // 2
    if half < 2:
        # Degenerate polygon: just return the points as-is
        return points.copy()

    side_a = points[:half]             # forward direction
    side_b = points[half:][::-1]       # return side, reversed so it aligns

    # If n is odd, the extra point is usually a cap; use as-is for side_b.
    min_len = min(len(side_a), len(side_b))
    centerline = (side_a[:min_len] + side_b[:min_len]) / 2.0
    return np.clip(centerline, 0.0, 1.0)


def polygon_to_keypoints(points: np.ndarray, kpt_count: int = 5) -> np.ndarray:
    """
    Sample kpt_count evenly-spaced keypoints from the rachis centerline.

    We first reconstruct the true centerline (average of the two sides of the
    thin polygon), then resample it at kpt_count arc-length-equidistant positions.
    """
    centerline = polygon_centerline(points)

    if len(centerline) < 2:
        # Can't interpolate a single point; duplicate it
        return np.tile(centerline[0], (kpt_count, 1))

    if len(centerline) == kpt_count:
        return centerline

    # Arc-length parameterisation
    deltas = np.diff(centerline, axis=0)
    seg_lens = np.linalg.norm(deltas, axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cumlen[-1]

    if total < 1e-9:
        # Degenerate: all points coincide
        return np.tile(centerline[0], (kpt_count, 1))

    targets = np.linspace(0.0, total, num=kpt_count)
    keypoints = []
    for t in targets:
        # Find which segment this target falls in
        idx = int(np.searchsorted(cumlen, t, side='right')) - 1
        idx = min(idx, len(centerline) - 2)
        seg_start = cumlen[idx]
        seg_end = cumlen[idx + 1]
        seg_len = seg_end - seg_start
        alpha = (t - seg_start) / seg_len if seg_len > 1e-12 else 0.0
        kp = centerline[idx] * (1 - alpha) + centerline[idx + 1] * alpha
        keypoints.append(kp)

    return np.clip(np.asarray(keypoints, dtype=np.float64), 0.0, 1.0)


def points_to_bbox(points: np.ndarray) -> tuple[float, float, float, float]:
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    cx, cy = (min_xy + max_xy) / 2.0
    wh = np.maximum(max_xy - min_xy, 1e-6)
    return float(cx), float(cy), float(wh[0]), float(wh[1])


def convert_label_file(src_label: Path, dst_label: Path, kpt_count: int = 5) -> tuple[int, int]:
    if not src_label.exists():
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        dst_label.write_text("", encoding="utf-8")
        return 0, 0

    converted = 0
    skipped = 0
    out_lines: list[str] = []
    for raw in src_label.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue

        parsed = parse_seg_line(raw)
        if parsed is None:
            skipped += 1
            continue

        class_id, pts = parsed
        kpts = polygon_to_keypoints(pts, kpt_count=kpt_count)
        cx, cy, w, h = points_to_bbox(pts)

        # YOLO pose row: class cx cy w h (kpx kpy v)*N
        fields = [
            str(class_id),
            f"{cx:.6f}",
            f"{cy:.6f}",
            f"{w:.6f}",
            f"{h:.6f}",
        ]
        for kp in kpts:
            fields.append(f"{kp[0]:.6f}")
            fields.append(f"{kp[1]:.6f}")
            fields.append("2")
        out_lines.append(" ".join(fields))
        converted += 1

    dst_label.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(out_lines)
    if content:
        content += "\n"
    dst_label.write_text(content, encoding="utf-8")
    return converted, skipped


def ensure_images(src_images_dir: Path, dst_images_dir: Path, mode: str) -> None:
    dst_images_dir.parent.mkdir(parents=True, exist_ok=True)

    if dst_images_dir.exists() or dst_images_dir.is_symlink():
        if dst_images_dir.is_symlink() or dst_images_dir.is_file():
            dst_images_dir.unlink()
        else:
            shutil.rmtree(dst_images_dir)

    if mode == "symlink":
        dst_images_dir.symlink_to(src_images_dir.resolve(), target_is_directory=True)
    else:
        shutil.copytree(src_images_dir, dst_images_dir)


def write_data_yaml(dst_root: Path) -> None:
    data_yaml = dst_root / "data.yaml"
    text = (
        "# YOLO-pose dataset for rachis keypoints\n"
        f"path: {dst_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "nc: 1\n"
        "names:\n"
        "  0: rachis\n\n"
        "# 5 keypoints: base, 25%, middle, 75%, apex\n"
        "kpt_shape: [5, 3]\n"
        "flip_idx: [0, 1, 2, 3, 4]\n"
    )
    data_yaml.write_text(text, encoding="utf-8")


def convert_split(src_root: Path, dst_root: Path, split: str, kpt_count: int) -> tuple[int, int, int]:
    src_labels = src_root / "labels" / split
    src_images = src_root / "images" / split
    dst_labels = dst_root / "labels" / split

    converted_instances = 0
    skipped_instances = 0
    label_files = 0

    for src_txt in sorted(src_labels.glob("*.txt")):
        dst_txt = dst_labels / src_txt.name
        c, s = convert_label_file(src_txt, dst_txt, kpt_count=kpt_count)
        converted_instances += c
        skipped_instances += s
        label_files += 1

    # Also ensure images without labels have an empty label file.
    for image_path in sorted(src_images.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        txt_name = f"{image_path.stem}.txt"
        dst_txt = dst_labels / txt_name
        if not dst_txt.exists():
            dst_txt.parent.mkdir(parents=True, exist_ok=True)
            dst_txt.write_text("", encoding="utf-8")

    return label_files, converted_instances, skipped_instances


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training_rachis_pose from training_rachis")
    parser.add_argument("--src", default="training_rachis", help="Source YOLO-seg dataset root")
    parser.add_argument("--dst", default="training_rachis_pose", help="Destination YOLO-pose dataset root")
    parser.add_argument("--kpt-count", type=int, default=5, help="Number of keypoints per rachis")
    parser.add_argument(
        "--images-mode",
        choices=["symlink", "copy"],
        default="copy",
        help="Copy images by default (safe). Symlink mode is faster but can confuse label resolution.",
    )
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    for split in ("train", "val"):
        ensure_images(
            src_root / "images" / split,
            dst_root / "images" / split,
            mode=args.images_mode,
        )

    totals = {"files": 0, "converted": 0, "skipped": 0}
    for split in ("train", "val"):
        files, converted, skipped = convert_split(
            src_root, dst_root, split=split, kpt_count=args.kpt_count
        )
        totals["files"] += files
        totals["converted"] += converted
        totals["skipped"] += skipped
        print(
            f"[{split}] files={files}, instances_converted={converted}, instances_skipped={skipped}"
        )

    write_data_yaml(dst_root)
    print(f"Dataset pose genere: {dst_root}")
    print(f"Fichiers labels: {totals['files']}")
    print(f"Instances converties: {totals['converted']}")
    print(f"Instances ignorees: {totals['skipped']}")
    print(f"Config: {dst_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
