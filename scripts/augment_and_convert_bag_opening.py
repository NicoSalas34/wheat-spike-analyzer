#!/usr/bin/env python3
"""
Augmente un dataset X-AnyLabeling (LabelMe JSON) par rotations aléatoires,
puis convertit les annotations au format YOLO détection (bbox).

Cible par défaut:
  training_bag_opening/images/{train,val}
  training_bag_opening/labels/{train,val}

Fonctionnement:
  1) Pour chaque image avec JSON, crée N augmentations (défaut: 3)
     avec angle uniforme dans [0, 180).
  2) Applique la même transformation affine aux points d'annotation.
  3) Sauvegarde image + JSON augmentés dans le même split.
  4) Convertit tous les JSON (originaux + augmentés) en labels YOLO bbox.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CLASS_MAP = {
    "bag_opening": 0,
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def list_image_json_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(split_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix not in IMG_EXTS:
            continue
        if "_rotaug" in img_path.stem:
            continue
        json_path = split_dir / f"{img_path.stem}.json"
        if json_path.exists():
            pairs.append((img_path, json_path))
    return pairs


def ensure_points_from_shape(shape: dict[str, Any]) -> list[list[float]]:
    pts = shape.get("points", [])
    if not isinstance(pts, list) or len(pts) == 0:
        return []

    shape_type = shape.get("shape_type", "polygon")
    if shape_type == "rectangle" and len(pts) == 2:
        (x1, y1), (x2, y2) = pts
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

    return [[float(x), float(y)] for x, y in pts]


def rotate_image_keep_all(img: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy

    out = cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return out, M


def transform_points(points: list[list[float]], M: np.ndarray, w: int, h: int) -> list[list[float]]:
    if not points:
        return []

    arr = np.array(points, dtype=np.float32)
    ones = np.ones((arr.shape[0], 1), dtype=np.float32)
    homo = np.hstack([arr, ones])
    tr = (M @ homo.T).T

    tr[:, 0] = np.clip(tr[:, 0], 0, w - 1)
    tr[:, 1] = np.clip(tr[:, 1], 0, h - 1)
    return tr.tolist()


def shape_to_yolo_bbox_line(shape: dict[str, Any], img_w: int, img_h: int) -> str | None:
    label = shape.get("label")
    if label not in CLASS_MAP:
        return None

    pts = ensure_points_from_shape(shape)
    if len(pts) == 0:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    xmin = max(0.0, min(xs))
    xmax = min(float(img_w - 1), max(xs))
    ymin = max(0.0, min(ys))
    ymax = min(float(img_h - 1), max(ys))

    bw = xmax - xmin
    bh = ymax - ymin
    if bw <= 1 or bh <= 1:
        return None

    xc = (xmin + xmax) / 2.0 / img_w
    yc = (ymin + ymax) / 2.0 / img_h
    w = bw / img_w
    h = bh / img_h

    class_id = CLASS_MAP[label]
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def augment_split(images_dir: Path, per_image: int, seed: int) -> tuple[int, int]:
    rng = random.Random(seed)
    pairs = list_image_json_pairs(images_dir)

    generated = 0
    skipped = 0

    for img_path, json_path in pairs:
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        shapes = data.get("shapes", [])

        for i in range(1, per_image + 1):
            angle = rng.uniform(0.0, 180.0)
            aug_img, M = rotate_image_keep_all(img, angle)
            ah, aw = aug_img.shape[:2]

            aug_stem = f"{img_path.stem}_rotaug{i}"
            aug_img_name = f"{aug_stem}.jpg"
            aug_json_name = f"{aug_stem}.json"

            aug_img_path = images_dir / aug_img_name
            aug_json_path = images_dir / aug_json_name

            if aug_img_path.exists() or aug_json_path.exists():
                skipped += 1
                continue

            ok = cv2.imwrite(str(aug_img_path), aug_img)
            if not ok:
                skipped += 1
                continue

            aug_shapes: list[dict[str, Any]] = []
            for shape in shapes:
                pts = ensure_points_from_shape(shape)
                if not pts:
                    continue
                tpts = transform_points(pts, M, aw, ah)
                if len(tpts) < 2:
                    continue

                aug_shape = {
                    "label": shape.get("label", ""),
                    "points": tpts,
                    "group_id": shape.get("group_id"),
                    "description": shape.get("description", ""),
                    "shape_type": "polygon" if len(tpts) >= 3 else "rectangle",
                    "flags": shape.get("flags", {}),
                    "mask": None,
                }
                aug_shapes.append(aug_shape)

            aug_data = {
                "version": data.get("version", "5.10.1"),
                "flags": data.get("flags", {}),
                "shapes": aug_shapes,
                "imagePath": aug_img_name,
                "imageData": None,
                "imageHeight": ah,
                "imageWidth": aw,
            }

            with open(aug_json_path, "w", encoding="utf-8") as f:
                json.dump(aug_data, f, ensure_ascii=False, indent=2)

            generated += 1

    return generated, skipped


def convert_split_to_yolo(images_dir: Path, labels_dir: Path) -> tuple[int, int]:
    labels_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(images_dir.glob("*.json"))
    written = 0
    empty = 0

    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_w = int(data.get("imageWidth", 0))
        img_h = int(data.get("imageHeight", 0))
        if img_w <= 0 or img_h <= 0:
            continue

        lines: list[str] = []
        for shape in data.get("shapes", []):
            line = shape_to_yolo_bbox_line(shape, img_w, img_h)
            if line is not None:
                lines.append(line)

        out_label = labels_dir / f"{json_path.stem}.txt"
        out_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        written += 1
        if not lines:
            empty += 1

    return written, empty


def run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[1]
    dataset = root / args.dataset

    train_images = dataset / "images" / "train"
    val_images = dataset / "images" / "val"
    train_labels = dataset / "labels" / "train"
    val_labels = dataset / "labels" / "val"

    if not train_images.exists() or not val_images.exists():
        raise FileNotFoundError(f"Images train/val introuvables dans: {dataset}")

    print(f"Dataset: {dataset}")
    print(f"Augmentations par image: {args.per_image}")
    print(f"Seed: {args.seed}\n")

    g_train, s_train = augment_split(train_images, args.per_image, args.seed)
    g_val, s_val = augment_split(val_images, args.per_image, args.seed + 1)

    y_train, e_train = convert_split_to_yolo(train_images, train_labels)
    y_val, e_val = convert_split_to_yolo(val_images, val_labels)

    print("✅ Terminé")
    print(f"Augmentations train créées: {g_train} (skipped: {s_train})")
    print(f"Augmentations val créées:   {g_val} (skipped: {s_val})")
    print(f"Labels YOLO train écrits:   {y_train} (vides: {e_train})")
    print(f"Labels YOLO val écrits:     {y_val} (vides: {e_val})")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augmentation rotation + conversion X-AnyLabeling -> YOLO (bbox)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="training_bag_opening",
        help="Chemin du dataset relatif à la racine du projet",
    )
    parser.add_argument(
        "--per-image",
        type=int,
        default=3,
        help="Nombre d'images augmentées par image source",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
