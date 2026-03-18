#!/usr/bin/env python3
"""Convert X-AnyLabeling rotation annotations to YOLO OBB format
and copy images + labels into training_obb_angled/{images,labels}/{train,val}.
"""

import json
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / "training_obb_angled"
TMP_NEW = BASE_DIR / "tmp_new"

# Class mapping from data.yaml
CLASS_MAP = {
    "ruler": 0,
    "spike": 1,
    "bag": 2,
    "whole_spike": 3,
}


def convert_json_to_yolo_obb(json_path: Path) -> str:
    """Convert a single X-AnyLabeling JSON to YOLO OBB label text."""
    with open(json_path, "r") as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]
    lines = []

    for shape in data.get("shapes", []):
        label = shape["label"]
        if label not in CLASS_MAP:
            print(f"  Warning: unknown label '{label}' in {json_path.name}, skipping")
            continue

        class_id = CLASS_MAP[label]
        points = shape["points"]

        if len(points) != 4:
            print(f"  Warning: expected 4 points, got {len(points)} in {json_path.name}, skipping")
            continue

        # Normalize coordinates
        coords = []
        for x, y in points:
            coords.append(f"{max(0, min(1, x / img_w)):.6f}")
            coords.append(f"{max(0, min(1, y / img_h)):.6f}")

        lines.append(f"{class_id} {' '.join(coords)}")

    return "\n".join(lines)


def process_split(split: str):
    """Process train or val split."""
    src_dir = TMP_NEW / split
    dst_images = BASE_DIR / "images" / split
    dst_labels = BASE_DIR / "labels" / split

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    json_files = sorted(src_dir.glob("*.json"))
    print(f"Processing {split}: {len(json_files)} annotations found")

    copied = 0
    skipped = 0
    for json_path in json_files:
        stem = json_path.stem
        # Find corresponding image
        img_path = None
        for ext in [".JPG", ".jpg", ".png", ".PNG", ".jpeg", ".JPEG"]:
            candidate = src_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"  No image found for {json_path.name}, skipping")
            skipped += 1
            continue

        # Convert annotation
        label_text = convert_json_to_yolo_obb(json_path)
        if not label_text.strip():
            print(f"  No valid annotations in {json_path.name}, skipping")
            skipped += 1
            continue

        # Copy image
        shutil.copy2(img_path, dst_images / img_path.name)

        # Write label
        label_path = dst_labels / (stem + ".txt")
        with open(label_path, "w") as f:
            f.write(label_text + "\n")

        copied += 1

    print(f"  {copied} images+labels copied, {skipped} skipped")


def main():
    for split in ["train", "val"]:
        if (TMP_NEW / split).exists():
            process_split(split)
        else:
            print(f"Directory {TMP_NEW / split} not found, skipping")

    # Remove stale cache files
    for cache in BASE_DIR.glob("labels/*/*.cache"):
        cache.unlink()
        print(f"Removed cache: {cache}")

    print("Done!")


if __name__ == "__main__":
    main()
