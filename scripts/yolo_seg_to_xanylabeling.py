#!/usr/bin/env python3
"""Convert YOLO segmentation annotations to X-AnyLabeling (LabelMe) JSON format.

Reads YOLO polygon segmentation labels from training_spikelets_segmentation/labels/
and writes JSON files alongside images in training_spikelets_segmentation/images/.
"""

import json
import os
from pathlib import Path
from PIL import Image

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent / "training_spikelets_segmentation"
IMAGES_DIR = BASE_DIR / "images"
LABELS_DIR = BASE_DIR / "labels"

# Class mapping (from data.yaml)
CLASS_NAMES = {0: "spikelet"}


def yolo_seg_to_json(label_path: Path, image_path: Path) -> dict:
    """Convert a YOLO segmentation label file to X-AnyLabeling JSON format."""
    # Get image dimensions
    with Image.open(image_path) as img:
        width, height = img.size

    shapes = []
    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:  # class_id + at least 3 points (6 coords)
                    continue

                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))

                # Convert normalized coords to pixel coords
                points = []
                for i in range(0, len(coords), 2):
                    x = coords[i] * width
                    y = coords[i + 1] * height
                    points.append([x, y])

                shapes.append({
                    "label": CLASS_NAMES.get(class_id, f"class_{class_id}"),
                    "points": points,
                    "group_id": None,
                    "description": "",
                    "shape_type": "polygon",
                    "flags": {},
                    "mask": None,
                })

    return {
        "version": "5.10.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def process_split(split: str):
    """Process a single split (train or val)."""
    img_dir = IMAGES_DIR / split
    lbl_dir = LABELS_DIR / split

    if not img_dir.exists():
        print(f"  Skipping {split}: {img_dir} not found")
        return

    image_files = sorted(img_dir.glob("*.jpg"))
    created = 0
    skipped = 0

    for img_path in image_files:
        json_path = img_path.with_suffix(".json")
        label_path = lbl_dir / img_path.with_suffix(".txt").name

        data = yolo_seg_to_json(label_path, img_path)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        if data["shapes"]:
            created += 1
        else:
            skipped += 1

    print(f"  {split}: {created} annotated + {skipped} empty = {created + skipped} total JSON files")


def main():
    print(f"Base directory: {BASE_DIR}")
    print(f"Converting YOLO segmentation annotations to X-AnyLabeling JSON...\n")

    for split in ["train", "val"]:
        print(f"Processing {split}...")
        process_split(split)

    print("\nDone!")


if __name__ == "__main__":
    main()
