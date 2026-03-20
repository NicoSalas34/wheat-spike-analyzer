#!/usr/bin/env python3
"""
Équilibre le dataset training_bag_opening en conservant les familles d'images:
base image + ses augmentations (_rotaug*).

Actions:
  - regroupe par stem de base
  - redistribue train/val selon un ratio cible (défaut 0.8)
  - déplace images + JSON + labels ensemble
  - supprime les labels orphelins
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path


ROT_PATTERN = re.compile(r"^(?P<base>.+)_rotaug\d+$")
IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def base_stem(stem: str) -> str:
    m = ROT_PATTERN.match(stem)
    return m.group("base") if m else stem


def stems_in_dir(images_dir: Path) -> set[str]:
    stems: set[str] = set()
    for ext in IMG_EXTS:
        stems |= {p.stem for p in images_dir.glob(f"*{ext}")}
    return stems


def family_members(stems: set[str]) -> dict[str, list[str]]:
    fam: dict[str, list[str]] = {}
    for s in stems:
        b = base_stem(s)
        fam.setdefault(b, []).append(s)
    return fam


def move_stem(dataset: Path, stem: str, src: str, dst: str) -> int:
    moved = 0

    src_img = dataset / "images" / src
    dst_img = dataset / "images" / dst
    src_lbl = dataset / "labels" / src
    dst_lbl = dataset / "labels" / dst

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    for ext in IMG_EXTS:
        p = src_img / f"{stem}{ext}"
        if p.exists():
            shutil.move(str(p), str(dst_img / p.name))
            moved += 1
            break

    p_json = src_img / f"{stem}.json"
    if p_json.exists():
        shutil.move(str(p_json), str(dst_img / p_json.name))

    p_txt = src_lbl / f"{stem}.txt"
    if p_txt.exists():
        shutil.move(str(p_txt), str(dst_lbl / p_txt.name))

    return moved


def cleanup_orphan_labels(dataset: Path) -> tuple[int, int]:
    deleted_train = 0
    deleted_val = 0

    for split in ["train", "val"]:
        img_stems = stems_in_dir(dataset / "images" / split)
        lbl_dir = dataset / "labels" / split
        for txt in lbl_dir.glob("*.txt"):
            if txt.stem not in img_stems:
                txt.unlink()
                if split == "train":
                    deleted_train += 1
                else:
                    deleted_val += 1

    return deleted_train, deleted_val


def run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[1]
    dataset = root / args.dataset

    train_img = dataset / "images" / "train"
    val_img = dataset / "images" / "val"

    train_stems = stems_in_dir(train_img)
    val_stems = stems_in_dir(val_img)
    all_stems = train_stems | val_stems

    fam = family_members(all_stems)
    base_keys = sorted(fam.keys())

    rng = random.Random(args.seed)
    rng.shuffle(base_keys)

    target_train_bases = int(round(len(base_keys) * args.train_ratio))
    desired_train = set(base_keys[:target_train_bases])

    moved_images = 0

    for base, members in fam.items():
        # Déterminer split actuel de la famille à partir du premier membre image trouvé
        current_split = None
        for stem in members:
            if stem in train_stems:
                current_split = "train"
                break
            if stem in val_stems:
                current_split = "val"
                break

        if current_split is None:
            continue

        desired_split = "train" if base in desired_train else "val"
        if current_split == desired_split:
            continue

        for stem in members:
            moved_images += move_stem(dataset, stem, current_split, desired_split)

    deleted_train, deleted_val = cleanup_orphan_labels(dataset)

    # Comptages finaux
    final_train = len(stems_in_dir(dataset / "images" / "train"))
    final_val = len(stems_in_dir(dataset / "images" / "val"))
    final_train_lbl = len(list((dataset / "labels" / "train").glob("*.txt")))
    final_val_lbl = len(list((dataset / "labels" / "val").glob("*.txt")))

    print("✅ Équilibrage terminé")
    print(f"Dataset: {dataset}")
    print(f"Familles (bases): {len(base_keys)}")
    print(f"Ratio cible train: {args.train_ratio:.2f}")
    print(f"Images déplacées: {moved_images}")
    print(f"Labels orphelins supprimés train/val: {deleted_train}/{deleted_val}")
    print(f"Final images train/val: {final_train}/{final_val}")
    print(f"Final labels train/val: {final_train_lbl}/{final_val_lbl}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Équilibrer training_bag_opening")
    parser.add_argument("--dataset", default="training_bag_opening", type=str)
    parser.add_argument("--train-ratio", default=0.8, type=float)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
