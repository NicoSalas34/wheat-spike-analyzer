#!/usr/bin/env python3
"""
Extrait de nouveaux ROI de sachets depuis data/raw et les ajoute à
training_bag_opening pour annotation manuelle.

Le script:
  - utilise le modèle OBB principal (classe 'bag' = id 2)
  - évite les doublons déjà présents dans le dataset
  - redresse chaque sachet (warp affine sur OBB) + marge
  - écrit les images dans images/train et images/val
  - crée des labels YOLO vides dans labels/train et labels/val

Exemple:
  python scripts/extract_bag_opening_rois.py --target-new 240
"""

import argparse
import random
from pathlib import Path

import cv2
from ultralytics import YOLO


BAG_CLASS_ID = 2


def crop_obb_upright(img, xyxyxyxy, margin=24, min_side=80):
    """Retourne un crop redressé du sachet (ou None si trop petit)."""
    pts = xyxyxyxy.reshape(4, 2).astype("float32")
    rect = cv2.minAreaRect(pts)
    (cx, cy), (rw, rh), angle = rect

    if rw < 1 or rh < 1:
        return None

    # Conserver le grand côté en hauteur pour homogénéiser les ROI
    if rw > rh:
        rw, rh = rh, rw
        angle += 90.0

    if min(rw, rh) < min_side:
        return None

    out_w = int(rw + 2 * margin)
    out_h = int(rh + 2 * margin)

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    M[0, 2] += out_w / 2 - cx
    M[1, 2] += out_h / 2 - cy

    crop = cv2.warpAffine(
        img,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return crop


def get_existing_names(dataset_root: Path):
    """Noms d'images déjà présents (sans extension) pour éviter les doublons."""
    existing = set()
    for split in ("train", "val"):
        img_dir = dataset_root / "images" / split
        if not img_dir.exists():
            continue
        for f in img_dir.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                existing.add(f.stem)
    return existing


def get_raw_images(raw_dir: Path):
    exts = {".jpg", ".jpeg", ".png"}
    files = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def run(args):
    root = Path(__file__).resolve().parents[1]
    dataset = root / "training_bag_opening"
    raw_dir = root / "data" / "raw"
    model_path = root / args.model

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")
    if not raw_dir.exists():
        raise FileNotFoundError(f"Dossier data/raw introuvable: {raw_dir}")

    random.seed(args.seed)

    existing = get_existing_names(dataset)
    raw_images = get_raw_images(raw_dir)
    random.shuffle(raw_images)

    print(f"Images raw trouvées: {len(raw_images)}")
    print(f"ROI déjà présents: {len(existing)}")

    model = YOLO(str(model_path))

    candidates = []
    for img_path in raw_images:
        stem = img_path.stem

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model(img, conf=args.conf, verbose=False)
        if not results or results[0].obb is None:
            continue

        obb = results[0].obb
        classes = obb.cls.cpu().numpy().astype(int)
        confs = obb.conf.cpu().numpy()
        polys = obb.xyxyxyxy.cpu().numpy()

        for idx in range(len(obb)):
            if classes[idx] != BAG_CLASS_ID:
                continue

            name = f"{stem}_bag{idx}"
            if name in existing:
                continue

            crop = crop_obb_upright(
                img,
                polys[idx],
                margin=args.margin,
                min_side=args.min_side,
            )
            if crop is None:
                continue

            candidates.append((float(confs[idx]), name, crop))

        if len(candidates) >= args.target_new * 4:
            break

    if not candidates:
        print("Aucun nouveau ROI candidate trouvé.")
        return 0

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = candidates[: args.target_new]
    random.shuffle(selected)

    n_val = max(1, int(round(len(selected) * args.val_ratio)))
    n_train = len(selected) - n_val

    splits = [
        ("val", selected[:n_val]),
        ("train", selected[n_val:n_val + n_train]),
    ]

    written = 0
    for split, items in splits:
        img_dir = dataset / "images" / split
        lbl_dir = dataset / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for _, name, crop in items:
            img_file = img_dir / f"{name}.jpg"
            lbl_file = lbl_dir / f"{name}.txt"

            if img_file.exists():
                continue

            ok = cv2.imwrite(str(img_file), crop)
            if not ok:
                continue

            lbl_file.write_text("")
            written += 1

    print("\n✅ Extraction terminée")
    print(f"Candidats: {len(candidates)}")
    print(f"Sélectionnés: {len(selected)}")
    print(f"Ajoutés: {written}")
    print(f"Train ajoutés: {n_train}")
    print(f"Val ajoutés: {n_val}")
    print("\nAnnotation ensuite:")
    print(f"  labelme {dataset / 'images' / 'train'}")
    print(f"  labelme {dataset / 'images' / 'val'}")

    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Extraction de ROI bag_opening")
    parser.add_argument("--target-new", type=int, default=240,
                        help="Nombre de nouveaux ROI à ajouter (défaut: 240)")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Part de validation (défaut: 0.2)")
    parser.add_argument("--model", type=str, default="models/wheat_spike_yolo.pt",
                        help="Modèle OBB principal")
    parser.add_argument("--conf", type=float, default=0.45,
                        help="Seuil confiance OBB")
    parser.add_argument("--margin", type=int, default=24,
                        help="Marge autour du crop redressé")
    parser.add_argument("--min-side", type=int, default=80,
                        help="Taille minimale du petit côté du sachet")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
