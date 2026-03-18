#!/usr/bin/env python3
"""
Extrait des ROIs d'épis depuis les images problématiques identifiées
dans output_bacs/_problematic_images, pré-annote avec les modèles YOLO
correspondants, et sauvegarde au format X-AnyLabeling pour correction manuelle.

Datasets ciblés :
  - training_spike_segmentation : images sans segmentation (07_no_segmentation)
  - training_rachis : images sans rachis détecté (08_no_rachis)
  - training_spikelets_segmentation : images avec faible comptage épillets (06_low_spikelet_count)

Usage :
    python scripts/extract_and_preannotate_problematic.py --dataset spike_seg
    python scripts/extract_and_preannotate_problematic.py --dataset rachis
    python scripts/extract_and_preannotate_problematic.py --dataset spikelets
    python scripts/extract_and_preannotate_problematic.py --dataset all
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO

# ───────────────────────── constantes ─────────────────────────
SPIKE_CLASS_ID = 1
MIN_ASPECT_RATIO = 1.5
MIN_CROP_SIDE = 80

DATASET_CONFIG = {
    "spike_seg": {
        "problem_category": "07_no_segmentation",
        "model_path": "models/spike_seg_yolo.pt",
        "dest": "training_spike_segmentation/tmp_new",
        "label_name": "spike",
        "shape_type": "polygon",  # segmentation = polygon
        "n_train": 160,
        "n_val": 40,
        "model_conf": 0.25,
        "margin": 50,
    },
    "rachis": {
        "problem_category": "08_no_rachis",
        "model_path": "models/rachis_yolo.pt",
        "dest": "training_rachis/tmp_new",
        "label_name": "rachis",
        "shape_type": "polygon",
        "n_train": 120,
        "n_val": 30,
        "model_conf": 0.25,
        "margin": 50,
    },
    "spikelets": {
        "problem_category": "06_low_spikelet_count",
        "model_path": "models/spikelets_yolo.pt",
        "dest": "training_spikelets_segmentation/tmp_new",
        "label_name": "spikelet",
        "shape_type": "polygon",
        "n_train": 160,
        "n_val": 40,
        "model_conf": 0.25,
        "margin": 30,
    },
}


# ───────────────────────── fonctions utilitaires ──────────────
def crop_obb_upright(img, xyxyxyxy, margin=50):
    """Crop redressé via warp-affine sur l'OBB orienté."""
    pts = xyxyxyxy.reshape(4, 2).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (rw, rh), angle = rect

    if rw > rh:
        rw, rh = rh, rw
        angle += 90.0

    if rh < 1 or rw < 1:
        return None
    if rh / rw < MIN_ASPECT_RATIO:
        return None
    if rw < MIN_CROP_SIDE:
        return None

    out_w = int(rw + 2 * margin)
    out_h = int(rh + 2 * margin)

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    M[0, 2] += out_w / 2 - cx
    M[1, 2] += out_h / 2 - cy

    crop = cv2.warpAffine(img, M, (out_w, out_h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))
    return crop


def mask_to_polygon(mask_xy, simplify_eps=1.5):
    """Convertit un masque YOLO (Nx2) en liste de [x, y]."""
    pts = mask_xy.astype(np.float32).reshape(-1, 1, 2)
    if simplify_eps > 0 and len(pts) > 6:
        pts = cv2.approxPolyDP(pts, simplify_eps, closed=True)
    return [[float(p[0][0]), float(p[0][1])] for p in pts]


def preannotate_crop_seg(model, crop, label_name, conf=0.25):
    """
    Lance un modèle YOLO-Seg sur un crop et retourne des shapes X-AnyLabeling.
    """
    results = model(crop, conf=conf, verbose=False)
    shapes = []

    for r in results:
        if r.masks is None or r.masks.xy is None:
            continue
        boxes = r.boxes
        for i, mask_xy in enumerate(r.masks.xy):
            if len(mask_xy) < 3:
                continue
            poly = mask_to_polygon(mask_xy, simplify_eps=1.5)
            if len(poly) < 3:
                continue
            score = float(boxes[i].conf.cpu().numpy().item()) if i < len(boxes) else None
            shapes.append({
                "label": label_name,
                "score": round(score, 4) if score else None,
                "points": poly,
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "polygon",
                "flags": {},
                "attributes": {},
            })

    return shapes


def preannotate_crop_detect(model, crop, label_name, conf=0.25):
    """
    Lance un modèle YOLO detect/seg sur un crop et retourne des shapes.
    Fallback pour les modèles sans masques (détection simple = rectangle).
    """
    results = model(crop, conf=conf, verbose=False)
    shapes = []

    for r in results:
        # Try segmentation masks first
        if r.masks is not None and r.masks.xy is not None:
            boxes = r.boxes
            for i, mask_xy in enumerate(r.masks.xy):
                if len(mask_xy) < 3:
                    continue
                poly = mask_to_polygon(mask_xy, simplify_eps=1.5)
                if len(poly) < 3:
                    continue
                score = float(boxes[i].conf.cpu().numpy().item()) if i < len(boxes) else None
                shapes.append({
                    "label": label_name,
                    "score": round(score, 4) if score else None,
                    "points": poly,
                    "group_id": None,
                    "description": "",
                    "difficult": False,
                    "shape_type": "polygon",
                    "flags": {},
                    "attributes": {},
                })
        # Fallback to bounding boxes
        elif r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = float(box.conf.cpu().numpy().item())
                shapes.append({
                    "label": label_name,
                    "score": round(score, 4),
                    "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                    "group_id": None,
                    "description": "",
                    "difficult": False,
                    "shape_type": "rectangle",
                    "flags": {},
                    "attributes": {},
                })

    return shapes


def build_xanylabeling_json(image_name, img_w, img_h, shapes):
    """Construit un JSON X-AnyLabeling / LabelMe."""
    return {
        "version": "3.3.5",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w,
    }


def get_problematic_images(category, root):
    """Lit la liste d'images problématiques depuis _summary.txt."""
    summary = root / "output" / "output_bacs" / "_problematic_images" / category / "_summary.txt"
    if not summary.exists():
        print(f"Erreur: {summary} n'existe pas")
        return []
    lines = summary.read_text().strip().split('\n')
    return [l.strip() for l in lines[3:] if l.strip()]


def get_existing_stems(training_dir):
    """Récupère les stems déjà dans un dataset d'entraînement."""
    existing = set()
    training_path = Path(training_dir)
    for split in ['train', 'val']:
        d = training_path / 'images' / split
        if d.exists():
            for f in d.iterdir():
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    parts = f.stem.split("_spike")
                    existing.add(parts[0])
    return existing


def run_dataset(dataset_key, root, seed=42):
    """Pipeline complet pour un dataset donné."""
    random.seed(seed)
    config = DATASET_CONFIG[dataset_key]
    
    print(f"\n{'='*60}")
    print(f"  DATASET: {dataset_key}")
    print(f"  Catégorie: {config['problem_category']}")
    print(f"{'='*60}")

    # Get problematic images
    problem_images = get_problematic_images(config['problem_category'], root)
    print(f"Images problématiques: {len(problem_images)}")

    # Get existing training stems to avoid duplicates
    training_base = config['dest'].split('/tmp_new')[0]
    existing_stems = get_existing_stems(root / training_base)
    print(f"Images source déjà dans le dataset: {len(existing_stems)}")

    # Filter new images
    new_images = [img for img in problem_images if img not in existing_stems]
    print(f"Nouvelles images à traiter: {len(new_images)}")

    if not new_images:
        print("Aucune nouvelle image. Terminé.")
        return

    # Load models
    obb_model_path = root / "models" / "wheat_spike_yolo.pt"
    seg_model_path = root / config['model_path']
    
    print(f"Chargement modèle OBB: {obb_model_path.name}")
    obb_model = YOLO(str(obb_model_path))
    print(f"Chargement modèle {dataset_key}: {seg_model_path.name}")
    seg_model = YOLO(str(seg_model_path))

    # Prepare output
    dest = root / config['dest']
    import shutil
    if dest.exists():
        shutil.rmtree(dest)

    n_train = config['n_train']
    n_val = config['n_val']
    n_total = n_train + n_val
    margin = config['margin']
    model_conf = config['model_conf']
    label_name = config['label_name']

    # Extract ROIs
    random.shuffle(new_images)
    candidates = []

    for img_name in tqdm(new_images, desc="Extraction ROIs"):
        # Find raw image
        img_path = None
        for ext in ['.JPG', '.jpg', '.jpeg', '.png']:
            candidate = root / 'data' / 'raw' / (img_name + ext)
            if candidate.exists():
                img_path = candidate
                break
        
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = obb_model(img, conf=0.5, verbose=False)
        if not results or results[0].obb is None:
            continue

        obb_data = results[0].obb
        classes = obb_data.cls.cpu().numpy().astype(int)
        confs = obb_data.conf.cpu().numpy()
        xyxyxyxy = obb_data.xyxyxyxy.cpu().numpy()

        for i in range(len(obb_data)):
            if classes[i] != SPIKE_CLASS_ID:
                continue

            crop = crop_obb_upright(img, xyxyxyxy[i], margin=margin)
            if crop is None:
                continue

            # Pre-annotate
            shapes = preannotate_crop_detect(seg_model, crop, label_name, conf=model_conf)

            crop_name = f"{img_name}_spike{i:02d}"
            candidates.append((float(confs[i]), crop, crop_name, shapes))

        if len(candidates) >= n_total * 3:
            break

    print(f"\nCandidats extraits: {len(candidates)}")
    total_shapes = sum(len(c[3]) for c in candidates)
    print(f"Annotations pré-générées: {total_shapes}")

    if not candidates:
        print("Aucun crop extrait !")
        return

    # Select top-N by confidence, then split
    candidates.sort(key=lambda x: x[0], reverse=True)

    if len(candidates) < n_total:
        n_total = len(candidates)
        n_val = max(1, int(n_total * 0.2))
        n_train = n_total - n_val

    selected = candidates[:n_total]
    random.shuffle(selected)
    val_sel = selected[:n_val]
    train_sel = selected[n_val:n_val + n_train]

    # Save
    for split_name, items in [('train', train_sel), ('val', val_sel)]:
        out_dir = dest / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for conf_score, crop, name, shapes in items:
            img_file = out_dir / f"{name}.jpg"
            json_file = out_dir / f"{name}.json"

            cv2.imwrite(str(img_file), crop)
            h, w = crop.shape[:2]
            anno = build_xanylabeling_json(img_file.name, w, h, shapes)
            with open(json_file, "w") as f:
                json.dump(anno, f, indent=2)

    print(f"\n✅ {dataset_key} terminé:")
    print(f"   train: {len(train_sel)} crops → {dest / 'train'}")
    print(f"   val:   {len(val_sel)} crops → {dest / 'val'}")
    print(f"   Annotations totales: train={sum(len(c[3]) for c in train_sel)}, val={sum(len(c[3]) for c in val_sel)}")


def main():
    parser = argparse.ArgumentParser(
        description="Extrait des ROIs problématiques et pré-annote pour correction"
    )
    parser.add_argument("--dataset", "-d", required=True,
                        choices=["spike_seg", "rachis", "spikelets", "all"],
                        help="Dataset à préparer")
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    if args.dataset == "all":
        for key in ["spike_seg", "rachis", "spikelets"]:
            run_dataset(key, root, seed=args.seed)
    else:
        run_dataset(args.dataset, root, seed=args.seed)


if __name__ == "__main__":
    main()
