#!/usr/bin/env python3
"""
Script d'entraînement YOLO-Seg pour la segmentation du rachis.

Usage:
    python scripts/train_spike.py
    python scripts/train_spike.py --resume
    python scripts/train_spike.py --epochs 300 --batch 8 --imgsz 640
"""

import argparse
from ultralytics import YOLO


# =============================================================================
# CONFIGURATION
# =============================================================================

# Modèle de base (pré-entraîné)
MODEL = "yolo26s-obb.pt"

# Dataset
DATA = "training_obb_angled/data.yaml"

# Hyperparamètres d'entraînement
EPOCHS = 500
BATCH = 16
IMGSZ = 1024
PATIENCE = 50

# GPU / Hardware
DEVICE = 0
WORKERS = 8
AMP = True

# Sortie
PROJECT = "runs/"
NAME = "spike_yolo26s_obb"
EXIST_OK = True

# Optimiseur (auto = AdamW avec lr auto)
OPTIMIZER = "auto"
LR0 = 0.01
LRF = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005

# Augmentations (modérées, avec rotation)
HSV_H = 0.01
HSV_S = 0.3
HSV_V = 0.2
DEGREES = 180    # Rotation ±15° (couvre les inclinaisons réelles des épis)
TRANSLATE = 0.1
SCALE = 0         # Zoom modéré ±30%
SHEAR = 0       # Cisaillement léger ±2°
FLIPUD = 0        # Flip vertical (l'épi peut être dans les 2 sens)
FLIPLR = 0         # Flip horizontal
MOSAIC = 0
MIXUP = 0.0          # Désactivé (peu utile pour segmentation fine)
CLOSE_MOSAIC = 10

# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO-Seg spike model")
    parser.add_argument("--model", default=MODEL, help=f"Base model (default: {MODEL})")
    parser.add_argument("--data", default=DATA, help=f"Dataset config (default: {DATA})")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=float, default=BATCH,
                        help="Batch size (int) or GPU fraction (0-1)")
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--device", default=str(DEVICE))
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--name", default=NAME)
    parser.add_argument("--resume", action="store_true", help="Resume last training")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    return parser.parse_args()


def main():
    args = parse_args()

    # Charger le modèle
    if args.resume:
        weights = f"{args.project}/{args.name}/weights/last.pt"
        print(f"Reprise depuis: {weights}")
        model = YOLO(weights)
    else:
        print(f"Modèle de base: {args.model}")
        model = YOLO(args.model)

    # Batch: int ou float selon la valeur
    batch = int(args.batch) if args.batch >= 1 else args.batch

    # Lancer l'entraînement
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=EXIST_OK,
        pretrained=True,
        amp=not args.no_amp,
        # Optimiseur
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        # Augmentations
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        shear=SHEAR,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        mosaic=MOSAIC,
        mixup=MIXUP,
        close_mosaic=CLOSE_MOSAIC,
        # Resume
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
