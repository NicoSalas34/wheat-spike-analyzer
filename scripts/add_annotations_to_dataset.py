#!/usr/bin/env python3
"""
Script pour convertir les annotations JSON (format X-Anylabeling/LabelMe) 
en format YOLO OBB et les ajouter au dataset d'entraînement.
"""

import json
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


# Mapping des classes
CLASS_MAPPING = {
    "ruler": 0,
    "spike": 1,
    "bag": 2,
    "whole_spike": 3
}


def normalize_obb_points(points: list, img_width: int, img_height: int) -> list:
    """
    Normalise les coordonnées des 4 coins par rapport aux dimensions de l'image.
    
    Args:
        points: Liste de 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        img_width: Largeur de l'image
        img_height: Hauteur de l'image
    
    Returns:
        Liste de coordonnées normalisées [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    normalized = []
    for point in points:
        x_norm = point[0] / img_width
        y_norm = point[1] / img_height
        # Clamp values to [0, 1]
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        normalized.extend([x_norm, y_norm])
    return normalized


def json_to_yolo_obb(json_path: Path, output_label_path: Path, img_width: int, img_height: int) -> int:
    """
    Convertit un fichier JSON LabelMe en format YOLO OBB.
    
    Args:
        json_path: Chemin du fichier JSON
        output_label_path: Chemin du fichier label de sortie (.txt)
        img_width: Largeur de l'image
        img_height: Hauteur de l'image
    
    Returns:
        Nombre d'annotations converties
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    lines = []
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        if label not in CLASS_MAPPING:
            print(f"Warning: Unknown label '{label}' in {json_path.name}")
            continue
        
        class_id = CLASS_MAPPING[label]
        points = shape.get("points", [])
        
        if len(points) != 4:
            print(f"Warning: Expected 4 points, got {len(points)} in {json_path.name}")
            continue
        
        # Normaliser les coordonnées
        normalized = normalize_obb_points(points, img_width, img_height)
        
        # Format YOLO OBB: class x1 y1 x2 y2 x3 y3 x4 y4
        line = f"{class_id} " + " ".join(f"{v:.6f}" for v in normalized)
        lines.append(line)
    
    # Écrire le fichier label
    with open(output_label_path, 'w') as f:
        f.write("\n".join(lines))
    
    return len(lines)


def add_to_dataset(
    source_dir: str,
    target_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42
):
    """
    Ajoute les images annotées au dataset d'entraînement YOLO OBB.
    
    Args:
        source_dir: Répertoire source contenant les images et JSON
        target_dir: Répertoire cible (training_obb_angled)
        train_ratio: Ratio train/val (défaut: 80% train, 20% val)
        seed: Graine aléatoire pour la répartition
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Créer les répertoires cibles si nécessaires
    (target_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (target_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (target_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (target_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Trouver toutes les images avec JSON correspondant
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = []
    
    for img_file in source_path.iterdir():
        if img_file.suffix in image_extensions:
            json_file = source_path / f"{img_file.stem}.json"
            if json_file.exists():
                image_files.append((img_file, json_file))
    
    print(f"Trouvé {len(image_files)} images avec annotations")
    
    # Mélanger et répartir
    random.seed(seed)
    random.shuffle(image_files)
    
    n_train = int(len(image_files) * train_ratio)
    train_files = image_files[:n_train]
    val_files = image_files[n_train:]
    
    print(f"Répartition: {len(train_files)} train, {len(val_files)} val")
    
    # Traiter les fichiers
    stats = {"train": 0, "val": 0, "annotations": 0}
    
    for split_name, files in [("train", train_files), ("val", val_files)]:
        for img_file, json_file in tqdm(files, desc=f"Processing {split_name}"):
            # Lire les dimensions de l'image
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"Warning: Cannot read {img_file.name}")
                continue
            
            height, width = img.shape[:2]
            
            # Chemin de destination (utiliser .jpg en minuscules)
            dest_img = target_path / "images" / split_name / f"{img_file.stem}.jpg"
            dest_label = target_path / "labels" / split_name / f"{img_file.stem}.txt"
            
            # Copier l'image
            shutil.copy2(img_file, dest_img)
            
            # Convertir et sauvegarder le label
            n_annotations = json_to_yolo_obb(json_file, dest_label, width, height)
            
            stats[split_name] += 1
            stats["annotations"] += n_annotations
    
    print(f"\n✅ Dataset mis à jour:")
    print(f"   - Images ajoutées au train: {stats['train']}")
    print(f"   - Images ajoutées au val: {stats['val']}")
    print(f"   - Total annotations: {stats['annotations']}")
    
    # Supprimer les caches existants
    cache_files = list(target_path.glob("labels/*.cache"))
    for cache in cache_files:
        cache.unlink()
        print(f"   - Cache supprimé: {cache.name}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ajoute des images annotées au dataset YOLO OBB"
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="Répertoire source avec images et JSON"
    )
    parser.add_argument(
        "--target", "-t",
        type=str,
        required=True,
        help="Répertoire cible (training_obb_angled)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio train/val (défaut: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire"
    )
    
    args = parser.parse_args()
    
    add_to_dataset(
        source_dir=args.source,
        target_dir=args.target,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
