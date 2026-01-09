#!/usr/bin/env python3
"""
Script pour pré-annoter des images avec un modèle YOLO OBB.
Génère des fichiers JSON au format X-Anylabeling/LabelMe avec des boîtes orientées.
"""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm


def obb_to_labelme_format(results, image_path: Path, class_names: dict) -> dict:
    """
    Convertit les résultats OBB YOLO en format LabelMe/X-Anylabeling.
    
    Args:
        results: Résultats de prédiction YOLO OBB
        image_path: Chemin vers l'image
        class_names: Dictionnaire des noms de classes {id: name}
    
    Returns:
        Dictionnaire au format LabelMe
    """
    # Lire les dimensions de l'image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Impossible de lire l'image: {image_path}")
    
    height, width = img.shape[:2]
    
    # Structure de base LabelMe
    labelme_data = {
        "version": "3.3.4",
        "flags": {},
        "shapes": [],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }
    
    # Traiter les résultats OBB
    if results[0].obb is not None and len(results[0].obb) > 0:
        obb = results[0].obb
        
        # Obtenir les 4 coins des boîtes orientées (xyxyxyxy format)
        if hasattr(obb, 'xyxyxyxy'):
            boxes_corners = obb.xyxyxyxy.cpu().numpy()  # Shape: (N, 4, 2)
        else:
            print("Warning: OBB results don't have xyxyxyxy attribute")
            return labelme_data
        
        classes = obb.cls.cpu().numpy()
        confidences = obb.conf.cpu().numpy()
        
        for i, (corners, cls_id, conf) in enumerate(zip(boxes_corners, classes, confidences)):
            cls_name = class_names.get(int(cls_id), f"class_{int(cls_id)}")
            
            # Convertir les coins en liste de points
            points = corners.tolist()  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            
            shape = {
                "label": cls_name,
                "score": float(conf),
                "points": points,
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "rotation",
                "flags": {},
                "attributes": {},
                "kie_linking": [],
                "direction": 0
            }
            labelme_data["shapes"].append(shape)
    
    return labelme_data


def preannotate_images(
    model_path: str,
    images_dir: str,
    output_dir: str = None,
    conf_threshold: float = 0.25,
    overwrite: bool = False
):
    """
    Pré-annote les images avec un modèle YOLO OBB.
    
    Args:
        model_path: Chemin vers le modèle YOLO OBB
        images_dir: Répertoire contenant les images
        output_dir: Répertoire de sortie (par défaut: même que images_dir)
        conf_threshold: Seuil de confiance minimum
        overwrite: Écraser les annotations existantes
    """
    # Charger le modèle
    print(f"Chargement du modèle: {model_path}")
    model = YOLO(model_path)
    
    # Obtenir les noms de classes du modèle
    class_names = model.names
    print(f"Classes du modèle: {class_names}")
    
    # Répertoires
    images_path = Path(images_dir)
    output_path = Path(output_dir) if output_dir else images_path
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Trouver toutes les images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG'}
    image_files = [f for f in images_path.iterdir() 
                   if f.suffix in image_extensions]
    
    print(f"Trouvé {len(image_files)} images à annoter")
    
    # Traiter chaque image
    annotated_count = 0
    skipped_count = 0
    
    for image_file in tqdm(image_files, desc="Pré-annotation"):
        json_path = output_path / f"{image_file.stem}.json"
        
        # Vérifier si l'annotation existe déjà
        if json_path.exists() and not overwrite:
            skipped_count += 1
            continue
        
        try:
            # Faire la prédiction
            results = model.predict(
                source=str(image_file),
                conf=conf_threshold,
                verbose=False
            )
            
            # Convertir en format LabelMe
            labelme_data = obb_to_labelme_format(results, image_file, class_names)
            
            # Sauvegarder le JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(labelme_data, f, indent=2, ensure_ascii=False)
            
            annotated_count += 1
            
        except Exception as e:
            print(f"\nErreur pour {image_file.name}: {e}")
    
    print(f"\n✅ Pré-annotation terminée:")
    print(f"   - Images annotées: {annotated_count}")
    print(f"   - Images ignorées (déjà annotées): {skipped_count}")
    print(f"   - Fichiers JSON sauvegardés dans: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pré-annote des images avec un modèle YOLO OBB"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Chemin vers le modèle YOLO OBB (.pt)"
    )
    parser.add_argument(
        "--images", "-i",
        type=str,
        required=True,
        help="Répertoire contenant les images"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Répertoire de sortie (par défaut: même que images)"
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.25,
        help="Seuil de confiance minimum (défaut: 0.25)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Écraser les annotations existantes"
    )
    
    args = parser.parse_args()
    
    preannotate_images(
        model_path=args.model,
        images_dir=args.images,
        output_dir=args.output,
        conf_threshold=args.conf,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
