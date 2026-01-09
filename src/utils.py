#!/usr/bin/env python3
"""
Fonctions utilitaires pour l'analyse d'épis de blé
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd

def setup_logging(debug=False):
    """
    Configure le système de logging
    
    Args:
        debug: Si True, active le mode debug avec plus de détails
    
    Returns:
        Logger configuré
    """
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Configuration du format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Configuration du logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('wheat_analyzer.log', mode='a')
        ]
    )
    
    logger = logging.getLogger('WheatAnalyzer')
    logger.setLevel(log_level)
    
    return logger


def load_config(config_path='config/config.yaml'):
    """
    Charge la configuration depuis un fichier YAML
    
    Args:
        config_path: Chemin vers le fichier de configuration
    
    Returns:
        Dictionnaire de configuration
    """
    if not os.path.exists(config_path):
        logging.warning(f"Fichier de configuration non trouvé: {config_path}")
        logging.info("Utilisation de la configuration par défaut")
        return get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration chargée depuis: {config_path}")
        return config
    except Exception as e:
        logging.error(f"Erreur lors du chargement de la configuration: {e}")
        logging.info("Utilisation de la configuration par défaut")
        return get_default_config()


def get_default_config():
    """
    Retourne la configuration par défaut
    
    Returns:
        Dictionnaire de configuration par défaut
    """
    return {
        'calibration': {
            'ruler_min_length_px': 200,
            'ruler_angle_tolerance': 5,
            'ruler_color_lower_hsv': [0, 0, 200],
            'ruler_color_upper_hsv': [180, 30, 255],
            'expected_ruler_length_mm': 100
        },
        'segmentation': {
            'hsv_wheat_lower': [10, 20, 100],
            'hsv_wheat_upper': [40, 255, 255],
            'min_spike_area_px': 5000,
            'morphology_kernel_size': 5
        },
        'separation': {
            'watershed_min_distance': 50,
            'watershed_threshold': 0.3
        },
        'spikelets': {
            'min_distance_px': 10,
            'prominence': 5,
            'profile_smoothing_sigma': 3
        },
        'morphology': {
            'skeleton_pruning': 5,
            'angle_window_size': 10
        },
        'visualization': {
            'save_intermediate': True,
            'dpi': 300,
            'figure_size': [12, 8]
        }
    }


def create_output_structure(session_dir):
    """
    Crée la structure de dossiers pour une session d'analyse
    
    Args:
        session_dir: Dossier de la session
    
    Returns:
        Dictionnaire avec les chemins des sous-dossiers
    """
    session_dir = Path(session_dir)
    
    subdirs = {
        'visualizations': session_dir / 'visualizations',
        'data': session_dir / 'data',
        'intermediate': session_dir / 'intermediate',
        'masks': session_dir / 'masks'
    }
    
    # Créer tous les dossiers
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    
    return subdirs


def save_results(results, output_dir, image_name):
    """
    Sauvegarde les résultats dans plusieurs formats
    
    Args:
        results: Liste de dictionnaires avec les résultats
        output_dir: Dossier de sortie
        image_name: Nom de l'image analysée
    """
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Créer le dossier de sortie
    session_dir = output_dir / f"{image_name}_{timestamp}"
    subdirs = create_output_structure(session_dir)
    
    # Sauvegarder en JSON
    json_path = subdirs['data'] / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    logging.info(f"Résultats JSON sauvegardés: {json_path}")
    
    # Sauvegarder en CSV
    if results:
        df = pd.DataFrame(results)
        csv_path = subdirs['data'] / 'results.csv'
        df.to_csv(csv_path, index=False)
        logging.info(f"Résultats CSV sauvegardés: {csv_path}")
        
        # Sauvegarder en Excel
        excel_path = subdirs['data'] / 'results.xlsx'
        df.to_excel(excel_path, index=False, engine='openpyxl')
        logging.info(f"Résultats Excel sauvegardés: {excel_path}")
    
    return session_dir


def validate_image_path(image_path):
    """
    Vérifie qu'un chemin d'image est valide
    
    Args:
        image_path: Chemin vers l'image
    
    Returns:
        True si valide, False sinon
    """
    if not os.path.exists(image_path):
        logging.error(f"Fichier non trouvé: {image_path}")
        return False
    
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    ext = os.path.splitext(image_path)[1].lower()
    
    if ext not in valid_extensions:
        logging.error(f"Extension non supportée: {ext}")
        logging.info(f"Extensions valides: {', '.join(valid_extensions)}")
        return False
    
    return True


def format_results_summary(results):
    """
    Formate un résumé des résultats pour affichage
    
    Args:
        results: Liste de dictionnaires avec les résultats
    
    Returns:
        Chaîne formatée
    """
    if not results:
        return "Aucun épi détecté"
    
    summary = []
    summary.append("\n" + "="*60)
    summary.append("RÉSUMÉ DES RÉSULTATS")
    summary.append("="*60)
    summary.append(f"Nombre d'épis détectés: {len(results)}\n")
    
    for i, result in enumerate(results, 1):
        summary.append(f"Épi {i}:")
        summary.append(f"  • Longueur: {result.get('longueur_epi_mm', 'N/A'):.1f} mm")
        summary.append(f"  • Largeur: {result.get('largeur_epi_mm', 'N/A'):.1f} mm")
        summary.append(f"  • Surface: {result.get('surface_epi_mm2', 'N/A'):.1f} mm²")
        summary.append(f"  • Épillets: {result.get('nombre_epillets', 'N/A')}")
        summary.append(f"  • Densité: {result.get('densite_epillets_par_cm', 'N/A'):.2f} /cm")
        summary.append("")
    
    summary.append("="*60)
    
    return "\n".join(summary)
