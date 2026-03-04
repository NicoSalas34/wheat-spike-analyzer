#!/usr/bin/env python3
"""
Application de vérification des résultats d'analyse

Interface web Flask pour visualiser et corriger les résultats
de l'analyse des épis de blé.

Raccourcis clavier:
    ← / → : Image précédente / suivante
    V     : Valider l'image courante (ajoute tag "Validé")
    R     : Rejeter l'image courante
    S     : Sauvegarder les corrections
    F     : Filtrer (non validés seulement)
    T     : Focus sur les tags
    1-9   : Toggle tag rapide

Tags prédéfinis:
    - Validé : Tout est correct
    - Règle non détectée
    - Épi mal détecté
    - Sachet illisible
    - Épillets incorrects
    - Image floue
    - Plusieurs épis confondus

Usage:
    python app/verification_app.py --output output/
    python app/verification_app.py --output output/ --port 5001
"""

import argparse
import csv
import json
import logging
import os
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, Response, jsonify, render_template_string, request, send_file

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Variables globales
RESULTS_DIR = None
RESULTS_CACHE = {}
CORRECTIONS_FILE = None

# Tags prédéfinis pour les problèmes
PREDEFINED_TAGS = [
    {"id": "validated", "label": "✓ Validé", "color": "#4ade80", "shortcut": "1"},
    {"id": "ruler_missing", "label": "Règle non détectée", "color": "#f87171", "shortcut": "2"},
    {"id": "spike_wrong", "label": "Épi mal détecté", "color": "#fb923c", "shortcut": "3"},
    {"id": "bag_unreadable", "label": "Sachet illisible", "color": "#fbbf24", "shortcut": "4"},
    {"id": "spikelets_wrong", "label": "Épillets incorrects", "color": "#a78bfa", "shortcut": "5"},
    {"id": "blurry", "label": "Image floue", "color": "#60a5fa", "shortcut": "6"},
    {"id": "multiple_spikes", "label": "Plusieurs épis confondus", "color": "#f472b6", "shortcut": "7"},
    {"id": "calibration_wrong", "label": "Calibration incorrecte", "color": "#94a3b8", "shortcut": "8"},
]


def load_all_results() -> List[Dict]:
    """Charge tous les résultats depuis le dossier output"""
    global RESULTS_CACHE
    
    results = []
    results_dir = Path(RESULTS_DIR)
    
    for results_file in sorted(results_dir.glob('**/results.json')):
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            # Ajouter le chemin du dossier de session
            session_dir = results_file.parent
            data['_session_dir'] = str(session_dir)
            data['_results_file'] = str(results_file)
            
            # Charger la vérification existante si présente
            if '_verification' in data:
                data['_corrections'] = {
                    'status': data['_verification'].get('status', 'pending'),
                    'tags': data['_verification'].get('tags', []),
                    'notes': data['_verification'].get('notes', ''),
                }
            else:
                data['_corrections'] = {'status': 'pending', 'tags': []}
            
            results.append(data)
        except Exception as e:
            logger.warning(f"Erreur chargement {results_file}: {e}")
    
    # Mettre en cache
    RESULTS_CACHE = {Path(r['image']).stem: r for r in results}
    
    return results


def save_correction(image_id: str, corrections: Dict) -> bool:
    """Sauvegarde les corrections directement dans results.json"""
    if image_id not in RESULTS_CACHE:
        return False
    
    result = RESULTS_CACHE[image_id]
    results_file = Path(result['_results_file'])
    
    # Mettre à jour le résultat avec les corrections
    result['_verification'] = {
        'status': corrections.get('status', 'pending'),
        'tags': corrections.get('tags', []),
        'notes': corrections.get('notes', ''),
        'verified_at': datetime.now().isoformat(),
        'verified_by': 'verification_app'
    }
    
    # Mettre à jour les valeurs corrigées
    if corrections.get('sample_id'):
        if 'bag' not in result:
            result['bag'] = {}
        result['bag']['sample_id_corrected'] = corrections['sample_id']
    
    # Corrections des épis
    for key, value in corrections.items():
        if key.startswith('spike_') and '_' in key[6:]:
            parts = key.split('_')
            if len(parts) >= 3:
                spike_idx = int(parts[1])
                field = '_'.join(parts[2:])
                
                if 'spikes' in result and spike_idx < len(result['spikes']):
                    if 'corrections' not in result['spikes'][spike_idx]:
                        result['spikes'][spike_idx]['corrections'] = {}
                    result['spikes'][spike_idx]['corrections'][field] = value
    
    # Sauvegarder le fichier results.json modifié
    try:
        # Créer une copie sans les clés internes pour la sauvegarde
        save_data = {k: v for k, v in result.items() if not k.startswith('_') or k == '_verification'}
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        # Mettre à jour le cache
        RESULTS_CACHE[image_id].update(result)
        
        logger.info(f"Sauvegardé: {results_file}")
        return True
        
    except Exception as e:
        logger.error(f"Erreur sauvegarde {results_file}: {e}")
        return False


def get_debug_image_path(session_dir: str, image_type: str = 'final') -> Optional[str]:
    """Trouve l'image de debug correspondante"""
    session_path = Path(session_dir)
    # Chercher différents noms possibles — d'abord dans debug/, puis à la racine
    patterns = {
        'final': ['result_annotated_corrected*.png', '05_final*.jpg', '05_final*.png', '*final*.jpg', '*final*.png', 'result_annotated*.png'],
        'detections': ['01_detections*.jpg', '01_detections*.png', '01_detections*.png'],
        'spikes': ['02_spikes*.jpg', '02_spikes*.png', '03_spikelets*.jpg', '03_spikelets*.png'],
        'bag': ['04_bag*.jpg', '04_bag*.png'],
    }

    # Helper to search a path for patterns
    def search_path(p: Path):
        for pattern in patterns.get(image_type, patterns['final']):
            matches = list(p.glob(pattern))
            if matches:
                return str(matches[0])
        # Try any image fallback
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            matches = list(p.glob(ext))
            if matches:
                return str(matches[0])
        return None

    # 1) check debug subfolder
    debug_path = session_path / 'debug'
    if debug_path.exists() and debug_path.is_dir():
        found = search_path(debug_path)
        if found:
            return found

    # 2) check session root
    found = search_path(session_path)
    if found:
        return found

    # 3) recursive search (any matching file under session)
    for pattern in patterns.get(image_type, patterns['final']):
        matches = list(session_path.rglob(pattern))
        if matches:
            return str(matches[0])

    return None


def add_spike_numbers_overlay(session_dir: str) -> bool:
    """Ajoute les numéros d'épis en surimpression sur l'image annotée originale.
    
    Lit result_annotated.png et ajoute uniquement les numéros d'épis visibles
    basés sur les données de results.json. Sauvegarde dans result_annotated_corrected.png.
    """
    try:
        import cv2
        session_path = Path(session_dir)
        results_file = session_path / 'results.json'
        
        if not results_file.exists():
            logger.warning(f"results.json introuvable pour la session: {session_dir}")
            return False

        with open(results_file, 'r') as f:
            results = json.load(f)

        # Chercher l'image annotée originale (pas la corrigée)
        original_annotated = session_path / 'result_annotated.png'
        if not original_annotated.exists():
            # Essayer dans le dossier debug
            original_annotated = session_path / 'debug' / 'result_annotated.png'
        
        if not original_annotated.exists():
            logger.warning(f"result_annotated.png introuvable: {session_dir}")
            return False

        img = cv2.imread(str(original_annotated))
        if img is None:
            logger.warning(f"Impossible de lire l'image: {original_annotated}")
            return False

        viz = img.copy()
        
        # Dessiner les numéros d'épis en surimpression
        spikes = results.get('spikes', []) or []
        for i, spike in enumerate(spikes):
            spikelet = spike.get('spikelets', {}) or {}
            
            # Calculer le centre de l'épi
            positions = spikelet.get('positions', [])
            bboxes = spikelet.get('bboxes', [])
            
            if positions:
                cx = int(sum(p[0] for p in positions) / len(positions))
                cy = int(sum(p[1] for p in positions) / len(positions))
            elif bboxes:
                centers = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in bboxes]
                cx = int(sum(c[0] for c in centers) / len(centers))
                cy = int(sum(c[1] for c in centers) / len(centers))
            else:
                # Position par défaut basée sur l'index
                cx, cy = 150 + i * 300, 150
            
            # Numéro de l'épi (index+1)
            spike_num = i + 1
            label = f"#{spike_num}"
            
            # Taille du texte
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            thickness = 4
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Position du texte (centré)
            text_x = cx - text_w // 2
            text_y = cy + text_h // 2
            
            # Fond semi-transparent (rectangle vert)
            padding = 10
            cv2.rectangle(viz, 
                         (text_x - padding, text_y - text_h - padding), 
                         (text_x + text_w + padding, text_y + padding), 
                         (0, 120, 0), -1)
            
            # Bordure blanche
            cv2.rectangle(viz, 
                         (text_x - padding, text_y - text_h - padding), 
                         (text_x + text_w + padding, text_y + padding), 
                         (255, 255, 255), 2)
            
            # Texte blanc avec contour
            cv2.putText(viz, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(viz, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        # Sauvegarder l'image avec surimpression
        out_path = session_path / 'result_annotated_corrected.png'
        cv2.imwrite(str(out_path), viz)
        logger.info(f"Added spike numbers overlay: {out_path}")
        return True

    except Exception as e:
        logger.error(f"Erreur ajout numéros épis: {e}")
        return False


# Template HTML principal
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vérification - Wheat Spike Analyzer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        /* Header */
        .header {
            background: #16213e;
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #0f3460;
        }
        
        .header h1 {
            font-size: 1.2rem;
            color: #e94560;
        }
        
        .nav-info {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        
        .counter {
            background: #0f3460;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
        }
        
        .header-btn {
            background: #0f3460;
            border: none;
            color: #eee;
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.85rem;
        }
        
        .header-btn:hover {
            background: #1a4a80;
        }
        
        .header-btn.active {
            background: #e94560;
        }
        
        .header-btn.export {
            background: #10b981;
        }
        
        .header-btn.export:hover {
            background: #059669;
        }
        
        /* Filter dropdown */
        .filter-dropdown {
            position: relative;
            display: inline-block;
        }
        
        .filter-select {
            background: #0f3460;
            border: none;
            color: #eee;
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.85rem;
            appearance: none;
            padding-right: 30px;
            min-width: 180px;
        }
        
        .filter-select:hover {
            background: #1a4a80;
        }
        
        .filter-select:focus {
            outline: none;
            background: #1a4a80;
        }
        
        .filter-dropdown::after {
            content: '▼';
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            pointer-events: none;
            font-size: 0.7rem;
        }

        /* Main content */
        .main {
            flex: 1;
            display: flex;
            overflow: hidden;
        }
        
        /* Image panel */
        .image-panel {
            flex: 2;
            display: flex;
            flex-direction: column;
            background: #0f0f1a;
            position: relative;
        }
        
        .image-container {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            padding: 10px;
            position: relative;
            cursor: grab;
        }
        
        .image-container.dragging {
            cursor: grabbing;
        }
        
        .image-container img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            transform-origin: center center;
            transition: transform 0.1s ease-out;
            pointer-events: none;
            user-select: none;
        }
        
        .zoom-controls {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 10px;
            background: rgba(0,0,0,0.7);
            padding: 8px 15px;
            border-radius: 20px;
            z-index: 100;
        }
        
        .zoom-btn {
            background: #0f3460;
            border: none;
            color: white;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 1.2rem;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
        }
        
        .zoom-btn:hover {
            background: #e94560;
        }
        
        .zoom-level {
            color: white;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            min-width: 50px;
            justify-content: center;
        }
        
        .image-nav {
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            background: rgba(233, 69, 96, 0.8);
            border: none;
            color: white;
            font-size: 2rem;
            padding: 20px 15px;
            cursor: pointer;
            transition: background 0.2s;
            z-index: 150;
        }
        
        .image-nav:hover {
            background: #e94560;
        }
        
        .image-nav.prev { left: 0; border-radius: 0 5px 5px 0; }
        .image-nav.next { right: 0; border-radius: 5px 0 0 5px; }
        
        /* Info panel */
        .info-panel {
            flex: 1;
            background: #16213e;
            padding: 15px;
            overflow-y: auto;
            min-width: 380px;
            max-width: 420px;
        }
        
        .section {
            background: #0f3460;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 12px;
        }
        
        .section h3 {
            color: #e94560;
            margin-bottom: 10px;
            font-size: 0.85rem;
            text-transform: uppercase;
        }
        
        .info-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 4px 0;
            border-bottom: 1px solid #1a1a3e;
        }
        
        .info-row:last-child {
            border-bottom: none;
        }
        
        .info-label {
            color: #888;
            font-size: 0.9rem;
        }
        
        .info-value {
            font-weight: bold;
        }
        
        .info-value.success { color: #4ade80; }
        .info-value.warning { color: #fbbf24; }
        .info-value.error { color: #f87171; }
        
        /* Editable fields */
        .editable {
            background: #1a1a3e;
            border: 1px solid #333;
            color: #eee;
            padding: 3px 8px;
            border-radius: 3px;
            width: 80px;
            text-align: right;
        }
        
        .editable:focus {
            border-color: #e94560;
            outline: none;
        }
        
        /* Tags */
        .tags-container {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 8px;
        }
        
        .tag {
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 0.75rem;
            cursor: pointer;
            border: 2px solid transparent;
            transition: all 0.2s;
            opacity: 0.5;
        }
        
        .tag:hover {
            opacity: 0.8;
        }
        
        .tag.active {
            opacity: 1;
            border-color: white;
            box-shadow: 0 0 8px rgba(255,255,255,0.3);
        }
        
        .tag-shortcut {
            background: rgba(0,0,0,0.3);
            padding: 1px 5px;
            border-radius: 3px;
            margin-right: 5px;
            font-size: 0.7rem;
        }
        
        .custom-tag-input {
            background: #1a1a3e;
            border: 1px dashed #555;
            color: #eee;
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 0.75rem;
            width: 120px;
        }
        
        .custom-tag-input::placeholder {
            color: #666;
        }
        
        /* Active tags display */
        .active-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            margin-top: 8px;
        }
        
        .active-tag {
            display: inline-flex;
            align-items: center;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.7rem;
            color: #000;
        }
        
        /* Spike list */
        .spike-item {
            background: #1a1a3e;
            border-radius: 5px;
            padding: 8px;
            margin-bottom: 6px;
        }
        
        .spike-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        
        .spike-id {
            color: #e94560;
            font-weight: bold;
            font-size: 0.9rem;
        }
        
        /* Actions */
        .actions {
            display: flex;
            gap: 8px;
            margin-top: 12px;
        }
        
        .btn {
            flex: 1;
            padding: 10px;
            border: none;
            border-radius: 5px;
            font-size: 0.9rem;
            cursor: pointer;
            transition: transform 0.1s, opacity 0.2s;
        }
        
        .btn:hover {
            transform: scale(1.02);
        }
        
        .btn:active {
            transform: scale(0.98);
        }
        
        .btn-validate {
            background: #4ade80;
            color: #000;
        }
        
        .btn-reject {
            background: #f87171;
            color: #000;
        }
        
        .btn-save {
            background: #60a5fa;
            color: #000;
        }
        
        .btn-danger {
            background: #f87171;
            color: #000;
            padding: 4px 8px;
            font-size: 0.75rem;
        }
        
        .spike-actions {
            margin-top: 6px;
            text-align: right;
        }
        
        /* Status badge */
        .status-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 10px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        
        .status-validated { background: #4ade80; color: #000; }
        .status-rejected { background: #f87171; color: #000; }
        .status-pending { background: #fbbf24; color: #000; }
        
        /* Shortcuts help */
        .shortcuts {
            position: fixed;
            bottom: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            padding: 10px 15px;
            border-radius: 5px;
            font-size: 0.7rem;
            color: #888;
        }
        
        .shortcuts kbd {
            background: #333;
            padding: 2px 5px;
            border-radius: 3px;
            margin-right: 3px;
        }
        
        /* Loading */
        .loading {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }
        
        .loading.show {
            display: flex;
        }
        
        /* Toast notifications */
        .toast {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 25px;
            border-radius: 5px;
            font-weight: bold;
            transform: translateX(150%);
            transition: transform 0.3s;
            z-index: 1001;
        }
        
        .toast.show {
            transform: translateX(0);
        }
        
        .toast.success { background: #4ade80; color: #000; }
        .toast.error { background: #f87171; color: #000; }
        .toast.info { background: #60a5fa; color: #000; }
        
        /* File navigator panel */
        .nav-panel {
            width: 250px;
            min-width: 200px;
            max-width: 350px;
            background: #16213e;
            border-right: 1px solid #0f3460;
            display: flex;
            flex-direction: column;
            transition: width 0.2s;
        }
        
        .nav-panel.collapsed {
            width: 40px;
            min-width: 40px;
        }
        
        .nav-panel.collapsed .nav-content {
            display: none;
        }
        
        .nav-toggle {
            background: #0f3460;
            border: none;
            color: #eee;
            padding: 8px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            border-bottom: 1px solid #1a4a80;
        }
        
        .nav-toggle:hover {
            background: #1a4a80;
        }
        
        .nav-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .nav-search {
            padding: 10px;
            border-bottom: 1px solid #0f3460;
        }
        
        .nav-search input {
            width: 100%;
            background: #0f3460;
            border: 1px solid #1a4a80;
            color: #eee;
            padding: 8px 10px;
            border-radius: 5px;
            font-size: 0.85rem;
        }
        
        .nav-search input:focus {
            outline: none;
            border-color: #e94560;
        }
        
        .nav-search input::placeholder {
            color: #666;
        }
        
        .nav-stats {
            padding: 5px 10px;
            font-size: 0.75rem;
            color: #888;
            border-bottom: 1px solid #0f3460;
        }
        
        .file-list {
            flex: 1;
            overflow-y: auto;
            padding: 5px 0;
        }
        
        .file-item {
            padding: 6px 10px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.8rem;
            border-left: 3px solid transparent;
            transition: all 0.15s;
        }
        
        .file-item:hover {
            background: #0f3460;
        }
        
        .file-item.active {
            background: #1a4a80;
            border-left-color: #e94560;
        }
        
        .file-item .file-status {
            font-size: 0.7rem;
        }
        
        .file-item .file-name {
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .file-item.validated .file-status { color: #4ade80; }
        .file-item.rejected .file-status { color: #f87171; }
        .file-item.pending .file-status { color: #fbbf24; }
        
        /* Go to input */
        .nav-goto {
            padding: 8px 10px;
            border-top: 1px solid #0f3460;
            display: flex;
            gap: 5px;
        }
        
        .nav-goto input {
            flex: 1;
            background: #0f3460;
            border: 1px solid #1a4a80;
            color: #eee;
            padding: 5px 8px;
            border-radius: 3px;
            font-size: 0.8rem;
            width: 60px;
        }
        
        .nav-goto button {
            background: #e94560;
            border: none;
            color: white;
            padding: 5px 10px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 0.8rem;
        }
        
        .nav-goto button:hover {
            background: #d03050;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌾 Wheat Spike Analyzer - Vérification</h1>
        <div class="nav-info">
            <button class="header-btn" onclick="toggleNavPanel()" title="Afficher/masquer la liste (L)">
                📂 Liste
            </button>
            <div class="filter-dropdown">
                <select class="filter-select" id="filterSelect" onchange="applyAdvancedFilter()">
                    <option value="all">📋 Tous</option>
                    <option value="awaiting">⏳ En attente</option>
                    <option value="pending">⏳ Non validés</option>
                    <option value="validated">✓ Validés</option>
                    <option value="rejected">✗ Rejetés</option>
                    <optgroup label="── Problèmes détection ──">
                        <option value="no_ruler">📏 Pas de règle</option>
                        <option value="no_spikes">🌾 Pas d'épis</option>
                        <option value="no_bag">🏷️ Pas de sachet</option>
                        <option value="no_spikelets">🔢 Pas d'épillets</option>
                    </optgroup>
                    <optgroup label="── Qualité ──">
                        <option value="low_confidence">⚠️ Confiance faible (&lt;80%)</option>
                        <option value="multiple_spikes">🌾🌾 Plusieurs épis</option>
                        <option value="single_spike">🌾 Un seul épi</option>
                    </optgroup>
                    <optgroup label="── Tags ──">
                        <option value="tag_ruler_missing">🏷️ Tag: Règle non détectée</option>
                        <option value="tag_spike_bad">🏷️ Tag: Épi mal détecté</option>
                        <option value="tag_bag_unreadable">🏷️ Tag: Sachet illisible</option>
                        <option value="tag_spikelets_wrong">🏷️ Tag: Épillets incorrects</option>
                    </optgroup>
                </select>
            </div>
            <button class="header-btn export" onclick="regenerateCSV()">
                📊 Régénérer CSV
            </button>
            <button class="header-btn" onclick="regenerateCurrentImage()">
                🔁 Régénérer image
            </button>
            <button class="header-btn" onclick="undoLastDelete()">
                ↶ Annuler suppression
            </button>
            <div class="counter">
                <span id="currentIndex">0</span> / <span id="totalCount">0</span>
                (<span id="validatedCount">0</span> ✓)
            </div>
        </div>
    </div>
    
    <div class="main">
        <!-- Navigation panel -->
        <div class="nav-panel" id="navPanel">
            <button class="nav-toggle" onclick="toggleNavPanel()" title="Masquer la liste">
                ◀ Liste des fichiers
            </button>
            <div class="nav-content">
                <div class="nav-search">
                    <input type="text" id="fileSearch" placeholder="🔍 Rechercher..." oninput="filterFileList()">
                </div>
                <div class="nav-stats" id="navStats">0 fichiers</div>
                <div class="file-list" id="fileList"></div>
                <div class="nav-goto">
                    <input type="number" id="gotoIndex" min="1" placeholder="#" onkeypress="handleGotoKeypress(event)">
                    <button onclick="gotoIndex()">Aller</button>
                </div>
            </div>
        </div>
        
        <div class="image-panel">
            <button class="image-nav prev" onclick="navigate(-1)">‹</button>
            <div class="image-container" id="imageContainer">
                <img id="mainImage" src="" alt="Image en cours">
                <div class="zoom-controls">
                    <button class="zoom-btn" onclick="zoomOut()" title="Zoom -">-</button>
                    <span class="zoom-level" id="zoomLevel">100%</span>
                    <button class="zoom-btn" onclick="zoomIn()" title="Zoom +">+</button>
                    <button class="zoom-btn" onclick="resetZoom()" title="Reset">↻</button>
                </div>
            </div>
            <button class="image-nav next" onclick="navigate(1)">›</button>
        </div>
        
        <div class="info-panel">
            <div class="section">
                <h3>📷 Image</h3>
                <div class="info-row">
                    <span class="info-label">Fichier</span>
                    <span class="info-value" id="imageName">-</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Statut</span>
                    <span id="statusBadge" class="status-badge status-pending">En attente</span>
                </div>
                <div id="activeTagsDisplay" class="active-tags"></div>
            </div>
            
            <div class="section">
                <h3>🏷️ Tags (1-8 pour toggle)</h3>
                <div class="tags-container" id="tagsContainer"></div>
                <div style="margin-top:8px;">
                    <input type="text" class="custom-tag-input" id="customTag" 
                           placeholder="+ Tag personnalisé" onkeypress="addCustomTag(event)">
                </div>
            </div>
            
            <div class="section">
                <h3>📏 Calibration</h3>
                <div class="info-row">
                    <span class="info-label">Règle détectée</span>
                    <span class="info-value" id="rulerDetected">-</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Échelle</span>
                    <span class="info-value" id="pixelPerMm">-</span>
                </div>
            </div>
            
            <div class="section">
                <h3>🏷️ Identification</h3>
                <div class="info-row">
                    <span class="info-label">ID Sachet</span>
                    <span class="info-value" id="sampleIdOriginal" style="font-size:0.8rem;color:#888;"></span>
                </div>
                <div class="info-row" style="gap:5px;">
                    <span class="info-label">Bac</span>
                    <input type="number" min="1" class="editable" id="sampleBac" placeholder="?" style="width:50px;">
                    <span class="info-label">Ligne</span>
                    <input type="number" min="1" class="editable" id="sampleLigne" placeholder="?" style="width:50px;">
                    <span class="info-label">Col</span>
                    <input type="number" min="1" class="editable" id="sampleCol" placeholder="?" style="width:50px;">
                </div>
                <div class="info-row">
                    <span class="info-label">Confiance</span>
                    <span class="info-value" id="bagConfidence">-</span>
                </div>
            </div>
            
            <div class="section">
                <h3>🌾 Épis (<span id="spikeCount">0</span>)</h3>
                <div id="spikeList"></div>
            </div>
            
            <div class="section">
                <h3>📝 Notes</h3>
                <textarea id="notes" style="width:100%; height:50px; background:#1a1a3e; border:1px solid #333; color:#eee; border-radius:5px; padding:8px; resize:vertical; font-size:0.85rem;"></textarea>
            </div>
            
            <div class="actions">
                <button class="btn btn-validate" onclick="validateWithTag()">
                    ✓ Valider (V)
                </button>
                <button class="btn btn-reject" onclick="setStatus('rejected')">
                    ✗ Rejeter (R)
                </button>
            </div>
            <div class="actions">
                <button class="btn btn-save" onclick="saveCorrections()">
                    💾 Sauvegarder (S)
                </button>
            </div>
        </div>
    </div>
    
    <div class="shortcuts">
        <kbd>←</kbd><kbd>→</kbd> Nav
        <kbd>V</kbd> Valider
        <kbd>R</kbd> Rejeter
        <kbd>S</kbd> Sauver
        <kbd>F</kbd> Cycle filtres
        <kbd>L</kbd> Liste
        <kbd>G</kbd> Aller à
        <kbd>1-8</kbd> Tags
    </div>
    
    <div class="loading" id="loading">Chargement...</div>
    <div class="toast" id="toast"></div>

    <script>
        // Tags prédéfinis (depuis le serveur)
        const PREDEFINED_TAGS = {{ predefined_tags | safe }};
        
        let results = [];
        let filteredResults = [];
        let currentIndex = 0;
        let currentTags = [];
        
        // Charger les résultats au démarrage
        async function loadResults() {
            showLoading(true);
            // Sauvegarder l'image courante pour la retrouver après rechargement
            const currentImagePath = filteredResults[currentIndex]?.image || null;
            try {
                const response = await fetch('/api/results');
                results = await response.json();
                applyFilter();
                renderTags();
                
                // Retrouver l'image courante dans les résultats filtrés
                if (currentImagePath) {
                    const newIndex = filteredResults.findIndex(r => r.image === currentImagePath);
                    if (newIndex >= 0) {
                        currentIndex = newIndex;
                    }
                }
                
                if (filteredResults.length > 0) {
                    displayResult(currentIndex);
                }
                updateCounter();
            } catch (e) {
                showToast('Erreur de chargement', 'error');
            }
            showLoading(false);
        }
        
        function renderTags() {
            const container = document.getElementById('tagsContainer');
            container.innerHTML = PREDEFINED_TAGS.map(tag => `
                <div class="tag" id="tag_${tag.id}" 
                     style="background:${tag.color}; color:#000;"
                     onclick="toggleTag('${tag.id}')">
                    <span class="tag-shortcut">${tag.shortcut}</span>
                    ${tag.label}
                </div>
            `).join('');
        }
        
        function toggleTag(tagId) {
            const idx = currentTags.indexOf(tagId);
            if (idx >= 0) {
                currentTags.splice(idx, 1);
            } else {
                // Si on ajoute "validated", on retire les autres problèmes
                if (tagId === 'validated') {
                    currentTags = ['validated'];
                } else {
                    // Si on ajoute un problème, on retire "validated"
                    currentTags = currentTags.filter(t => t !== 'validated');
                    currentTags.push(tagId);
                }
            }
            updateTagDisplay();
        }
        
        function updateTagDisplay() {
            // Mettre à jour les badges de tags
            PREDEFINED_TAGS.forEach(tag => {
                const el = document.getElementById(`tag_${tag.id}`);
                if (el) {
                    el.classList.toggle('active', currentTags.includes(tag.id));
                }
            });
            
            // Afficher les tags actifs
            const display = document.getElementById('activeTagsDisplay');
            display.innerHTML = currentTags.map(tagId => {
                const tag = PREDEFINED_TAGS.find(t => t.id === tagId) || {label: tagId, color: '#666'};
                return `<span class="active-tag" style="background:${tag.color}">${tag.label}</span>`;
            }).join('');
        }
        
        function addCustomTag(event) {
            if (event.key === 'Enter') {
                const input = document.getElementById('customTag');
                const tag = input.value.trim();
                if (tag && !currentTags.includes(tag)) {
                    currentTags = currentTags.filter(t => t !== 'validated');
                    currentTags.push(tag);
                    updateTagDisplay();
                }
                input.value = '';
            }
        }
        
        function applyFilter() {
            // Appelé au chargement initial - applique le filtre courant
            applyAdvancedFilter();
        }
        
        function applyAdvancedFilter() {
            const filterValue = document.getElementById('filterSelect').value;
            
            filteredResults = results.filter(r => {
                const corrections = r._corrections || {};
                const status = corrections.status || 'pending';
                const tags = corrections.tags || [];
                const cal = r.calibration || {};
                const bag = r.bag || {};
                const spikes = r.spikes || [];
                
                switch(filterValue) {
                    case 'all':
                        return true;
                    case 'awaiting':
                        return status === 'pending';
                    case 'pending':
                        return status !== 'validated';
                    case 'validated':
                        return status === 'validated';
                    case 'rejected':
                        return status === 'rejected';
                    
                    // Problèmes de détection
                    case 'no_ruler':
                        return !cal.ruler_detected;
                    case 'no_spikes':
                        return spikes.length === 0;
                    case 'no_bag':
                        return !bag.detected;
                    case 'no_spikelets':
                        return spikes.some(s => !s.spikelet_count || s.spikelet_count === 0);
                    
                    // Qualité
                    case 'low_confidence':
                        return (bag.confidence && bag.confidence < 0.8) || 
                               spikes.some(s => s.confidence && s.confidence < 0.8);
                    case 'multiple_spikes':
                        return spikes.length > 1;
                    case 'single_spike':
                        return spikes.length === 1;
                    
                    // Tags
                    case 'tag_ruler_missing':
                        return tags.includes('ruler_missing');
                    case 'tag_spike_bad':
                        return tags.includes('spike_bad');
                    case 'tag_bag_unreadable':
                        return tags.includes('bag_unreadable');
                    case 'tag_spikelets_wrong':
                        return tags.includes('spikelets_wrong');
                    
                    default:
                        return true;
                }
            });
            
            currentIndex = Math.min(currentIndex, Math.max(0, filteredResults.length - 1));
            if (filteredResults.length > 0) {
                displayResult(currentIndex);
            } else {
                // Aucun résultat pour ce filtre
                document.getElementById('mainImage').src = '';
                document.getElementById('imageName').textContent = 'Aucune image';
            }
            updateCounter();
            // Mettre à jour la liste de fichiers
            if (typeof renderFileList === 'function') {
                renderFileList();
            }
        }
        
        function toggleFilter() {
            // Legacy - bascule entre tous et non validés
            const select = document.getElementById('filterSelect');
            select.value = select.value === 'all' ? 'pending' : 'all';
            applyAdvancedFilter();
        }
        
        function cycleFilter() {
            // Cycle à travers les filtres principaux avec la touche F
            const select = document.getElementById('filterSelect');
            const mainFilters = ['all', 'awaiting', 'pending', 'no_ruler', 'no_spikes', 'no_bag'];
            const currentIdx = mainFilters.indexOf(select.value);
            const nextIdx = (currentIdx + 1) % mainFilters.length;
            select.value = mainFilters[nextIdx];
            applyAdvancedFilter();
            
            // Afficher quel filtre est actif
            const filterNames = {
                'all': 'Tous',
                'awaiting': 'En attente',
                'pending': 'Non validés',
                'no_ruler': 'Sans règle',
                'no_spikes': 'Sans épis',
                'no_bag': 'Sans sachet'
            };
            showToast('Filtre: ' + filterNames[mainFilters[nextIdx]], 'info');
        }
        
        // ===== ZOOM FUNCTIONALITY =====
        let currentZoom = 1;
        let panX = 0, panY = 0;
        let isDragging = false;
        let dragStartX = 0, dragStartY = 0;
        let panStartX = 0, panStartY = 0;
        const MIN_ZOOM = 0.5;
        const MAX_ZOOM = 5;
        const ZOOM_STEP = 0.25;
        
        function updateImageTransform() {
            const img = document.getElementById('mainImage');
            img.style.transform = `scale(${currentZoom}) translate(${panX}px, ${panY}px)`;
            document.getElementById('zoomLevel').textContent = Math.round(currentZoom * 100) + '%';
        }
        
        function zoomIn() {
            currentZoom = Math.min(MAX_ZOOM, currentZoom + ZOOM_STEP);
            updateImageTransform();
        }
        
        function zoomOut() {
            currentZoom = Math.max(MIN_ZOOM, currentZoom - ZOOM_STEP);
            // Ajuster le pan si on dézoom trop
            if (currentZoom <= 1) {
                panX = 0;
                panY = 0;
            }
            updateImageTransform();
        }
        
        function resetZoom() {
            currentZoom = 1;
            panX = 0;
            panY = 0;
            updateImageTransform();
        }
        
        // Initialiser le zoom sur le container
        document.addEventListener('DOMContentLoaded', () => {
            const container = document.getElementById('imageContainer');
            const img = document.getElementById('mainImage');
            
            // Zoom avec la molette
            container.addEventListener('wheel', (e) => {
                e.preventDefault();
                const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP;
                const oldZoom = currentZoom;
                currentZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, currentZoom + delta));
                
                // Zoom vers le curseur
                if (currentZoom > 1) {
                    const rect = container.getBoundingClientRect();
                    const mouseX = e.clientX - rect.left - rect.width / 2;
                    const mouseY = e.clientY - rect.top - rect.height / 2;
                    const zoomRatio = currentZoom / oldZoom;
                    panX = panX * zoomRatio + mouseX * (1 - zoomRatio) / currentZoom;
                    panY = panY * zoomRatio + mouseY * (1 - zoomRatio) / currentZoom;
                } else {
                    panX = 0;
                    panY = 0;
                }
                updateImageTransform();
            }, { passive: false });
            
            // Pan avec la souris
            container.addEventListener('mousedown', (e) => {
                if (currentZoom > 1) {
                    isDragging = true;
                    dragStartX = e.clientX;
                    dragStartY = e.clientY;
                    panStartX = panX;
                    panStartY = panY;
                    container.classList.add('dragging');
                }
            });
            
            document.addEventListener('mousemove', (e) => {
                if (isDragging) {
                    const dx = (e.clientX - dragStartX) / currentZoom;
                    const dy = (e.clientY - dragStartY) / currentZoom;
                    panX = panStartX + dx;
                    panY = panStartY + dy;
                    updateImageTransform();
                }
            });
            
            document.addEventListener('mouseup', () => {
                isDragging = false;
                document.getElementById('imageContainer').classList.remove('dragging');
            });
            
            // Double-clic pour reset
            container.addEventListener('dblclick', () => {
                resetZoom();
            });
        });
        // ===== END ZOOM =====
        
        function displayResult(index) {
            if (index < 0 || index >= filteredResults.length) return;
            
            const result = filteredResults[index];
            const corrections = result._corrections || {};
            
            // Image (request image via query param to avoid path encoding issues)
            document.getElementById('mainImage').src = '/api/image?session_dir=' + encodeURIComponent(result._session_dir);
            document.getElementById('imageName').textContent = result.image.split('/').pop();
            
            // Status
            updateStatusBadge(corrections.status || 'pending');
            
            // Tags
            currentTags = corrections.tags || [];
            updateTagDisplay();
            
            // Calibration
            const cal = result.calibration || {};
            document.getElementById('rulerDetected').textContent = cal.ruler_detected ? '✓ Oui' : '✗ Non';
            document.getElementById('rulerDetected').className = 'info-value ' + (cal.ruler_detected ? 'success' : 'error');
            document.getElementById('pixelPerMm').textContent = cal.pixel_per_mm ? 
                `${cal.pixel_per_mm.toFixed(3)} px/mm` : '-';
            
            // Bag - Parser le sample_id en bac-ligne-colonne
            const bag = result.bag || {};
            const currentSampleId = bag.sample_id_corrected || bag.sample_id || '';
            const originalSampleId = bag.sample_id || '';
            
            // Afficher l'ID original pour référence
            document.getElementById('sampleIdOriginal').textContent = originalSampleId ? `(original: ${originalSampleId})` : '';
            
            // Parser bac-ligne-colonne
            const parts = currentSampleId.split('-');
            document.getElementById('sampleBac').value = parts[0] || '';
            document.getElementById('sampleLigne').value = parts[1] || '';
            document.getElementById('sampleCol').value = parts[2] || '';
            
            document.getElementById('bagConfidence').textContent = bag.confidence ? 
                `${(bag.confidence * 100).toFixed(0)}%` : '-';
            
            // Spikes
            const spikes = result.spikes || [];
            document.getElementById('spikeCount').textContent = spikes.length;
            
            // Extraire l'imageId une seule fois (évite les problèmes de regex dans template literals)
            const imageId = result.image.split('/').pop().replace(/\.[^.]+$/, '');
            
            const spikeList = document.getElementById('spikeList');
            spikeList.innerHTML = spikes.map((spike, i) => {
                const m = spike.measurements || {};
                const corr = spike.corrections || {};
                
                // Déterminer le type d'épi
                const hasSpike = m.has_spike;
                const hasWholeSpike = m.has_whole_spike;
                const spikeletCount = spike.spikelet_count || 0;
                
                // Si c'est un whole_spike sans spike détecté et sans épillets = mal détecté
                const isBadDetection = !hasSpike && hasWholeSpike && spikeletCount === 0;
                
                // Infos sur la longueur
                const spikeLength = m.spike_length_mm || m.length_mm;
                const wholeSpikeLength = m.whole_spike_length_mm;
                const awnsLength = m.awns_length_mm;
                
                // Badge de type
                let typeBadge = '';
                if (isBadDetection) {
                    typeBadge = '<span style="color:#e94560;font-size:0.7rem;">⚠️ Détection douteuse</span>';
                } else if (hasWholeSpike && hasSpike) {
                    typeBadge = '<span style="color:#4ade80;font-size:0.7rem;">✓ Épi complet</span>';
                } else if (hasSpike) {
                    typeBadge = '<span style="color:#fbbf24;font-size:0.7rem;">Épi seul</span>';
                }
                
                // Ligne barbes si disponible
                const awnsRow = awnsLength ? `
                        <div class="info-row">
                            <span class="info-label">Barbes</span>
                            <span style="color:#a78bfa;">${awnsLength.toFixed(1)} mm</span>
                        </div>` : '';
                
                // Ligne whole spike si disponible
                const wholeRow = wholeSpikeLength ? `
                        <div class="info-row">
                            <span class="info-label">Épi total</span>
                            <span style="color:#60a5fa;">${wholeSpikeLength.toFixed(1)} mm</span>
                        </div>` : '';
                
                return `
                    <div class="spike-item" data-spike-index="${i}" style="${isBadDetection ? 'border-color:#e94560;opacity:0.7;' : ''}">
                        <div class="spike-header">
                            <span class="spike-id">Épi #${i+1}</span>
                            ${typeBadge}
                        </div>
                        <div class="info-row">
                            <span class="info-label">Longueur</span>
                            <input type="number" step="0.1" class="editable" 
                                   id="spike_${i}_length"
                                   value="${corr.length || spikeLength || ''}"
                                   placeholder="${spikeLength ? 'mm' : 'px'}">
                        </div>
                        ${wholeRow}
                        ${awnsRow}
                        <div class="info-row">
                            <span class="info-label">Épillets</span>
                            <input type="number" class="editable"
                                   id="spike_${i}_spikelets"
                                   value="${corr.spikelets || spikeletCount || ''}"
                                   placeholder="?">
                        </div>
                        <div class="spike-actions">
                            <button class="btn btn-danger" onclick="deleteSpike('${imageId}', ${i})">🗑️ Supprimer</button>
                        </div>
                    </div>
                `;
            }).join('');
            
            // Notes
            document.getElementById('notes').value = corrections.notes || '';
        }
        
        function updateStatusBadge(status) {
            const badge = document.getElementById('statusBadge');
            badge.className = 'status-badge status-' + status;
            badge.textContent = {
                'validated': '✓ Validé',
                'rejected': '✗ Rejeté',
                'pending': 'En attente'
            }[status] || 'En attente';
        }
        
        function updateCounter() {
            document.getElementById('currentIndex').textContent = 
                filteredResults.length > 0 ? currentIndex + 1 : 0;
            document.getElementById('totalCount').textContent = filteredResults.length;
            
            const validated = results.filter(r => 
                r._corrections && r._corrections.status === 'validated'
            ).length;
            document.getElementById('validatedCount').textContent = validated;
        }
        
        function validateWithTag() {
            // Ajouter le tag "validated" si aucun tag de problème
            if (currentTags.length === 0) {
                currentTags = ['validated'];
                updateTagDisplay();
            }
            setStatus('validated');
        }
        
        function setStatus(status) {
            const result = filteredResults[currentIndex];
            if (!result) return;
            
            result._corrections = result._corrections || {};
            result._corrections.status = status;
            result._corrections.tags = currentTags;
            updateStatusBadge(status);
            
            // Auto-save et passer au suivant pour validated ET rejected
            saveCorrections().then(() => {
                if ((status === 'validated' || status === 'rejected') && currentIndex < filteredResults.length - 1) {
                    setTimeout(() => navigate(1), 300);
                }
            });
        }
        
        async function saveCorrections() {
            const result = filteredResults[currentIndex];
            if (!result) return;
            
            const imageId = result.image.split('/').pop().replace(/\.[^.]+$/, '');
            
            // Combiner bac-ligne-colonne en sample_id
            const bac = document.getElementById('sampleBac').value;
            const ligne = document.getElementById('sampleLigne').value;
            const col = document.getElementById('sampleCol').value;
            const combinedSampleId = (bac || ligne || col) ? `${bac}-${ligne}-${col}` : '';
            
            // Collecter les corrections
            const corrections = {
                status: result._corrections?.status || 'pending',
                tags: currentTags,
                sample_id: combinedSampleId,
                notes: document.getElementById('notes').value,
            };
            
            // Mettre à jour le cache local pour que ça s'affiche correctement au retour
            if (combinedSampleId) {
                if (!result.bag) result.bag = {};
                result.bag.sample_id_corrected = combinedSampleId;
            }
            
            // Corrections des épis
            const spikeCount = (result.spikes || []).length;
            for (let i = 0; i < spikeCount; i++) {
                const lengthEl = document.getElementById(`spike_${i}_length`);
                const spikeletsEl = document.getElementById(`spike_${i}_spikelets`);
                
                if (lengthEl && lengthEl.value) {
                    corrections[`spike_${i}_length`] = parseFloat(lengthEl.value);
                }
                if (spikeletsEl && spikeletsEl.value) {
                    corrections[`spike_${i}_spikelets`] = parseInt(spikeletsEl.value);
                }
            }
            
            try {
                const response = await fetch('/api/save', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        image_id: imageId,
                        corrections: corrections
                    })
                });
                
                if (response.ok) {
                    result._corrections = corrections;
                    updateCounter();
                    showToast('Sauvegardé', 'success');
                } else {
                    showToast('Erreur de sauvegarde', 'error');
                }
            } catch (e) {
                showToast('Erreur réseau', 'error');
            }
        }

        async function deleteSpike(imageId, spikeIndex) {
            if (!confirm("Supprimer cet épi ? Cette action peut être annulée via l'historique.")) return;
            
            // Masquer immédiatement l'épi dans le listing (feedback visuel)
            const spikeItem = document.querySelector(`[data-spike-index="${spikeIndex}"]`);
            if (spikeItem) {
                spikeItem.style.transition = 'all 0.3s';
                spikeItem.style.opacity = '0';
                spikeItem.style.transform = 'translateX(100%)';
            }
            
            try {
                const response = await fetch('/api/delete_spike', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({image_id: imageId, spike_index: spikeIndex})
                });

                const data = await response.json();
                if (response.ok && data.success) {
                    showToast('Épi #' + (spikeIndex + 1) + ' supprimé', 'success');
                    
                    // Supprimer l'élément du DOM après l'animation
                    if (spikeItem) {
                        setTimeout(() => spikeItem.remove(), 300);
                    }
                    
                    // Mettre à jour le compteur local
                    const result = filteredResults[currentIndex];
                    if (result && result.spikes) {
                        result.spikes.splice(spikeIndex, 1);
                        document.getElementById('spikeCount').textContent = result.spikes.length;
                        
                        // Renuméroter les épis restants dans le DOM
                        setTimeout(() => {
                            const remaining = document.querySelectorAll('.spike-item');
                            remaining.forEach((item, newIdx) => {
                                item.dataset.spikeIndex = newIdx;
                                const header = item.querySelector('.spike-id');
                                if (header) header.textContent = 'Épi #' + (newIdx + 1);
                                const btn = item.querySelector('.btn-danger');
                                if (btn) btn.onclick = () => deleteSpike(imageId, newIdx);
                            });
                        }, 350);
                    }
                    
                    // Recharger les résultats en arrière-plan pour synchroniser
                    loadResults().then(() => {});
                } else {
                    // Restaurer l'affichage si erreur
                    if (spikeItem) {
                        spikeItem.style.opacity = '1';
                        spikeItem.style.transform = 'translateX(0)';
                    }
                    showToast('Erreur suppression: ' + (data.error || 'unknown'), 'error');
                }
            } catch (e) {
                // Restaurer l'affichage si erreur
                if (spikeItem) {
                    spikeItem.style.opacity = '1';
                    spikeItem.style.transform = 'translateX(0)';
                }
                showToast('Erreur réseau', 'error');
            }
        }

        async function regenerateCurrentImage() {
            const result = filteredResults[currentIndex];
            if (!result) return showToast('Aucun résultat sélectionné', 'error');
            showLoading(true);
            try {
                const response = await fetch('/api/regenerate_image', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_dir: result._session_dir})
                });
                const data = await response.json();
                if (response.ok && data.success) {
                    showToast('Image régénérée', 'success');
                    await loadResults();
                    displayResult(currentIndex);
                } else {
                    showToast('Erreur régénération: ' + (data.error || 'unknown'), 'error');
                }
            } catch (e) {
                showToast('Erreur réseau', 'error');
            }
            showLoading(false);
        }

        async function undoLastDelete() {
            const result = filteredResults[currentIndex];
            if (!result) return showToast('Aucun résultat sélectionné', 'error');
            const imageId = result.image.split('/').pop().replace(/\.[^.]+$/, '');
            if (!confirm("Annuler la dernière suppression pour cette image ?")) return;
            showLoading(true);
            try {
                const response = await fetch('/api/undo_delete_spike', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({image_id: imageId})
                });
                const data = await response.json();
                if (response.ok && data.success) {
                    showToast('Suppression annulée', 'success');
                    await loadResults();
                    displayResult(currentIndex);
                } else {
                    showToast('Erreur annulation: ' + (data.error || 'none'), 'error');
                }
            } catch (e) {
                showToast('Erreur réseau', 'error');
            }
            showLoading(false);
        }
        
        async function regenerateCSV() {
            showLoading(true);
            try {
                const response = await fetch('/api/regenerate-csv', {method: 'POST'});
                const data = await response.json();
                
                if (data.success) {
                    showToast(`CSV régénéré: ${data.count} lignes`, 'success');
                } else {
                    showToast('Erreur: ' + data.error, 'error');
                }
            } catch (e) {
                showToast('Erreur réseau', 'error');
            }
            showLoading(false);
        }
        
        function showLoading(show) {
            document.getElementById('loading').classList.toggle('show', show);
        }
        
        function showToast(message, type = 'info') {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.className = 'toast ' + type + ' show';
            setTimeout(() => toast.classList.remove('show'), 2000);
        }
        
        // ===== FILE NAVIGATION PANEL =====
        let navPanelVisible = true;
        let fileSearchQuery = '';
        
        function toggleNavPanel() {
            navPanelVisible = !navPanelVisible;
            const panel = document.getElementById('navPanel');
            panel.classList.toggle('collapsed', !navPanelVisible);
            
            // Mettre à jour le texte du bouton toggle
            const toggleBtn = panel.querySelector('.nav-toggle');
            toggleBtn.innerHTML = navPanelVisible ? '◀ Liste des fichiers' : '▶';
        }
        
        function renderFileList() {
            const container = document.getElementById('fileList');
            const searchQuery = fileSearchQuery.toLowerCase();
            
            let html = '';
            let visibleCount = 0;
            
            filteredResults.forEach((result, idx) => {
                const fileName = result.image.split('/').pop();
                const fileNameLower = fileName.toLowerCase();
                
                // Filtrer par recherche
                if (searchQuery && !fileNameLower.includes(searchQuery)) {
                    return;
                }
                
                visibleCount++;
                const status = result._corrections?.status || 'pending';
                const statusIcon = status === 'validated' ? '✓' : 
                                  status === 'rejected' ? '✗' : '○';
                const isActive = idx === currentIndex;
                
                html += `
                    <div class="file-item ${status} ${isActive ? 'active' : ''}" 
                         onclick="navigateToIndex(${idx})" 
                         data-file-index="${idx}">
                        <span class="file-status">${statusIcon}</span>
                        <span class="file-name" title="${fileName}">${fileName}</span>
                    </div>
                `;
            });
            
            container.innerHTML = html || '<div style="padding:10px;color:#888;font-size:0.8rem;">Aucun fichier trouvé</div>';
            
            // Mettre à jour les stats
            document.getElementById('navStats').textContent = 
                `${visibleCount}/${filteredResults.length} fichiers`;
            
            // Scroll vers l'élément actif
            scrollToActiveFile();
        }
        
        function scrollToActiveFile() {
            const activeItem = document.querySelector('.file-item.active');
            if (activeItem) {
                activeItem.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
            }
        }
        
        function filterFileList() {
            fileSearchQuery = document.getElementById('fileSearch').value;
            renderFileList();
        }
        
        function navigateToIndex(idx) {
            if (idx >= 0 && idx < filteredResults.length) {
                resetZoom();
                currentIndex = idx;
                displayResult(currentIndex);
                updateCounter();
                renderFileList();
            }
        }
        
        function gotoIndex() {
            const input = document.getElementById('gotoIndex');
            const idx = parseInt(input.value) - 1; // 1-based to 0-based
            if (idx >= 0 && idx < filteredResults.length) {
                navigateToIndex(idx);
                input.value = '';
            } else {
                showToast(`Index invalide (1-${filteredResults.length})`, 'error');
            }
        }
        
        function handleGotoKeypress(event) {
            if (event.key === 'Enter') {
                gotoIndex();
            }
        }
        
        // Modifier navigate pour mettre à jour la liste
        function navigate(delta) {
            resetZoom();
            const newIndex = currentIndex + delta;
            if (newIndex >= 0 && newIndex < filteredResults.length) {
                currentIndex = newIndex;
                displayResult(currentIndex);
                updateCounter();
                renderFileList();
            }
        }
        // ===== END FILE NAVIGATION PANEL =====
        
        // Raccourcis clavier
        document.addEventListener('keydown', (e) => {
            // Ignorer si on est dans un input
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                if (e.key === 'Escape') {
                    e.target.blur();
                }
                return;
            }
            
            // Raccourcis numériques pour les tags
            if (e.key >= '1' && e.key <= '8') {
                const tagIndex = parseInt(e.key) - 1;
                if (tagIndex < PREDEFINED_TAGS.length) {
                    toggleTag(PREDEFINED_TAGS[tagIndex].id);
                }
                return;
            }
            
            switch(e.key) {
                case 'ArrowLeft':
                case 'a':
                case 'A':
                    navigate(-1);
                    break;
                case 'ArrowRight':
                case 'd':
                case 'D':
                    navigate(1);
                    break;
                case 'v':
                case 'V':
                    validateWithTag();
                    break;
                case 'r':
                case 'R':
                    setStatus('rejected');
                    break;
                case 's':
                case 'S':
                    saveCorrections();
                    break;
                case 'f':
                case 'F':
                    cycleFilter();
                    break;
                case 't':
                case 'T':
                    document.getElementById('customTag').focus();
                    break;
                case 'l':
                case 'L':
                    toggleNavPanel();
                    break;
                case 'g':
                case 'G':
                    document.getElementById('gotoIndex').focus();
                    break;
            }
        });
        
        // Démarrage
        loadResults().then(() => {
            renderFileList();
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, predefined_tags=json.dumps(PREDEFINED_TAGS))


@app.route('/api/tags')
def api_tags():
    """Retourne les tags prédéfinis"""
    return jsonify(PREDEFINED_TAGS)


@app.route('/api/results')
def api_results():
    """Retourne tous les résultats"""
    results = load_all_results()
    return jsonify(results)


@app.route('/api/image')
def api_image():
    """Retourne l'image de debug pour une session (paramètre query `session_dir`).

    Utiliser une query string évite les problèmes d'encoding des slashes dans l'URL.
    """
    session_dir = request.args.get('session_dir')
    logger.info(f"API image request: session_dir={session_dir}")
    if not session_dir:
        logger.warning("No session_dir provided")
        return '', 404

    # session_dir est normalement décodé par Flask; garantir string
    image_path = get_debug_image_path(session_dir, 'final')
    logger.info(f"get_debug_image_path returned: {image_path}")
    if image_path:
        # Convertir en chemin absolu depuis la racine du projet
        abs_path = Path(image_path).resolve()
        logger.info(f"Absolute path: {abs_path}")
        if abs_path.exists():
            logger.info(f"Serving image: {abs_path}")
            # Déterminer le mime type en fonction de l'extension
            if str(abs_path).lower().endswith('.png'):
                mimetype = 'image/png'
            else:
                mimetype = 'image/jpeg'
            return send_file(str(abs_path), mimetype=mimetype)

    logger.warning(f"Image not found for session_dir={session_dir}, image_path={image_path}")
    return '', 404


@app.route('/api/save', methods=['POST'])
def api_save():
    """Sauvegarde les corrections"""
    data = request.json
    image_id = data.get('image_id')
    corrections = data.get('corrections', {})
    
    if save_correction(image_id, corrections):
        return jsonify({'success': True})
    
    return jsonify({'success': False, 'error': 'Image not found'}), 404


@app.route('/api/delete_spike', methods=['POST'])
def api_delete_spike():
    """Supprime un épi d'un résultat (par index) et sauvegarde le results.json"""
    data = request.json or {}
    image_id = data.get('image_id')
    spike_index = data.get('spike_index')

    if not image_id or spike_index is None:
        return jsonify({'success': False, 'error': 'Missing parameters'}), 400

    # Vérifier cache
    if image_id not in RESULTS_CACHE:
        return jsonify({'success': False, 'error': 'Image not found'}), 404

    result = RESULTS_CACHE[image_id]
    results_file = Path(result['_results_file'])

    try:
        spikes = result.get('spikes', [])
        if not (0 <= int(spike_index) < len(spikes)):
            return jsonify({'success': False, 'error': 'Invalid spike index'}), 400

        # Retirer et conserver une trace
        removed = spikes.pop(int(spike_index))

        # Mettre à jour le compteur
        result['spike_count'] = len(spikes)

        # Historique simple des modifications
        ver = result.get('_verification', {})
        history = ver.get('history', [])
        history.append({
            'action': 'delete_spike',
            'spike_index': int(spike_index),
            'removed': removed,
            'when': datetime.now().isoformat(),
            'by': 'verification_app'
        })
        ver['history'] = history
        result['_verification'] = ver

        # Sauvegarder (enlevant les clés internes)
        save_data = {k: v for k, v in result.items() if not k.startswith('_') or k == '_verification'}
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        # Mettre à jour le cache
        RESULTS_CACHE[image_id].update(result)

        # Ajouter les numéros d'épis en surimpression sur l'image annotée
        try:
            regenerated = add_spike_numbers_overlay(str(results_file.parent))
        except Exception:
            regenerated = False

        logger.info(f"Épi supprimé: {results_file} index={spike_index} regenerated={regenerated}")
        corrected_name = None
        if regenerated:
            corrected_name = str((results_file.parent / 'result_annotated_corrected.png').name)
        return jsonify({'success': True, 'image_regenerated': bool(regenerated), 'corrected_image': corrected_name})

    except Exception as e:
        logger.error(f"Erreur suppression épi: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/export')
def api_export():
    """Exporte toutes les corrections en CSV téléchargeable"""
    results = load_all_results()
    
    output = StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        'image_id', 'status', 'tags', 'sample_id_original', 'sample_id_corrected',
        'spike_count', 'notes', 'verified_at'
    ])
    
    for result in results:
        verification = result.get('_verification', {})
        corrections = result.get('_corrections', {})
        bag = result.get('bag', {})
        
        writer.writerow([
            Path(result['image']).stem,
            verification.get('status', corrections.get('status', 'pending')),
            ';'.join(verification.get('tags', corrections.get('tags', []))),
            bag.get('sample_id', ''),
            bag.get('sample_id_corrected', ''),
            len(result.get('spikes', [])),
            verification.get('notes', corrections.get('notes', '')),
            verification.get('verified_at', '')
        ])
    
    output.seek(0)
    
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=corrections_export.csv'}
    )


@app.route('/api/regenerate-csv', methods=['POST'])
def api_regenerate_csv():
    """Régénère le fichier results_summarised.csv complet avec les corrections"""
    try:
        results = load_all_results()
        
        # Chemin du CSV de sortie
        csv_path = Path(RESULTS_DIR) / 'results_summarised.csv'
        
        # D'abord, trouver le nombre maximum de tags
        max_tags = 0
        for result in results:
            verification = result.get('_verification', {})
            corrections = result.get('_corrections', {})
            tags = verification.get('tags', corrections.get('tags', []))
            max_tags = max(max_tags, len(tags))
        
        # Minimum 1 colonne tag même si pas de tags
        max_tags = max(max_tags, 1)
        
        rows = []
        for result in results:
            verification = result.get('_verification', {})
            corrections = result.get('_corrections', {})
            bag = result.get('bag', {})
            cal = result.get('calibration', {})
            
            # Utiliser les valeurs corrigées si disponibles
            sample_id = bag.get('sample_id_corrected') or bag.get('sample_id', '')
            
            # Récupérer les tags
            tags = verification.get('tags', corrections.get('tags', []))
            
            # Base row pour l'image (comme dans analyzer_obb.py)
            base_info = {
                'image_id': Path(result.get('image', '')).stem,
                'image_path': result.get('image', ''),
                'image_width': result.get('image_size', {}).get('width', ''),
                'image_height': result.get('image_size', {}).get('height', ''),
                'pixel_per_mm': cal.get('pixel_per_mm', ''),
                'ruler_detected': cal.get('ruler_detected', False),
                'ruler_length_px': cal.get('ruler_length_px', ''),
                'spike_count': len(result.get('spikes', [])),
                # Sachet
                'bag_detected': bag.get('detected', False),
                'sample_id': sample_id,
                'bac': bag.get('bac', ''),
                'ligne': bag.get('ligne', ''),
                'colonne': bag.get('colonne', ''),
                'bag_confidence': bag.get('confidence', ''),
                'bag_complete': bag.get('complete', ''),
                # Vérification
                'verification_status': verification.get('status', corrections.get('status', 'pending')),
                'verification_notes': verification.get('notes', corrections.get('notes', '')),
            }
            
            # Ajouter les colonnes de tags individuelles
            for i in range(max_tags):
                base_info[f'tag_{i+1}'] = tags[i] if i < len(tags) else ''
            
            # Ajouter une ligne par épi
            spikes = result.get('spikes', [])
            if spikes:
                for i, spike in enumerate(spikes):
                    m = spike.get('measurements', {})
                    corr = spike.get('corrections', {})
                    
                    row = base_info.copy()
                    row.update({
                        'spike_id': spike.get('id', i+1),
                        # Longueurs spike (sans barbes)
                        'spike_length_px': m.get('spike_length_pixels', m.get('length_pixels', '')),
                        'spike_length_mm': corr.get('length') or m.get('spike_length_mm', m.get('length_mm', '')),
                        'spike_width_px': m.get('spike_width_pixels', m.get('width_pixels', '')),
                        'spike_width_mm': m.get('spike_width_mm', m.get('width_mm', '')),
                        # Longueurs whole_spike (avec barbes)
                        'whole_spike_length_px': m.get('whole_spike_length_pixels', ''),
                        'whole_spike_length_mm': m.get('whole_spike_length_mm', ''),
                        # Barbes
                        'awns_length_px': m.get('awns_length_pixels', ''),
                        'awns_length_mm': m.get('awns_length_mm', ''),
                        'has_awns': m.get('has_awns', False),
                        # Autres mesures
                        'area_px': m.get('area_pixels', ''),
                        'area_mm2': m.get('area_mm2', ''),
                        'perimeter_px': m.get('perimeter_pixels', ''),
                        'perimeter_mm': m.get('perimeter_mm', ''),
                        'aspect_ratio': m.get('aspect_ratio', ''),
                        'angle_degrees': m.get('angle_degrees', ''),
                        # Épillets
                        'spikelet_count': corr.get('spikelets') or spike.get('spikelet_count', ''),
                        'spikelet_method': spike.get('spikelet_method', ''),
                        'spikelet_confidence': spike.get('spikelet_confidence', ''),
                        # Coordonnées
                        'center_x': m.get('center_x', ''),
                        'center_y': m.get('center_y', ''),
                        'confidence': m.get('confidence', spike.get('confidence', '')),
                    })
                    rows.append(row)
            else:
                # Image sans épi détecté
                row = base_info.copy()
                row.update({
                    'spike_id': '',
                    'spike_length_px': '',
                    'spike_length_mm': '',
                    'spike_width_px': '',
                    'spike_width_mm': '',
                    'whole_spike_length_px': '',
                    'whole_spike_length_mm': '',
                    'awns_length_px': '',
                    'awns_length_mm': '',
                    'has_awns': '',
                    'area_px': '',
                    'area_mm2': '',
                    'perimeter_px': '',
                    'perimeter_mm': '',
                    'aspect_ratio': '',
                    'angle_degrees': '',
                    'spikelet_count': '',
                    'spikelet_method': '',
                    'spikelet_confidence': '',
                    'center_x': '',
                    'center_y': '',
                    'confidence': '',
                })
                rows.append(row)
        
        # Définir l'ordre des colonnes explicitement (comme analyzer_obb.py)
        if rows:
            # Colonnes de base dans l'ordre souhaité
            base_columns = [
                'image_id', 'image_path', 'image_width', 'image_height',
                'pixel_per_mm', 'ruler_detected', 'ruler_length_px', 'spike_count',
                'bag_detected', 'sample_id', 'bac', 'ligne', 'colonne', 'bag_confidence', 'bag_complete',
                'verification_status', 'verification_notes'
            ]
            # Colonnes de tags
            tag_columns = [f'tag_{i+1}' for i in range(max_tags)]
            # Colonnes d'épis
            spike_columns = [
                'spike_id', 'spike_length_px', 'spike_length_mm', 'spike_width_px', 'spike_width_mm',
                'whole_spike_length_px', 'whole_spike_length_mm', 'awns_length_px', 'awns_length_mm', 'has_awns',
                'area_px', 'area_mm2', 'perimeter_px', 'perimeter_mm', 'aspect_ratio', 'angle_degrees',
                'spikelet_count', 'spikelet_method', 'spikelet_confidence',
                'center_x', 'center_y', 'confidence'
            ]
            fieldnames = base_columns + tag_columns + spike_columns
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            
            logger.info(f"CSV régénéré: {csv_path} ({len(rows)} lignes, {max_tags} colonnes tags)")
            return jsonify({'success': True, 'count': len(rows), 'path': str(csv_path)})
        else:
            return jsonify({'success': False, 'error': 'Aucun résultat à exporter'})
            
    except Exception as e:
        logger.error(f"Erreur régénération CSV: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/regenerate_image', methods=['POST'])
def api_regenerate_image():
    """Régénère l'image annotée corrigée pour une session donnée (body: {session_dir})."""
    data = request.json or {}
    session_dir = data.get('session_dir')
    if not session_dir:
        return jsonify({'success': False, 'error': 'Missing session_dir'}), 400

    try:
        ok = add_spike_numbers_overlay(session_dir)
        if ok:
            corrected_name = str(Path(session_dir).joinpath('result_annotated_corrected.png').name)
            return jsonify({'success': True, 'corrected_image': corrected_name})
        else:
            return jsonify({'success': False, 'error': 'Regeneration failed'})
    except Exception as e:
        logger.error(f"Erreur API regenerate_image: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/undo_delete_spike', methods=['POST'])
def api_undo_delete_spike():
    """Restaure la dernière suppression d'épi enregistrée dans _verification.history pour l'image donnée.

    Body: {image_id: 'GOPRxxxx'}
    """
    data = request.json or {}
    image_id = data.get('image_id')
    if not image_id:
        return jsonify({'success': False, 'error': 'Missing image_id'}), 400

    if image_id not in RESULTS_CACHE:
        return jsonify({'success': False, 'error': 'Image not found'}), 404

    result = RESULTS_CACHE[image_id]
    results_file = Path(result['_results_file'])

    try:
        ver = result.get('_verification', {})
        history = ver.get('history', [])
        # Find last delete_spike action from history (search from end)
        last_idx = None
        for idx in range(len(history)-1, -1, -1):
            if history[idx].get('action') == 'delete_spike':
                last_idx = idx
                break

        if last_idx is None:
            return jsonify({'success': False, 'error': 'No delete_spike in history'}), 400

        entry = history.pop(last_idx)
        removed = entry.get('removed')
        spike_index = entry.get('spike_index')

        if removed is None or spike_index is None:
            return jsonify({'success': False, 'error': 'History entry incomplete'}), 400

        # Re-insert removed spike at the stored index (or append if out of range)
        spikes = result.get('spikes', [])
        insert_idx = int(spike_index)
        if insert_idx < 0 or insert_idx > len(spikes):
            spikes.append(removed)
        else:
            spikes.insert(insert_idx, removed)

        # Update counters and verification
        result['spikes'] = spikes
        result['spike_count'] = len(spikes)
        ver['history'] = history
        result['_verification'] = ver

        # Save results.json
        save_data = {k: v for k, v in result.items() if not k.startswith('_') or k == '_verification'}
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        # Update cache
        RESULTS_CACHE[image_id].update(result)

        # Regenerate corrected image with spike numbers overlay
        regenerated = add_spike_numbers_overlay(str(results_file.parent))
        corrected_name = None
        if regenerated:
            corrected_name = str((results_file.parent / 'result_annotated_corrected.png').name)

        logger.info(f"Undo delete_spike: {results_file} restored_index={spike_index} regenerated={regenerated}")
        return jsonify({'success': True, 'restored_index': spike_index, 'corrected_image': corrected_name})

    except Exception as e:
        logger.error(f"Erreur undo delete spike: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def main():
    global RESULTS_DIR
    
    parser = argparse.ArgumentParser(description='Application de vérification des résultats')
    parser.add_argument('--output', '-o', default='output',
                        help='Dossier contenant les résultats (défaut: output)')
    parser.add_argument('--port', '-p', type=int, default=5000,
                        help='Port du serveur (défaut: 5000)')
    parser.add_argument('--host', default='127.0.0.1',
                        help='Adresse du serveur (défaut: 127.0.0.1)')
    parser.add_argument('--debug', action='store_true',
                        help='Mode debug Flask')
    
    args = parser.parse_args()
    
    # Convertir en chemin absolu pour éviter les problèmes de répertoire de travail
    RESULTS_DIR = str(Path(args.output).resolve())
    
    if not Path(RESULTS_DIR).exists():
        logger.error(f"Dossier de résultats non trouvé: {RESULTS_DIR}")
        return 1
    
    # Compter les résultats
    results_count = len(list(Path(RESULTS_DIR).glob('**/results.json')))
    logger.info(f"Dossier de résultats: {RESULTS_DIR}")
    logger.info(f"Résultats trouvés: {results_count}")
    
    print(f"\n{'='*60}")
    print(f"🌾 Wheat Spike Analyzer - Vérification")
    print(f"{'='*60}")
    print(f"Interface: http://{args.host}:{args.port}")
    print(f"Résultats: {results_count} images")
    print(f"\nRaccourcis clavier:")
    print(f"  ← →    Navigation entre images")
    print(f"  V      Valider l'image (ajoute tag 'Validé')")
    print(f"  R      Rejeter l'image")
    print(f"  S      Sauvegarder les corrections")
    print(f"  F      Filtrer les non-validés")
    print(f"  T      Focus sur le champ tag personnalisé")
    print(f"  1-8    Toggle tag rapide")
    print(f"\nTags disponibles:")
    for tag in PREDEFINED_TAGS:
        print(f"  {tag['shortcut']}  {tag['label']}")
    print(f"{'='*60}\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
