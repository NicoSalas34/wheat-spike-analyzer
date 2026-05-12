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
    - Calibration incorrecte
    - Mauvaise graduation
    - Sachet absent
    - Épi sans barbes
    - Épillets incorrects - pas améliorable
    - Mauvais angle insertion
    - Mauvais rachis

Usage:
    python app/verification_app.py --output output/
    python app/verification_app.py --output output/ --port 5001
"""

import argparse
import csv
import json
import logging
import os
import shutil
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
    {"id": "wrong_graduation", "label": "Mauvaise graduation", "color": "#e879f9", "shortcut": "9"},
    {"id": "bag_missing", "label": "Sachet absent", "color": "#fcd34d", "shortcut": ""},
    {"id": "spike_no_awns", "label": "Épi sans barbes", "color": "#34d399", "shortcut": ""},
    {"id": "spikelets_unimprovable", "label": "Épillets incorrects - pas améliorable", "color": "#c084fc", "shortcut": ""},
    {"id": "wrong_insertion_angle", "label": "Mauvais angle insertion", "color": "#fb7185", "shortcut": ""},
    {"id": "wrong_rachis", "label": "Mauvais rachis", "color": "#22d3ee", "shortcut": ""},
]


def load_all_results() -> List[Dict]:
    """Charge tous les résultats depuis le dossier output"""
    global RESULTS_CACHE
    
    results = []
    results_dir = Path(RESULTS_DIR)

    if not results_dir.exists():
        RESULTS_CACHE = {}
        return results
    
    for source_index, results_file in enumerate(sorted(results_dir.glob('**/results.json'))):
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            # Ajouter le chemin du dossier de session
            session_dir = results_file.parent
            data['_session_dir'] = str(session_dir)
            data['_results_file'] = str(results_file)
            data['_source_index'] = source_index
            
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


def _normalize_csv_value(value) -> str:
    """Normalize a CSV field value to a trimmed string."""
    if value is None:
        return ''
    return str(value).strip()


def _parse_csv_spike_id(value) -> Optional[int]:
    """Parse spike_id values from CSV (supports integer-like text such as '1' or '1.0')."""
    text = _normalize_csv_value(value)
    if not text:
        return None

    try:
        return int(text)
    except ValueError:
        try:
            numeric = float(text.replace(',', '.'))
        except ValueError:
            return None
        if numeric.is_integer():
            return int(numeric)

    return None


def _extract_csv_image_id(row: Dict) -> str:
    """Extract an image id from a CSV row using common column names."""
    image_id = _normalize_csv_value(row.get('image_id'))
    if image_id:
        return Path(image_id.replace('\\', '/')).stem

    image_name = _normalize_csv_value(row.get('image'))
    if image_name:
        return Path(image_name.replace('\\', '/')).stem

    image_path = _normalize_csv_value(row.get('image_path'))
    if image_path:
        return Path(image_path.replace('\\', '/')).stem

    return ''


def _is_clear_marker(value: str) -> bool:
    """Return True when a CSV value explicitly asks to clear a field."""
    return _normalize_csv_value(value).upper() in {'__CLEAR__', '[CLEAR]', 'CLEAR'}


def _dedupe_keep_order(values: List[str]) -> List[str]:
    """Deduplicate strings while preserving insertion order."""
    unique_values: List[str] = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)
    return unique_values


def _split_csv_tags(raw_tags: str) -> List[str]:
    """Split tags from a CSV cell and normalize whitespace."""
    text = _normalize_csv_value(raw_tags)
    if not text:
        return []

    # Prefer ';' (CSV export format), fallback to ',' when needed.
    separator = ';' if ';' in text else ','
    parts = [part.strip() for part in text.split(separator)]
    return _dedupe_keep_order([part for part in parts if part])


def _normalize_verification_status(value: str) -> str:
    """Normalize verification status values read from CSV."""
    normalized = _normalize_csv_value(value).lower()
    if not normalized:
        return ''

    status_map = {
        'validated': 'validated',
        'valide': 'validated',
        'validé': 'validated',
        'validate': 'validated',
        'rejected': 'rejected',
        'rejete': 'rejected',
        'rejeté': 'rejected',
        'reject': 'rejected',
        'pending': 'pending',
        'awaiting': 'pending',
        'en attente': 'pending',
    }
    return status_map.get(normalized, normalized)


def apply_back_modifications_from_csv_content(csv_content: str, dry_run: bool = False) -> Dict:
    """Apply CSV-based back modifications to results.json files.

    Rules:
        - sample_id_corrected: if provided in CSV, update bag.sample_id_corrected.
            To clear it, use an explicit value: __CLEAR__, [CLEAR], or CLEAR.
        - verification_status: if provided in CSV, update _verification.status.
        - tags/tag_1..tag_n: if provided in CSV, update _verification.tags.
        - spikes: for each image, keep only spikes whose `spike_id` still exists in CSV rows.
            If an image has rows but no numeric spike_id, all spikes are removed.
    """
    load_all_results()

    reader = csv.DictReader(StringIO(csv_content))
    if not reader.fieldnames:
        raise ValueError('CSV vide ou en-tête introuvable')

    fieldnames = list(reader.fieldnames)
    fieldnames_set = set(fieldnames)
    has_sample_col = 'sample_id_corrected' in fieldnames_set
    has_spike_col = 'spike_id' in fieldnames_set
    has_status_col = 'verification_status' in fieldnames_set
    has_tags_col = 'tags' in fieldnames_set
    tag_columns = [field for field in fieldnames if field.lower().startswith('tag_')]
    has_split_tag_cols = bool(tag_columns)

    if not any([has_sample_col, has_spike_col, has_status_col, has_tags_col, has_split_tag_cols]):
        raise ValueError(
            "Le CSV doit contenir au moins une des colonnes: "
            "sample_id_corrected, spike_id, verification_status, tags ou tag_*"
        )

    csv_by_image: Dict[str, Dict] = {}
    rows_total = 0
    rows_without_image_id = 0

    for row in reader:
        rows_total += 1
        image_id = _extract_csv_image_id(row)
        if not image_id:
            rows_without_image_id += 1
            continue

        entry = csv_by_image.setdefault(
            image_id,
            {
                'sample_values': [],
                'force_clear_sample': False,
                'keep_spike_ids': set(),
                'saw_spike_column': False,
                'has_numeric_spike_ids': False,
                'status_values': [],
                'saw_status_column': False,
                'tags_values': [],
                'saw_tags_column': False,
                'force_clear_tags': False,
            },
        )

        if has_sample_col:
            sample_value = _normalize_csv_value(row.get('sample_id_corrected'))
            if sample_value:
                if _is_clear_marker(sample_value):
                    entry['force_clear_sample'] = True
                else:
                    entry['sample_values'].append(sample_value)

        if has_status_col:
            entry['saw_status_column'] = True
            status_value = _normalize_csv_value(row.get('verification_status'))
            if status_value:
                if _is_clear_marker(status_value):
                    entry['status_values'].append('pending')
                else:
                    normalized_status = _normalize_verification_status(status_value)
                    if normalized_status:
                        entry['status_values'].append(normalized_status)

        if has_tags_col or has_split_tag_cols:
            entry['saw_tags_column'] = True
            row_tags: List[str] = []

            if has_tags_col:
                tags_value = _normalize_csv_value(row.get('tags'))
                if tags_value:
                    if _is_clear_marker(tags_value):
                        entry['force_clear_tags'] = True
                    else:
                        row_tags.extend(_split_csv_tags(tags_value))

            for tag_column in tag_columns:
                tag_value = _normalize_csv_value(row.get(tag_column))
                if not tag_value:
                    continue
                if _is_clear_marker(tag_value):
                    entry['force_clear_tags'] = True
                    continue
                row_tags.extend(_split_csv_tags(tag_value))

            if row_tags:
                entry['tags_values'].extend(row_tags)

        if has_spike_col:
            entry['saw_spike_column'] = True
            spike_id = _parse_csv_spike_id(row.get('spike_id'))
            if spike_id is not None:
                entry['keep_spike_ids'].add(spike_id)
                entry['has_numeric_spike_ids'] = True

    summary = {
        'rows_total': rows_total,
        'rows_without_image_id': rows_without_image_id,
        'images_in_csv': len(csv_by_image),
        'images_found': 0,
        'images_updated': 0,
        'sample_id_updated': 0,
        'sample_id_cleared': 0,
        'verification_status_updated': 0,
        'tags_updated': 0,
        'tags_cleared': 0,
        'spikes_removed_total': 0,
        'images_not_found': 0,
        'images_not_found_examples': [],
        'sample_id_conflict_count': 0,
        'sample_id_conflicts': [],
        'verification_status_conflict_count': 0,
        'verification_status_conflicts': [],
        'dry_run': bool(dry_run),
    }

    missing_images: List[str] = []
    sample_conflicts: List[Dict] = []
    status_conflicts: List[Dict] = []

    for image_id, entry in csv_by_image.items():
        result = RESULTS_CACHE.get(image_id)
        if result is None:
            missing_images.append(image_id)
            continue

        summary['images_found'] += 1

        result_changed = False
        sample_was_updated = False
        sample_was_cleared = False
        status_was_updated = False
        tags_were_updated = False
        tags_were_cleared = False
        spikes_removed_for_image = 0

        bag = result.get('bag')
        if not isinstance(bag, dict):
            bag = {}
            result['bag'] = bag

        verification = result.get('_verification')
        if not isinstance(verification, dict):
            verification = {}
        corrections = result.get('_corrections') or {}

        verification.setdefault('status', corrections.get('status', 'pending'))
        verification.setdefault('tags', corrections.get('tags', []))
        verification.setdefault('notes', corrections.get('notes', ''))

        # sample_id_corrected update
        if has_sample_col:
            unique_samples = _dedupe_keep_order(entry['sample_values'])

            target_sample = None
            has_conflict = len(unique_samples) > 1

            if has_conflict:
                sample_conflicts.append({'image_id': image_id, 'values': unique_samples})
            elif len(unique_samples) == 1:
                target_sample = unique_samples[0]
            elif entry['force_clear_sample']:
                target_sample = ''

            if target_sample is not None and not has_conflict:
                current_sample = _normalize_csv_value(bag.get('sample_id_corrected'))

                if target_sample:
                    if current_sample != target_sample:
                        bag['sample_id_corrected'] = target_sample

                        parts = [p.strip() for p in target_sample.split('-', 2)]
                        if len(parts) == 3:
                            bag['bac'], bag['ligne'], bag['colonne'] = parts

                        sample_was_updated = True
                        result_changed = True
                else:
                    if current_sample:
                        bag.pop('sample_id_corrected', None)
                        sample_was_cleared = True
                        result_changed = True

        # verification_status update
        if has_status_col and entry['saw_status_column']:
            unique_statuses = _dedupe_keep_order(entry['status_values'])
            target_status = None
            has_conflict = len(unique_statuses) > 1

            if has_conflict:
                status_conflicts.append({'image_id': image_id, 'values': unique_statuses})
            elif len(unique_statuses) == 1:
                target_status = unique_statuses[0]
            else:
                target_status = 'pending'

            if target_status is not None and not has_conflict:
                current_status = _normalize_verification_status(
                    verification.get('status', corrections.get('status', 'pending'))
                )
                if current_status != target_status:
                    verification['status'] = target_status
                    status_was_updated = True
                    result_changed = True

        # tags update from `tags` and/or `tag_*` columns
        if (has_tags_col or has_split_tag_cols) and entry['saw_tags_column']:
            if entry['force_clear_tags']:
                target_tags: List[str] = []
            else:
                target_tags = _dedupe_keep_order(entry['tags_values'])

            current_tags_raw = verification.get('tags', corrections.get('tags', []))
            if isinstance(current_tags_raw, list):
                current_tags = _dedupe_keep_order([
                    _normalize_csv_value(tag)
                    for tag in current_tags_raw
                    if _normalize_csv_value(tag)
                ])
            else:
                current_tags = _split_csv_tags(_normalize_csv_value(current_tags_raw))

            if current_tags != target_tags:
                verification['tags'] = target_tags
                tags_were_updated = True
                if not target_tags:
                    tags_were_cleared = True
                result_changed = True

        # Spike pruning based on remaining spike_id rows in CSV
        if has_spike_col and entry['saw_spike_column']:
            spikes = list(result.get('spikes') or [])

            if entry['has_numeric_spike_ids']:
                keep_ids = entry['keep_spike_ids']
                filtered_spikes = []

                for idx, spike in enumerate(spikes):
                    spike_identifier = _parse_csv_spike_id(spike.get('id'))
                    if spike_identifier is None:
                        spike_identifier = idx + 1

                    if spike_identifier in keep_ids:
                        filtered_spikes.append(spike)

                spikes_removed_for_image = len(spikes) - len(filtered_spikes)
                if spikes_removed_for_image > 0:
                    result['spikes'] = filtered_spikes
                    result['spike_count'] = len(filtered_spikes)
                    result_changed = True
            else:
                if spikes:
                    spikes_removed_for_image = len(spikes)
                    result['spikes'] = []
                    result['spike_count'] = 0
                    result_changed = True

        if not result_changed:
            continue

        summary['images_updated'] += 1
        summary['spikes_removed_total'] += spikes_removed_for_image
        if sample_was_updated:
            summary['sample_id_updated'] += 1
        if sample_was_cleared:
            summary['sample_id_cleared'] += 1
        if status_was_updated:
            summary['verification_status_updated'] += 1
        if tags_were_updated:
            summary['tags_updated'] += 1
        if tags_were_cleared:
            summary['tags_cleared'] += 1

        if status_was_updated or tags_were_updated:
            verification['verified_at'] = datetime.now().isoformat()
            verification['verified_by'] = 'verification_app_csv_import'

        # Track bulk operation in verification history
        history = verification.get('history', []) or []
        history.append({
            'action': 'apply_csv_back_modifications',
            'when': datetime.now().isoformat(),
            'by': 'verification_app',
            'spikes_removed': spikes_removed_for_image,
            'sample_id_corrected': bag.get('sample_id_corrected', ''),
            'verification_status': verification.get('status', 'pending'),
            'tags': verification.get('tags', []),
        })
        verification['history'] = history
        result['_verification'] = verification

        if dry_run:
            continue

        results_file = Path(result['_results_file'])
        save_data = {k: v for k, v in result.items() if not k.startswith('_') or k == '_verification'}
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        # Refresh corrected overlay if spike list changed
        if spikes_removed_for_image > 0:
            try:
                add_spike_numbers_overlay(str(results_file.parent))
            except Exception as overlay_error:
                logger.warning(f"Impossible de régénérer l'image corrigée pour {results_file.parent}: {overlay_error}")

    summary['images_not_found'] = len(missing_images)
    summary['images_not_found_examples'] = missing_images[:20]
    summary['sample_id_conflict_count'] = len(sample_conflicts)
    summary['sample_id_conflicts'] = sample_conflicts[:20]
    summary['verification_status_conflict_count'] = len(status_conflicts)
    summary['verification_status_conflicts'] = status_conflicts[:20]

    if not dry_run:
        load_all_results()

    return summary


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
            flex-wrap: wrap;
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

        .header-btn.danger {
            background: #e94560;
        }

        .header-btn.danger:hover {
            background: #d03050;
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
                    <optgroup label="── Échantillons ──">
                        <option value="duplicate_sample_corrected">🧩 Sample corrigé en doublon</option>
                    </optgroup>
                </select>
            </div>
            <div class="filter-dropdown">
                <select class="filter-select" id="sortSelect" onchange="applyAdvancedFilter()" title="Mode de tri">
                    <option value="original">↕ Ordre d'origine</option>
                    <option value="sample_id_corrected">🧩 Sample ID corrigé</option>
                </select>
            </div>
            <button class="header-btn" onclick="triggerCsvBackImport()">
                📥 CSV → JSON
            </button>
            <button class="header-btn export" onclick="regenerateCSV()">
                📊 Régénérer CSV
            </button>
            <button class="header-btn" onclick="regenerateCurrentImage()">
                🔁 Régénérer image
            </button>
            <button class="header-btn danger" onclick="deleteCurrentSessionDirectory()">
                🗑️ Supprimer session affichée
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

    <input type="file" id="csvBackImportInput" accept=".csv,text/csv" style="display:none;" onchange="handleCsvBackImport(event)">
    
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

        function getSourceIndex(result) {
            return Number.isFinite(result._source_index) ? result._source_index : 0;
        }

        function compareText(left, right) {
            const leftValue = (left || '').trim();
            const rightValue = (right || '').trim();

            if (!leftValue && rightValue) return 1;
            if (leftValue && !rightValue) return -1;

            return leftValue.localeCompare(rightValue, undefined, { numeric: true, sensitivity: 'base' });
        }

        function getCorrectedSampleIdSortKey(result) {
            const bag = result.bag || {};
            return (bag.sample_id_corrected || '').trim();
        }

        function getSampleIdSortKey(result) {
            const bag = result.bag || {};
            return (bag.sample_id_corrected || bag.sample_id || '').trim();
        }

        function buildCorrectedSampleIdCounts(items) {
            const counts = new Map();

            items.forEach((item) => {
                const key = getCorrectedSampleIdSortKey(item);
                if (!key) return;
                counts.set(key, (counts.get(key) || 0) + 1);
            });

            return counts;
        }

        function sortFilteredResults(items) {
            const sortValue = document.getElementById('sortSelect')?.value || 'original';
            if (sortValue === 'original') {
                return items;
            }

            if (sortValue === 'sample_id_corrected') {
                return [...items].sort((left, right) => {
                    const keyComparison = compareText(getSampleIdSortKey(left), getSampleIdSortKey(right));
                    if (keyComparison !== 0) {
                        return keyComparison;
                    }

                    return getSourceIndex(left) - getSourceIndex(right);
                });
            }

            return [...items].sort((left, right) => {
                const keyComparison = compareText(getSampleIdSortKey(left), getSampleIdSortKey(right));
                if (keyComparison !== 0) {
                    return keyComparison;
                }

                return getSourceIndex(left) - getSourceIndex(right);
            });
        }
        
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
            const currentImagePath = filteredResults[currentIndex]?.image || null;
            const filterValue = document.getElementById('filterSelect').value;
            const correctedSampleCounts = buildCorrectedSampleIdCounts(results);
            
            filteredResults = sortFilteredResults(results.filter(r => {
                const corrections = r._corrections || {};
                const status = corrections.status || 'pending';
                const tags = corrections.tags || [];
                const cal = r.calibration || {};
                const bag = r.bag || {};
                const spikes = r.spikes || [];
                const correctedSampleId = getCorrectedSampleIdSortKey(r);
                
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

                    // Échantillons
                    case 'duplicate_sample_corrected':
                        return !!correctedSampleId && (correctedSampleCounts.get(correctedSampleId) || 0) > 1;
                    
                    default:
                        return true;
                }
            }));
            
            if (currentImagePath) {
                const newIndex = filteredResults.findIndex(r => r.image === currentImagePath);
                currentIndex = newIndex >= 0 ? newIndex : Math.min(currentIndex, Math.max(0, filteredResults.length - 1));
            } else {
                currentIndex = Math.min(currentIndex, Math.max(0, filteredResults.length - 1));
            }

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
            const mainFilters = ['all', 'awaiting', 'pending', 'duplicate_sample_corrected', 'no_ruler', 'no_spikes', 'no_bag'];
            const currentIdx = mainFilters.indexOf(select.value);
            const nextIdx = (currentIdx + 1) % mainFilters.length;
            select.value = mainFilters[nextIdx];
            applyAdvancedFilter();
            
            // Afficher quel filtre est actif
            const filterNames = {
                'all': 'Tous',
                'awaiting': 'En attente',
                'pending': 'Non validés',
                'duplicate_sample_corrected': 'Samples corrigés en doublon',
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

        function triggerCsvBackImport() {
            const input = document.getElementById('csvBackImportInput');
            if (!input) return;
            input.value = '';
            input.click();
        }

        async function handleCsvBackImport(event) {
            const file = event?.target?.files?.[0];
            if (!file) return;

            const firstConfirm = confirm(
                `Appliquer les modifications du CSV "${file.name}" sur tous les results.json correspondants ?`
            );
            if (!firstConfirm) {
                event.target.value = '';
                return;
            }

            showLoading(true);
            try {
                const formData = new FormData();
                formData.append('file', file);
                formData.append('confirm', 'true');

                const response = await fetch('/api/apply_csv_back_modifications', {
                    method: 'POST',
                    body: formData,
                });

                const data = await response.json();
                if (response.ok && data.success) {
                    showToast(
                        `CSV appliqué: ${data.images_updated} images modifiées, ${data.spikes_removed_total} épis supprimés, ${data.verification_status_updated || 0} statuts, ${data.tags_updated || 0} tags`,
                        'success'
                    );

                    if (data.sample_id_conflict_count > 0) {
                        showToast(
                            `${data.sample_id_conflict_count} conflit(s) sample_id_corrected ignoré(s)`,
                            'info'
                        );
                    }

                    if ((data.verification_status_conflict_count || 0) > 0) {
                        showToast(
                            `${data.verification_status_conflict_count} conflit(s) de statut ignoré(s)`,
                            'info'
                        );
                    }

                    await loadResults();
                } else {
                    showToast(data.error || "Échec de l'application du CSV", 'error');
                }
            } catch (e) {
                showToast('Erreur réseau', 'error');
            } finally {
                showLoading(false);
                event.target.value = '';
            }
        }

        async function deleteCurrentSessionDirectory() {
            const result = filteredResults[currentIndex];
            if (!result) return showToast('Aucun résultat sélectionné', 'error');

            const sessionDir = result._session_dir;
            const fileName = (result.image || '').split('/').pop() || "l'image affichée";
            if (!sessionDir) return showToast('Session introuvable', 'error');

            const firstConfirm = confirm(`Supprimer le sous-répertoire de ${fileName} ? Cette action est irréversible.`);
            if (!firstConfirm) return;

            const secondConfirm = confirm(`Confirmer la suppression du dossier:\n${sessionDir}`);
            if (!secondConfirm) return;

            showLoading(true);
            try {
                const response = await fetch('/api/delete_current_session_dir', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_dir: sessionDir})
                });

                const data = await response.json();
                if (response.ok && data.success) {
                    showToast('Session supprimée', 'success');
                    await loadResults();
                } else {
                    showToast(data.error || 'Suppression impossible', 'error');
                }
            } catch (e) {
                showToast('Erreur réseau', 'error');
            } finally {
                showLoading(false);
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
            if (e.key >= '1' && e.key <= '9') {
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


@app.route('/api/apply_csv_back_modifications', methods=['POST'])
def api_apply_csv_back_modifications():
    """Applique des rétro-modifications JSON à partir d'un CSV (upload multipart)."""
    uploaded = request.files.get('file')
    if uploaded is None:
        return jsonify({'success': False, 'error': 'Missing CSV file'}), 400

    confirm = _normalize_csv_value(request.form.get('confirm')).lower()
    if confirm not in {'1', 'true', 'yes', 'oui'}:
        return jsonify({'success': False, 'error': 'Confirmation required'}), 400

    dry_run = _normalize_csv_value(request.form.get('dry_run')).lower() in {'1', 'true', 'yes', 'oui'}

    try:
        csv_bytes = uploaded.read()
        if not csv_bytes:
            return jsonify({'success': False, 'error': 'CSV file is empty'}), 400

        csv_content = None
        for encoding in ('utf-8-sig', 'utf-8', 'latin-1'):
            try:
                csv_content = csv_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if csv_content is None:
            return jsonify({'success': False, 'error': 'Unable to decode CSV (expected UTF-8 or latin-1)'}), 400

        summary = apply_back_modifications_from_csv_content(csv_content, dry_run=dry_run)
        logger.info(
            "CSV back-modifications applied: "
            f"updated={summary.get('images_updated', 0)} "
            f"spikes_removed={summary.get('spikes_removed_total', 0)} "
            f"sample_updated={summary.get('sample_id_updated', 0)} "
            f"status_updated={summary.get('verification_status_updated', 0)} "
            f"tags_updated={summary.get('tags_updated', 0)} "
            f"sample_cleared={summary.get('sample_id_cleared', 0)} dry_run={dry_run}"
        )

        return jsonify({'success': True, **summary})

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Erreur application CSV back-modifications: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/delete_current_session_dir', methods=['POST'])
def api_delete_current_session_dir():
    """Supprime le sous-répertoire de session correspondant à l'image affichée."""
    data = request.json or {}
    session_dir = data.get('session_dir')
    if not session_dir:
        return jsonify({'success': False, 'error': 'Missing session_dir'}), 400

    global RESULTS_CACHE

    output_dir = Path(RESULTS_DIR).resolve()
    session_path = Path(session_dir).resolve()

    if not output_dir.exists():
        RESULTS_CACHE = {}
        return jsonify({'success': False, 'error': 'Output directory not found'}), 404

    try:
        session_path.relative_to(output_dir)
    except ValueError:
        return jsonify({'success': False, 'error': 'Session directory must be inside output directory'}), 400

    if session_path == output_dir:
        return jsonify({'success': False, 'error': 'Refusing to delete output root directory'}), 400

    if not session_path.exists() or not session_path.is_dir():
        return jsonify({'success': False, 'error': 'Session directory not found'}), 404

    if not (session_path / 'results.json').exists():
        return jsonify({'success': False, 'error': 'Invalid session directory: results.json missing'}), 400

    try:
        shutil.rmtree(session_path)
        RESULTS_CACHE = {}
        logger.info(f"Sous-répertoire de session supprimé: {session_path}")
        return jsonify({'success': True, 'deleted': True, 'path': str(session_path)})
    except Exception as e:
        logger.error(f"Erreur suppression sous-répertoire session {session_path}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/regenerate-csv', methods=['POST'])
def api_regenerate_csv():
    """Régénère results_summary.csv avec schéma complet + colonnes de vérification."""
    try:
        results = load_all_results()

        # Écrire le CSV standard (même nom que le pipeline batch)
        csv_path = Path(RESULTS_DIR) / 'results_summary.csv'

        # Nombre max de tags pour générer tag_1...tag_n
        max_tags = 1
        for result in results:
            ver = result.get('_verification', {}) or {}
            cor = result.get('_corrections', {}) or {}
            tags = ver.get('tags', cor.get('tags', [])) or []
            max_tags = max(max_tags, len(tags))

        rows = []
        for result in results:
            ver = result.get('_verification', {}) or {}
            cor = result.get('_corrections', {}) or {}
            bag = result.get('bag', {}) or {}
            cal = result.get('calibration', {}) or {}

            sample_id_original = bag.get('sample_id', '')
            sample_id_corrected = bag.get('sample_id_corrected', '')
            sample_id = sample_id_corrected or sample_id_original

            # Si sample_id corrigé est au format bac-ligne-colonne,
            # on l'utilise pour refléter les modifications dans le CSV.
            bac = bag.get('bac', '')
            ligne = bag.get('ligne', '')
            colonne = bag.get('colonne', '')
            if sample_id_corrected and '-' in sample_id_corrected:
                parts = [p.strip() for p in sample_id_corrected.split('-', 2)]
                if len(parts) == 3:
                    bac, ligne, colonne = parts

            tags = ver.get('tags', cor.get('tags', [])) or []

            image_path = result.get('image', '')
            base_info = {
                'image_id': Path(image_path).stem,
                'image_path': image_path,
                'image_width': result.get('image_size', {}).get('width', ''),
                'image_height': result.get('image_size', {}).get('height', ''),
                'pixel_per_mm': cal.get('pixel_per_mm', ''),
                'ruler_detected': cal.get('ruler_detected', False),
                'ruler_length_px': cal.get('ruler_length_px', ''),
                'spike_count': result.get('spike_count', len(result.get('spikes', []))),
                # Sachet
                'bag_detected': bag.get('detected', False),
                'sample_id': sample_id,
                'sample_id_original': sample_id_original,
                'sample_id_corrected': sample_id_corrected,
                'bac': bac,
                'ligne': ligne,
                'colonne': colonne,
                'bag_confidence': bag.get('confidence', ''),
                'bag_complete': bag.get('complete', ''),
                # Vérification
                'verification_status': ver.get('status', cor.get('status', 'pending')),
                'verification_notes': ver.get('notes', cor.get('notes', '')),
                'tags': ';'.join(tags),
            }

            # Ajouter les colonnes de tags individuelles
            for i in range(max_tags):
                base_info[f'tag_{i+1}'] = tags[i] if i < len(tags) else ''

            # Ajouter une ligne par épi
            spikes = result.get('spikes', [])
            if spikes:
                for i, spike in enumerate(spikes):
                    m = spike.get('measurements', {}) or {}
                    corr = spike.get('corrections', {}) or {}

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
                        'spikelet_density_per_cm': spike.get('spikelet_density_per_cm', ''),
                        'has_segmentation': spike.get('has_segmentation', False),
                        # Coordonnées
                        'center_x': m.get('center_x', ''),
                        'center_y': m.get('center_y', ''),
                        'confidence': m.get('confidence', spike.get('confidence', '')),
                    })

                    # Métriques segmentation épi
                    seg_metrics = spike.get('segmentation_metrics') or {}
                    row.update({
                        'real_area_px': seg_metrics.get('real_area_px', ''),
                        'real_area_mm2': seg_metrics.get('real_area_mm2', ''),
                        'real_perimeter_px': seg_metrics.get('real_perimeter_px', ''),
                        'real_perimeter_mm': seg_metrics.get('real_perimeter_mm', ''),
                        'circularity': seg_metrics.get('circularity', ''),
                        'solidity': seg_metrics.get('solidity', ''),
                        'ellipse_eccentricity': seg_metrics.get('ellipse_eccentricity', ''),
                        'seg_length_px': seg_metrics.get('seg_length_px', ''),
                        'seg_length_mm': seg_metrics.get('seg_length_mm', ''),
                        'seg_width_px': seg_metrics.get('seg_width_px', ''),
                        'seg_width_mm': seg_metrics.get('seg_width_mm', ''),
                        'seg_aspect_ratio': seg_metrics.get('seg_aspect_ratio', ''),
                    })

                    # Profil de largeur
                    wp = spike.get('width_profile') or {}
                    row.update({
                        'shape_class': wp.get('shape_class', ''),
                        'apical_width_mm': wp.get('apical_width_mm', ''),
                        'medial_width_mm': wp.get('medial_width_mm', ''),
                        'basal_width_mm': wp.get('basal_width_mm', ''),
                        'max_width_mm': wp.get('max_width_mm', ''),
                        'max_width_position': wp.get('max_width_position', ''),
                    })

                    # Couleur
                    col = spike.get('color') or {}
                    row.update({
                        'hue_mean': col.get('hue_mean', ''),
                        'saturation_mean': col.get('saturation_mean', ''),
                        'value_mean': col.get('value_mean', ''),
                        'greenness_index': col.get('greenness_index', ''),
                        'yellowing_index': col.get('yellowing_index', ''),
                    })

                    # Statistiques épillets segmentés
                    sp_stats = spike.get('spikelet_stats') or {}
                    row.update({
                        'spikelet_seg_count': sp_stats.get('n_segmented', ''),
                        'spikelet_length_mm_mean': sp_stats.get('spikelet_length_mm_mean', ''),
                        'spikelet_length_mm_std': sp_stats.get('spikelet_length_mm_std', ''),
                        'spikelet_width_mm_mean': sp_stats.get('spikelet_width_mm_mean', ''),
                        'spikelet_width_mm_std': sp_stats.get('spikelet_width_mm_std', ''),
                        'spikelet_area_mm2_mean': sp_stats.get('spikelet_area_mm2_mean', ''),
                        'spikelet_area_mm2_std': sp_stats.get('spikelet_area_mm2_std', ''),
                        'spikelet_aspect_ratio_mean': sp_stats.get('spikelet_aspect_ratio_mean', ''),
                        'spikelet_circularity_mean': sp_stats.get('spikelet_circularity_mean', ''),
                        'spikelet_length_cv': sp_stats.get('spikelet_length_cv', ''),
                        'spikelet_area_cv': sp_stats.get('spikelet_area_cv', ''),
                    })

                    # Rachis
                    rachis_data = spike.get('rachis') or {}
                    row.update({
                        'rachis_detected': rachis_data.get('detected', False),
                        'rachis_confidence': rachis_data.get('confidence', ''),
                        'rachis_length_px': rachis_data.get('length_px', ''),
                        'rachis_length_mm': rachis_data.get('length_mm', ''),
                    })

                    # Angles d'insertion
                    angle_stats = spike.get('insertion_angle_stats') or {}
                    row.update({
                        'insertion_angle_mean': angle_stats.get('mean', ''),
                        'insertion_angle_std': angle_stats.get('std', ''),
                        'insertion_angle_min': angle_stats.get('min', ''),
                        'insertion_angle_max': angle_stats.get('max', ''),
                        'spikelets_left': angle_stats.get('spikelets_left', ''),
                        'spikelets_right': angle_stats.get('spikelets_right', ''),
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
                    'spikelet_density_per_cm': '',
                    'has_segmentation': '',
                    'real_area_px': '',
                    'real_area_mm2': '',
                    'real_perimeter_px': '',
                    'real_perimeter_mm': '',
                    'circularity': '',
                    'solidity': '',
                    'ellipse_eccentricity': '',
                    'seg_length_px': '',
                    'seg_length_mm': '',
                    'seg_width_px': '',
                    'seg_width_mm': '',
                    'seg_aspect_ratio': '',
                    'shape_class': '',
                    'apical_width_mm': '',
                    'medial_width_mm': '',
                    'basal_width_mm': '',
                    'max_width_mm': '',
                    'max_width_position': '',
                    'hue_mean': '',
                    'saturation_mean': '',
                    'value_mean': '',
                    'greenness_index': '',
                    'yellowing_index': '',
                    'spikelet_seg_count': '',
                    'spikelet_length_mm_mean': '',
                    'spikelet_length_mm_std': '',
                    'spikelet_width_mm_mean': '',
                    'spikelet_width_mm_std': '',
                    'spikelet_area_mm2_mean': '',
                    'spikelet_area_mm2_std': '',
                    'spikelet_aspect_ratio_mean': '',
                    'spikelet_circularity_mean': '',
                    'spikelet_length_cv': '',
                    'spikelet_area_cv': '',
                    'rachis_detected': '',
                    'rachis_confidence': '',
                    'rachis_length_px': '',
                    'rachis_length_mm': '',
                    'insertion_angle_mean': '',
                    'insertion_angle_std': '',
                    'insertion_angle_min': '',
                    'insertion_angle_max': '',
                    'spikelets_left': '',
                    'spikelets_right': '',
                    'center_x': '',
                    'center_y': '',
                    'confidence': '',
                })
                rows.append(row)

        # Définir l'ordre des colonnes explicitement (base analyzer + vérification)
        if rows:
            base_columns = [
                'image_id', 'image_path', 'image_width', 'image_height',
                'pixel_per_mm', 'ruler_detected', 'ruler_length_px', 'spike_count',
                'bag_detected', 'sample_id', 'sample_id_original', 'sample_id_corrected',
                'bac', 'ligne', 'colonne', 'bag_confidence', 'bag_complete',
                'verification_status', 'verification_notes', 'tags'
            ]
            tag_columns = [f'tag_{i+1}' for i in range(max_tags)]
            spike_columns = [
                'spike_id', 'spike_length_px', 'spike_length_mm', 'spike_width_px', 'spike_width_mm',
                'whole_spike_length_px', 'whole_spike_length_mm', 'awns_length_px', 'awns_length_mm', 'has_awns',
                'area_px', 'area_mm2', 'perimeter_px', 'perimeter_mm', 'aspect_ratio', 'angle_degrees',
                'spikelet_count', 'spikelet_method', 'spikelet_confidence', 'spikelet_density_per_cm',
                'has_segmentation', 'real_area_px', 'real_area_mm2', 'real_perimeter_px', 'real_perimeter_mm',
                'circularity', 'solidity', 'ellipse_eccentricity',
                'seg_length_px', 'seg_length_mm', 'seg_width_px', 'seg_width_mm', 'seg_aspect_ratio',
                'shape_class', 'apical_width_mm', 'medial_width_mm', 'basal_width_mm', 'max_width_mm',
                'max_width_position',
                'hue_mean', 'saturation_mean', 'value_mean', 'greenness_index', 'yellowing_index',
                'center_x', 'center_y', 'confidence',
                'spikelet_seg_count', 'spikelet_length_mm_mean', 'spikelet_length_mm_std',
                'spikelet_width_mm_mean', 'spikelet_width_mm_std', 'spikelet_area_mm2_mean',
                'spikelet_area_mm2_std', 'spikelet_aspect_ratio_mean', 'spikelet_circularity_mean',
                'spikelet_length_cv', 'spikelet_area_cv',
                'rachis_detected', 'rachis_confidence', 'rachis_length_px', 'rachis_length_mm',
                'insertion_angle_mean', 'insertion_angle_std', 'insertion_angle_min', 'insertion_angle_max',
                'spikelets_left', 'spikelets_right'
            ]

            # Fusionner avec tout champ additionnel éventuel pour compatibilité future
            all_fieldnames = set()
            for row in rows:
                all_fieldnames.update(row.keys())

            fieldnames = [f for f in (base_columns + tag_columns + spike_columns) if f in all_fieldnames]
            fieldnames += [f for f in sorted(all_fieldnames) if f not in fieldnames]

            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            logger.info(
                f"CSV régénéré: {csv_path} ({len(rows)} lignes, {len(fieldnames)} colonnes, {max_tags} tags max)"
            )
            return jsonify({'success': True, 'count': len(rows), 'path': str(csv_path), 'columns': len(fieldnames)})
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
    print(f"  1-9    Toggle tag rapide")
    print(f"\nTags disponibles:")
    for tag in PREDEFINED_TAGS:
        print(f"  {tag['shortcut']}  {tag['label']}")
    print(f"{'='*60}\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
