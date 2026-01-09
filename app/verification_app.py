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
    
    # Chercher différents noms possibles
    patterns = {
        'final': ['05_final*.jpg', '05_final*.png', 'final*.jpg', 'debug_final*.jpg'],
        'detections': ['01_detections*.jpg', '01_detections*.png'],
        'spikes': ['02_spikes*.jpg', '03_spikelets*.jpg'],
        'bag': ['04_bag*.jpg'],
    }
    
    for pattern in patterns.get(image_type, patterns['final']):
        matches = list(session_path.glob(pattern))
        if matches:
            return str(matches[0])
    
    # Fallback: première image jpg trouvée
    jpgs = list(session_path.glob('*.jpg'))
    if jpgs:
        return str(jpgs[0])
    
    return None


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
        }
        
        .image-container img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
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
    </style>
</head>
<body>
    <div class="header">
        <h1>🌾 Wheat Spike Analyzer - Vérification</h1>
        <div class="nav-info">
            <button class="header-btn" id="filterBtn" onclick="toggleFilter()">
                Afficher tous
            </button>
            <button class="header-btn export" onclick="regenerateCSV()">
                📊 Régénérer CSV
            </button>
            <div class="counter">
                <span id="currentIndex">0</span> / <span id="totalCount">0</span>
                (<span id="validatedCount">0</span> ✓)
            </div>
        </div>
    </div>
    
    <div class="main">
        <div class="image-panel">
            <button class="image-nav prev" onclick="navigate(-1)">‹</button>
            <div class="image-container">
                <img id="mainImage" src="" alt="Image en cours">
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
                    <input type="text" class="editable" id="sampleId" placeholder="?-?-?" style="width:100px;">
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
        <kbd>F</kbd> Filtre
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
        let filterPending = false;
        let currentTags = [];
        
        // Charger les résultats au démarrage
        async function loadResults() {
            showLoading(true);
            try {
                const response = await fetch('/api/results');
                results = await response.json();
                applyFilter();
                renderTags();
                if (filteredResults.length > 0) {
                    displayResult(0);
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
            if (filterPending) {
                filteredResults = results.filter(r => 
                    !r._corrections || r._corrections.status !== 'validated'
                );
            } else {
                filteredResults = [...results];
            }
            currentIndex = Math.min(currentIndex, Math.max(0, filteredResults.length - 1));
        }
        
        function toggleFilter() {
            filterPending = !filterPending;
            document.getElementById('filterBtn').textContent = 
                filterPending ? 'Non validés' : 'Afficher tous';
            document.getElementById('filterBtn').classList.toggle('active', filterPending);
            applyFilter();
            displayResult(currentIndex);
            updateCounter();
        }
        
        function navigate(delta) {
            const newIndex = currentIndex + delta;
            if (newIndex >= 0 && newIndex < filteredResults.length) {
                currentIndex = newIndex;
                displayResult(currentIndex);
                updateCounter();
            }
        }
        
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
            
            // Bag
            const bag = result.bag || {};
            document.getElementById('sampleId').value = bag.sample_id_corrected || bag.sample_id || '';
            document.getElementById('bagConfidence').textContent = bag.confidence ? 
                `${(bag.confidence * 100).toFixed(0)}%` : '-';
            
            // Spikes
            const spikes = result.spikes || [];
            document.getElementById('spikeCount').textContent = spikes.length;
            
            const spikeList = document.getElementById('spikeList');
            spikeList.innerHTML = spikes.map((spike, i) => {
                const m = spike.measurements || {};
                const corr = spike.corrections || {};
                
                return `
                    <div class="spike-item">
                        <div class="spike-header">
                            <span class="spike-id">Épi #${spike.id || i+1}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Longueur</span>
                            <input type="number" step="0.1" class="editable" 
                                   id="spike_${i}_length"
                                   value="${corr.length || m.length_mm || m.length_pixels || ''}"
                                   placeholder="${m.length_mm ? 'mm' : 'px'}">
                        </div>
                        <div class="info-row">
                            <span class="info-label">Épillets</span>
                            <input type="number" class="editable"
                                   id="spike_${i}_spikelets"
                                   value="${corr.spikelets || spike.spikelet_count || ''}"
                                   placeholder="?">
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
            
            // Auto-save et passer au suivant
            saveCorrections().then(() => {
                if (status === 'validated' && currentIndex < filteredResults.length - 1) {
                    setTimeout(() => navigate(1), 300);
                }
            });
        }
        
        async function saveCorrections() {
            const result = filteredResults[currentIndex];
            if (!result) return;
            
            const imageId = result.image.split('/').pop().replace(/\.[^.]+$/, '');
            
            // Collecter les corrections
            const corrections = {
                status: result._corrections?.status || 'pending',
                tags: currentTags,
                sample_id: document.getElementById('sampleId').value,
                notes: document.getElementById('notes').value,
            };
            
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
                    navigate(-1);
                    break;
                case 'ArrowRight':
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
                    toggleFilter();
                    break;
                case 't':
                case 'T':
                    document.getElementById('customTag').focus();
                    break;
            }
        });
        
        // Démarrage
        loadResults();
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


@app.route('/api/image/<path:session_dir>')
def api_image(session_dir):
    """Retourne l'image de debug pour une session"""
    image_path = get_debug_image_path(session_dir, 'final')
    
    if image_path and os.path.exists(image_path):
        return send_file(image_path, mimetype='image/jpeg')
    
    # Image placeholder si non trouvée
    return '', 404


@app.route('/api/image')
def api_image_query():
    """Retourne l'image de debug pour une session (paramètre query `session_dir`).

    Utiliser une query string évite les problèmes d'encoding des slashes dans l'URL.
    """
    session_dir = request.args.get('session_dir')
    if not session_dir:
        return '', 404

    # session_dir est normalement décodé par Flask; garantir string
    image_path = get_debug_image_path(session_dir, 'final')
    if image_path and os.path.exists(image_path):
        return send_file(image_path, mimetype='image/jpeg')

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
        
        rows = []
        for result in results:
            verification = result.get('_verification', {})
            bag = result.get('bag', {})
            cal = result.get('calibration', {})
            
            # Utiliser les valeurs corrigées si disponibles
            sample_id = bag.get('sample_id_corrected') or bag.get('sample_id', '')
            
            # Base row pour l'image
            base_info = {
                'image_id': Path(result.get('image', '')).stem,
                'image_path': result.get('image', ''),
                'verification_status': verification.get('status', 'pending'),
                'verification_tags': ';'.join(verification.get('tags', [])),
                'verification_notes': verification.get('notes', ''),
                'sample_id': sample_id,
                'sample_bac': bag.get('bac', ''),
                'sample_ligne': bag.get('ligne', ''),
                'sample_colonne': bag.get('colonne', ''),
                'ruler_detected': cal.get('ruler_detected', False),
                'pixel_per_mm': cal.get('pixel_per_mm', ''),
                'spike_count': len(result.get('spikes', [])),
            }
            
            # Ajouter une ligne par épi
            spikes = result.get('spikes', [])
            if spikes:
                for i, spike in enumerate(spikes):
                    m = spike.get('measurements', {})
                    corr = spike.get('corrections', {})
                    
                    row = base_info.copy()
                    row.update({
                        'spike_id': spike.get('id', i+1),
                        'spike_length_mm': corr.get('length') or m.get('length_mm', ''),
                        'spike_length_px': m.get('length_pixels', ''),
                        'spike_width_mm': m.get('width_mm', ''),
                        'spikelet_count': corr.get('spikelets') or spike.get('spikelet_count', ''),
                        'spikelet_method': spike.get('spikelet_method', ''),
                        'spikelet_confidence': spike.get('spikelet_confidence', ''),
                    })
                    rows.append(row)
            else:
                # Image sans épi détecté
                row = base_info.copy()
                row.update({
                    'spike_id': '',
                    'spike_length_mm': '',
                    'spike_length_px': '',
                    'spike_width_mm': '',
                    'spikelet_count': '',
                    'spikelet_method': '',
                    'spikelet_confidence': '',
                })
                rows.append(row)
        
        # Écrire le CSV
        if rows:
            fieldnames = rows[0].keys()
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            
            logger.info(f"CSV régénéré: {csv_path} ({len(rows)} lignes)")
            return jsonify({'success': True, 'count': len(rows), 'path': str(csv_path)})
        else:
            return jsonify({'success': False, 'error': 'Aucun résultat à exporter'})
            
    except Exception as e:
        logger.error(f"Erreur régénération CSV: {e}")
        return jsonify({'success': False, 'error': str(e)})


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
    
    RESULTS_DIR = args.output
    
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
