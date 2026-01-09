#!/usr/bin/env python3
"""
Application de vérification des résultats d'analyse

Interface web Flask pour visualiser et corriger les résultats
de l'analyse des épis de blé.

Raccourcis clavier:
    ← / → : Image précédente / suivante
    V     : Valider l'image courante
    R     : Rejeter l'image courante
    E     : Éditer les valeurs
    S     : Sauvegarder les corrections
    F     : Filtrer (non validés seulement)
    1-9   : Aller à l'épi N pour correction

Usage:
    python app/verification_app.py --output output/
    python app/verification_app.py --output output/ --port 5001
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, jsonify, render_template_string, request, send_file

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Variables globales
RESULTS_DIR = None
RESULTS_CACHE = {}
CORRECTIONS_FILE = None


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
            
            # Charger les corrections si existantes
            corrections_file = session_dir / 'corrections.json'
            if corrections_file.exists():
                with open(corrections_file, 'r') as f:
                    data['_corrections'] = json.load(f)
            else:
                data['_corrections'] = {}
            
            results.append(data)
        except Exception as e:
            logger.warning(f"Erreur chargement {results_file}: {e}")
    
    # Mettre en cache
    RESULTS_CACHE = {Path(r['image']).stem: r for r in results}
    
    return results


def save_correction(image_id: str, corrections: Dict) -> bool:
    """Sauvegarde les corrections pour une image"""
    if image_id not in RESULTS_CACHE:
        return False
    
    result = RESULTS_CACHE[image_id]
    session_dir = Path(result['_session_dir'])
    
    # Fusionner avec les corrections existantes
    existing = result.get('_corrections', {})
    existing.update(corrections)
    existing['_last_modified'] = datetime.now().isoformat()
    
    # Sauvegarder
    corrections_file = session_dir / 'corrections.json'
    with open(corrections_file, 'w') as f:
        json.dump(existing, f, indent=2)
    
    # Mettre à jour le cache
    RESULTS_CACHE[image_id]['_corrections'] = existing
    
    return True


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
            gap: 20px;
            align-items: center;
        }
        
        .counter {
            background: #0f3460;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
        }
        
        .filter-btn {
            background: #0f3460;
            border: none;
            color: #eee;
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
        }
        
        .filter-btn.active {
            background: #e94560;
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
            padding: 20px;
            overflow-y: auto;
            min-width: 350px;
            max-width: 400px;
        }
        
        .section {
            background: #0f3460;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }
        
        .section h3 {
            color: #e94560;
            margin-bottom: 10px;
            font-size: 0.9rem;
            text-transform: uppercase;
        }
        
        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #1a1a3e;
        }
        
        .info-row:last-child {
            border-bottom: none;
        }
        
        .info-label {
            color: #888;
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
        
        /* Spike list */
        .spike-item {
            background: #1a1a3e;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 8px;
        }
        
        .spike-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        
        .spike-id {
            color: #e94560;
            font-weight: bold;
        }
        
        /* Actions */
        .actions {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        
        .btn {
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 5px;
            font-size: 1rem;
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
            font-size: 0.75rem;
            color: #888;
        }
        
        .shortcuts kbd {
            background: #333;
            padding: 2px 6px;
            border-radius: 3px;
            margin-right: 5px;
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
            <button class="filter-btn" id="filterBtn" onclick="toggleFilter()">
                Afficher tous
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
                    <input type="text" class="editable" id="sampleId" placeholder="?-?-?">
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
                <textarea id="notes" style="width:100%; height:60px; background:#1a1a3e; border:1px solid #333; color:#eee; border-radius:5px; padding:8px; resize:vertical;"></textarea>
            </div>
            
            <div class="actions">
                <button class="btn btn-validate" onclick="setStatus('validated')">
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
        <kbd>←</kbd><kbd>→</kbd> Navigation
        <kbd>V</kbd> Valider
        <kbd>R</kbd> Rejeter
        <kbd>S</kbd> Sauvegarder
        <kbd>F</kbd> Filtrer
    </div>
    
    <div class="loading" id="loading">Chargement...</div>
    <div class="toast" id="toast"></div>

    <script>
        let results = [];
        let filteredResults = [];
        let currentIndex = 0;
        let filterPending = false;
        
        // Charger les résultats au démarrage
        async function loadResults() {
            showLoading(true);
            try {
                const response = await fetch('/api/results');
                results = await response.json();
                applyFilter();
                if (filteredResults.length > 0) {
                    displayResult(0);
                }
                updateCounter();
            } catch (e) {
                showToast('Erreur de chargement', 'error');
            }
            showLoading(false);
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
                filterPending ? 'Non validés uniquement' : 'Afficher tous';
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
            
            // Image
            document.getElementById('mainImage').src = `/api/image/${encodeURIComponent(result._session_dir)}`;
            document.getElementById('imageName').textContent = result.image.split('/').pop();
            
            // Status
            updateStatusBadge(corrections.status || 'pending');
            
            // Calibration
            const cal = result.calibration || {};
            document.getElementById('rulerDetected').textContent = cal.ruler_detected ? '✓ Oui' : '✗ Non';
            document.getElementById('rulerDetected').className = 'info-value ' + (cal.ruler_detected ? 'success' : 'error');
            document.getElementById('pixelPerMm').textContent = cal.pixel_per_mm ? 
                `${cal.pixel_per_mm.toFixed(3)} px/mm` : '-';
            
            // Bag
            const bag = result.bag || {};
            document.getElementById('sampleId').value = corrections.sample_id || bag.sample_id || '';
            document.getElementById('bagConfidence').textContent = bag.confidence ? 
                `${(bag.confidence * 100).toFixed(0)}%` : '-';
            
            // Spikes
            const spikes = result.spikes || [];
            document.getElementById('spikeCount').textContent = spikes.length;
            
            const spikeList = document.getElementById('spikeList');
            spikeList.innerHTML = spikes.map((spike, i) => {
                const m = spike.measurements || {};
                const correctedLength = corrections[`spike_${i}_length`];
                const correctedSpikelets = corrections[`spike_${i}_spikelets`];
                
                return `
                    <div class="spike-item">
                        <div class="spike-header">
                            <span class="spike-id">Épi #${spike.id || i+1}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Longueur</span>
                            <input type="number" step="0.1" class="editable" 
                                   id="spike_${i}_length"
                                   value="${correctedLength || m.length_mm || m.length_pixels || ''}"
                                   placeholder="${m.length_mm ? 'mm' : 'px'}">
                        </div>
                        <div class="info-row">
                            <span class="info-label">Épillets</span>
                            <input type="number" class="editable"
                                   id="spike_${i}_spikelets"
                                   value="${correctedSpikelets || spike.spikelet_count || ''}"
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
        
        function setStatus(status) {
            const result = filteredResults[currentIndex];
            if (!result) return;
            
            result._corrections = result._corrections || {};
            result._corrections.status = status;
            updateStatusBadge(status);
            
            // Auto-save et passer au suivant
            saveCorrections().then(() => {
                if (status === 'validated' && currentIndex < filteredResults.length - 1) {
                    navigate(1);
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
            
            switch(e.key) {
                case 'ArrowLeft':
                    navigate(-1);
                    break;
                case 'ArrowRight':
                    navigate(1);
                    break;
                case 'v':
                case 'V':
                    setStatus('validated');
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
    return render_template_string(HTML_TEMPLATE)


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
    """Exporte toutes les corrections en CSV"""
    import csv
    from io import StringIO
    
    results = load_all_results()
    
    output = StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        'image_id', 'status', 'sample_id_original', 'sample_id_corrected',
        'spike_count', 'notes', 'last_modified'
    ])
    
    for result in results:
        corrections = result.get('_corrections', {})
        bag = result.get('bag', {})
        
        writer.writerow([
            Path(result['image']).stem,
            corrections.get('status', 'pending'),
            bag.get('sample_id', ''),
            corrections.get('sample_id', ''),
            len(result.get('spikes', [])),
            corrections.get('notes', ''),
            corrections.get('_last_modified', '')
        ])
    
    output.seek(0)
    
    from flask import Response
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=corrections_export.csv'}
    )


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
    print(f"  ← →  Navigation entre images")
    print(f"  V    Valider l'image")
    print(f"  R    Rejeter l'image")
    print(f"  S    Sauvegarder les corrections")
    print(f"  F    Filtrer les non-validés")
    print(f"{'='*60}\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
