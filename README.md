# Wheat Spike Analyzer

**Outil d'analyse phénotypique automatisé pour épis de blé** utilisant YOLO OBB (Oriented Bounding Boxes) et YOLO-Seg.

> Développé par Nicolas Salas dans le cadre d'une thèse INRAE/SupAgro sur l'analyse phénotypique du blé.

---

## Vue d'ensemble

Pipeline complet en **10 étapes** pour l'analyse automatisée d'images d'épis de blé :

1. **Détection OBB** — Détection orientée (YOLO-OBB) : règle, épi sans barbes (`spike`), épi avec barbes (`whole_spike`), sachet
2. **Calibration** — Calcul px/mm via graduations détectées sur la règle (ou longueur OBB de la règle en fallback)
3. **Appariement spike ↔ whole_spike** — Algorithme hongrois avec IoU OBB pour un matching optimal
4. **Mesures morphométriques** — Longueur, largeur, aire, périmètre, aspect ratio, longueur des barbes (en mm)
5. **Segmentation des épis** — YOLO-Seg pixel-level avec TTA (vote majoritaire sur masques)
6. **Comptage des épillets** — Modèle YOLO dédié avec déduplication et TTA
7. **Segmentation des épillets** — YOLO-Seg instance segmentation (taille, forme, aire par épillet)
8. **Rachis et angles d'insertion** — YOLO-Seg rachis + calcul des angles d'insertion des épillets
9. **OCR sachets** — Identification automatique bac-ligne-colonne (chiffres 1-20) avec orientation du sachet
10. **Export** — JSON structuré + CSV résumé (~80+ colonnes) + images de debug

---

## Fonctionnalités

### Détection et mesures
- **4 classes OBB** : ruler (0), spike (1), bag (2), whole_spike (3)
- **Calibration automatique** via règle 30 cm (graduations ou longueur OBB)
- **Morphométrie complète** : longueur, largeur, aire, périmètre, aspect ratio, barbes
- **Segmentation pixel-level** des épis (YOLO-Seg remplaçant SAM2)
- **Profil de largeur** (apical/médial/basal) → classification de forme (fusiforme/claviforme/parallèle/obovale)
- **Stats couleur** : HSV, RGB, indices de verdeur/jaunissement

### Comptage et analyse fine
- **Comptage d'épillets** avec niveaux de confiance (haute ≥10, moyenne ≥5, basse <5)
- **Segmentation individuelle** des épillets (aire, forme)
- **Rachis** : segmentation de l'axe central + angles d'insertion des épillets

### OCR et identification
- **Chiffres 1-20** sur sachets d'échantillons → format bac-ligne-colonne
- **Orientation automatique** du sachet (détection de l'ouverture)

### Test-Time Augmentation (TTA)
TTA configurable par étape avec vote majoritaire / consensus :
- Segmentation épis, comptage épillets, rachis, OCR sachets
- Augmentations : flip_h, flip_v, rot180, brightness_up, contrast_up

### Export et vérification
- **JSON** : résultats structurés complets par image
- **CSV** : résumé tabulaire (~80+ colonnes) pour analyse statistique
- **Images de debug** : 8 étapes visuelles (OBB, calibration, graduations, analyse, bag, segmentation, épillets, rachis, angles)
- **Application web Flask** : vérification/correction interactive des résultats batch

---

## Installation

### Prérequis
- Python 3.12+
- GPU AMD (ROCm 7.1.1) ou NVIDIA (CUDA) recommandé

### Installation

```bash
# Cloner le projet (inclut les pointeurs Git LFS)
git clone https://github.com/NicoSalas34/wheat-spike-analyzer.git
cd wheat-spike-analyzer

# Récupérer les modèles YOLO (~290 MB via Git LFS)
git lfs install
git lfs pull

# Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer PyTorch selon votre GPU :

# AMD (ROCm) :
./install_amd.sh
# Ou : pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

# NVIDIA (CUDA) :
pip install torch torchvision torchaudio

# CPU uniquement :
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Installer les dépendances
pip install -r requirements.txt
```

### Vérification rapide

```bash
# Tester sur les 3 images incluses dans le repo
python src/main.py data/test_sample/ --batch --low-debug
```

---

## Utilisation

### Ligne de commande

```bash
# Analyser une image
python src/main.py data/raw/image.JPG

# Mode verbose
python src/main.py data/raw/image.JPG --verbose

# Mode batch (dossier d'images)
python src/main.py data/raw/ --batch

# Debug léger (uniquement result_annotated.png)
python src/main.py data/raw/image.JPG --low-debug

# Sans images de debug
python src/main.py data/raw/image.JPG --no-debug

# Reprendre un batch interrompu
python src/main.py data/raw/ --batch --resume

# Dossier de sortie personnalisé
python src/main.py data/raw/image.JPG --output results/
```

### Arguments CLI

| Argument | Description | Défaut |
|----------|-------------|--------|
| `input` | Chemin image ou dossier | *requis* |
| `--config` | Fichier de configuration | `config/config.yaml` |
| `--output` | Dossier de sortie | `output/` |
| `--verbose`, `-v` | Mode verbose | `False` |
| `--no-debug` | Désactiver toutes les images de debug | `False` |
| `--low-debug` | Debug léger (uniquement `result_annotated.png`) | `False` |
| `--batch` | Traiter un dossier d'images | `False` |
| `--resume` | Reprendre un batch interrompu | `False` |

### Utilisation programmatique

```python
from src.analyzer_obb import WheatSpikeAnalyzerOBB, create_analyzer_from_config

# Depuis la config YAML
analyzer = create_analyzer_from_config(
    config_path="config/config.yaml",
    output_dir="output",
    debug=True  # True=full, 'low'=léger, False=aucun
)

# Analyser une image
result = analyzer.analyze_image("data/raw/GOPR2587.JPG")

# Accéder aux résultats
print(f"Épis détectés: {result['spike_count']}")
print(f"Calibration: {result['calibration']['pixel_per_mm']:.3f} px/mm")
for spike in result['spikes']:
    m = spike['measurements']
    print(f"  Épi #{spike['id']}: L={m.get('length_mm', '?')}mm, épillets={spike.get('spikelet_count', '?')}")

# Mode batch
results = analyzer.analyze_batch(["img1.JPG", "img2.JPG", ...])
```

---

## Application de vérification

Application web Flask pour valider et corriger les résultats après un batch.

```bash
# Lancer l'application
python app/verification_app.py --output output/

# Options
python app/verification_app.py --output output/ --port 8080 --host 0.0.0.0
```

Interface accessible à **http://127.0.0.1:5000**

### Fonctionnalités
- Navigation image par image avec panneau d'information
- Édition inline : sample ID (bac-ligne-colonne), longueur épi, comptage épillets
- Suppression d'épis avec historique d'annulation
- Système de tags (8 prédéfinis + custom)
- Filtres avancés (statut, problèmes de détection, qualité, tags)
- Régénération CSV après corrections
- Zoom/pan sur les images
- Thème sombre

### Raccourcis clavier

| Touche | Action |
|--------|--------|
| `←` `→` | Navigation entre images |
| `V` | Valider l'image |
| `R` | Rejeter l'image |
| `S` | Sauvegarder les corrections |
| `F` | Cycle entre les filtres |
| `1-8` | Toggle les tags rapides |
| `T` | Focus sur le champ tag personnalisé |

### Tags prédéfinis

| # | Tag | Description |
|---|-----|-------------|
| 1 | Validé | Image correctement analysée |
| 2 | Règle non détectée | La règle n'a pas été trouvée |
| 3 | Épi mal détecté | Problème de détection d'épi |
| 4 | Sachet illisible | OCR du sachet incorrect |
| 5 | Épillets incorrects | Comptage d'épillets erroné |
| 6 | Image floue | Qualité d'image insuffisante |
| 7 | Plusieurs épis confondus | Épis superposés ou mal séparés |
| 8 | Calibration incorrecte | Problème de calibration |

---

## Structure du projet

```
wheat-spike-analyzer/
├── README.md                        # Ce fichier
├── setup.py                         # Package Python (entry point: wheat-analyzer)
├── requirements.txt                 # Dépendances Python
├── install_amd.sh                   # Installation PyTorch ROCm (AMD)
├── reset_gpu.sh                     # Script reset GPU AMD
├── .gitattributes                   # Git LFS tracking (models/*.pt)
│
├── config/
│   └── config.yaml                  # Configuration complète du pipeline (7 modèles, TTA, seuils)
│
├── src/                             # Code source principal
│   ├── __init__.py                  # Exports: WheatSpikeAnalyzerOBB
│   ├── main.py                      # Point d'entrée CLI
│   ├── analyzer_obb.py              # Pipeline principal (10 étapes, gestion mémoire)
│   ├── spike_matcher.py             # Appariement hongrois spike↔whole_spike (IoU OBB)
│   ├── spike_segmenter.py           # Segmentation épis + morphométrie avancée
│   ├── spikelet_counter.py          # Comptage épillets YOLO + déduplication
│   ├── bag_digit_detector.py        # OCR sachets (chiffres 1-20, orientation)
│   ├── tta.py                       # Test-Time Augmentation (géométrique + photométrique)
│   └── utils.py                     # Logging, config, I/O
│
├── app/
│   └── verification_app.py          # Application web Flask de vérification
│
├── models/                          # Modèles YOLO pré-entraînés (Git LFS, ~290 MB)
│   ├── wheat_spike_yolo.pt          # OBB : ruler, spike, bag, whole_spike
│   ├── graduations_yolo.pt          # OBB : graduations 0/10/20/30 cm
│   ├── spikelets_yolo.pt            # Détection/comptage épillets (YOLO-Seg)
│   ├── spike_seg_yolo.pt            # Segmentation épis (YOLO-Seg)
│   ├── rachis_yolo.pt               # Segmentation rachis (YOLO-Seg)
│   ├── bag_digits_yolo.pt           # OCR chiffres 1-20
│   └── bag_opening_yolo.pt          # Orientation sachet
│
├── scripts/                         # Scripts utilitaires (entraînement, conversion)
│   ├── train_spike.py               # Entraînement OBB (yolo26s-obb)
│   ├── train_spike_seg.py           # Entraînement segmentation épis
│   ├── train_spikelet_seg.py        # Entraînement segmentation épillets
│   ├── train_rachis.py              # Entraînement rachis
│   └── ...                          # Conversion annotations, pré-annotation, etc.
│
├── data/
│   ├── test_sample/                 # 3 images de test incluses dans le repo
│   ├── raw/                         # Images brutes (non versionné, .gitignore)
│   └── validation/                  # Images de validation (non versionné)
│
└── output/                          # Résultats d'analyse (non versionné)
    ├── results_summary.csv
    └── <image_name>/
        ├── results.json
        └── *.png                    # Images de debug
```

> **Note :** Les datasets d'entraînement (`training_*/`), les résultats (`output/`),
> les runs YOLO (`runs/`) et les images brutes (`data/raw/`, `data/validation/`)
> sont exclus du dépôt via `.gitignore`. Seuls les 3 images de test dans
> `data/test_sample/` sont versionnées.

---

## Modèles YOLO

7 modèles spécialisés, stockés via Git LFS dans `models/` :

| Modèle | Type | Classes | Architecture de base | Usage |
|--------|------|---------|---------------------|-------|
| `wheat_spike_yolo.pt` | OBB | ruler, spike, bag, whole_spike | yolo26s-obb | Détection principale |
| `graduations_yolo.pt` | OBB | 0cm, 10cm, 20cm, 30cm | — | Calibration par graduations |
| `spikelets_yolo.pt` | Detect | spikelet | — | Comptage des épillets |
| `spike_seg_yolo.pt` | Segment | spike | yolo26s-seg | Segmentation pixel-level des épis |
| `rachis_yolo.pt` | Segment | rachis | yolo26l-seg | Segmentation de l'axe central |
| `bag_digits_yolo.pt` | Detect | 1-20 | — | OCR chiffres sur sachets |
| `bag_opening_yolo.pt` | Detect | bag_opening | — | Orientation du sachet |

---

## Images de debug

L'analyseur génère des images de debug dans `output/<image>/debug/` :

| Fichier | Étape | Description |
|---------|-------|-------------|
| `01_detections_obb.png` | 1 | Toutes les détections OBB (4 classes) |
| `02_ruler_calibration.png` | 2 | Calibration avec la règle |
| `02b_graduations.png` | 2 | Graduations détectées sur la règle |
| `03_spikes_analysis.png` | 3-4 | Épis avec mesures et appariements |
| `04_bag_identification.png` | 9 | Sachet avec OCR |
| `05_final_result.png` | 10 | Résultat final annoté |
| `05b_segmentation.png` | 5 | Segmentation YOLO-Seg des épis |
| `06_spikelets.png` | 6-7 | Segmentation individuelle des épillets |
| `07_rachis.png` | 8 | Détection du rachis |
| `08_insertion_angles.png` | 8 | Angles d'insertion des épillets |

---

## Configuration

Le fichier `config/config.yaml` contrôle l'intégralité du pipeline :

```yaml
# Détection OBB principale
yolo:
  model_path: 'models/wheat_spike_yolo.pt'
  confidence_threshold: 0.35
  iou_threshold: 0.45
  tta:
    enabled: false

# Calibration (graduations sur la règle)
graduation_detection:
  enabled: true
  model_path: 'models/graduations_yolo.pt'

# Comptage épillets (avec TTA)
spikelet_counting:
  enabled: true
  yolo:
    model_path: 'models/spikelets_yolo.pt'
  tta:
    enabled: true

# Segmentation des épis (avec TTA)
segmentation:
  enabled: true
  spike_seg_model: 'models/spike_seg_yolo.pt'
  tta:
    enabled: true

# Rachis (avec TTA)
rachis_detection:
  enabled: true
  model_path: 'models/rachis_yolo.pt'
  tta:
    enabled: true
    consensus_threshold: 0.4

# OCR sachets (avec TTA)
bag_digits:
  enabled: true
  model_path: 'models/bag_digits_yolo.pt'
  opening_model_path: 'models/bag_opening_yolo.pt'
  tta:
    enabled: true
```

Voir `config/config.yaml` pour la configuration complète avec tous les seuils et paramètres.

### Gestion mémoire (batch)

```yaml
batch:
  memory_cleanup_interval: 5  # Nettoyage RAM+VRAM toutes les N images (défaut: 5)
```

Pour les longues séries (>100 images), le pipeline libère automatiquement la RAM et la VRAM GPU à intervalles réguliers. Sur GPU AMD (ROCm), les erreurs HIP (`hipErrorLaunchFailure`) sont gérées de manière résiliente : le batch continue en sautant les images problématiques.

Options utiles :
- `--low-debug` : réduit l'empreinte mémoire (1 seule image de debug au lieu de 10)
- `--no-debug` : aucune image de debug (minimal memory)

---

## Entraînement des modèles

Les datasets d'entraînement ne sont pas inclus dans le dépôt (trop volumineux). Pour ré-entraîner les modèles :

1. Préparer un dossier `training_<task>/` avec `data.yaml`, `images/` et `labels/`
2. Annoter dans X-AnyLabeling
3. Convertir les annotations (`scripts/add_annotations_to_dataset.py`)
4. Entraîner avec Ultralytics (`scripts/train_*.py`)

Scripts d'entraînement dans `scripts/` :

```bash
python scripts/train_spike.py          # OBB (yolo26s-obb, 500 epochs, batch 16, imgsz 1024)
python scripts/train_spike_seg.py      # Segmentation épis (yolo26s-seg, 500 epochs, batch 12)
python scripts/train_spikelet_seg.py   # Segmentation épillets (yolo26x-seg, 500 epochs, batch 8)
python scripts/train_rachis.py         # Rachis (yolo26l-seg, 500 epochs, batch 8, 180° rotation)
```

---

## Licence

Projet de recherche — Thèse INRAE/SupAgro

## Auteur

Nicolas Salas — Thèse sur l'analyse phénotypique du blé
