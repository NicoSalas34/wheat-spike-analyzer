# Wheat Spike Analyzer

🌾 **Outil d'analyse phénotypique automatisé pour épis de blé** utilisant YOLO OBB (Oriented Bounding Boxes).

> Développé dans le cadre d'une thèse sur l'analyse phénotypique du blé.

## 📋 Fonctionnalités

### 🔍 Détection YOLO OBB
- **Règle** : calibration automatique (30cm)
- **Épis** : détection avec et sans barbes (`spike` / `whole_spike`)
- **Sachets** : identification échantillon via OCR (bac-ligne-colonne)

### 📏 Mesures morphométriques
- **Longueur & largeur** de l'épi (mm)
- **Longueur des barbes** (différence whole_spike - spike)
- **Aire & périmètre** en mm² et mm

### 🌿 Comptage des épillets
- Modèle YOLO dédié
- Confiance haute/moyenne/basse selon le nombre de détections

### 🔢 OCR sachets
- Identification automatique : **bac-ligne-colonne** (ex: 12-11-7)
- Détection de l'orientation du sachet

### 📊 Export
- **JSON** : données structurées complètes
- **CSV** : résumé tabulaire
- **Images de debug** : visualisation des étapes d'analyse

---

## 🚀 Installation

### Prérequis
- Python 3.12+
- GPU AMD (ROCm 7.1.1) ou NVIDIA (CUDA) recommandé

### Installation

```bash
# Cloner le projet
git clone https://github.com/<user>/wheat-spike-analyzer.git
cd wheat-spike-analyzer

# Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer PyTorch selon votre GPU:

# AMD (ROCm 7.1.1):
./install_amd.sh

# NVIDIA (CUDA):
pip install torch torchvision torchaudio

# CPU uniquement:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les modèles (non inclus dans git)
# Voir section "Modèles" ci-dessous
```

---

## 💻 Utilisation

### Ligne de commande

```bash
# Analyser une image
python src/main.py data/raw/image.JPG

# Mode verbose (détails)
python src/main.py data/raw/image.JPG --verbose

# Analyser un dossier (mode batch)
python src/main.py data/raw/ --batch

# Désactiver les images de debug
python src/main.py data/raw/image.JPG --no-debug

# Reprendre un batch interrompu
python src/main.py data/raw/ --batch --resume

# Spécifier le dossier de sortie
python src/main.py data/raw/image.JPG --output results/
```

### Arguments CLI

| Argument | Description | Défaut |
|----------|-------------|--------|
| `input` | Chemin image ou dossier | *requis* |
| `--config` | Fichier de configuration | `config/config.yaml` |
| `--output` | Dossier de sortie | `output/` |
| `--verbose`, `-v` | Mode verbose | `False` |
| `--no-debug` | Désactiver images de debug | `False` |
| `--batch` | Traiter plusieurs images | `False` |
| `--resume` | Reprendre batch | `False` |

### Utilisation programmatique

```python
from src.analyzer_obb import WheatSpikeAnalyzerOBB
from src.utils import load_config

# Charger la configuration
config = load_config("config/config.yaml")

# Créer l'analyseur
analyzer = WheatSpikeAnalyzerOBB(config, output_dir="output", debug=True)

# Analyser une image
results = analyzer.analyze("data/raw/GOPR2587.JPG")

# Accéder aux résultats
print(f"Épis détectés: {len(results['spikes'])}")
print(f"Calibration: {results['calibration']['pixel_per_mm']:.2f} px/mm")
```

---

## 📁 Structure du projet

```
wheat-spike-analyzer/
├── config/
│   └── config.yaml              # Configuration principale
├── data/
│   └── raw/                     # Images à analyser
├── models/                      # Modèles YOLO (non versionnés)
│   ├── wheat_spike_yolo.pt      # Détection OBB (ruler, spike, bag, whole_spike)
│   ├── spikelets_yolo.pt        # Comptage épillets
│   ├── bag_digits_yolo.pt       # OCR chiffres sachets (1-20)
│   └── bag_opening_yolo.pt      # Orientation sachet
├── output/                      # Résultats d'analyse
├── scripts/
│   ├── preannotate_with_obb.py  # Pré-annotation pour labelling
│   └── add_annotations_to_dataset.py
├── src/
│   ├── main.py                  # Point d'entrée CLI
│   ├── analyzer_obb.py          # Analyseur principal
│   ├── spikelet_counter.py      # Compteur épillets
│   ├── bag_digit_detector.py    # Détecteur OCR sachets
│   └── utils.py                 # Utilitaires
├── training_*/                  # Datasets d'entraînement
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🧠 Modèles

Les modèles YOLO ne sont pas inclus dans le repository (trop volumineux).

| Modèle | Classes | Description |
|--------|---------|-------------|
| `wheat_spike_yolo.pt` | ruler, spike, bag, whole_spike | Détection OBB principale |
| `spikelets_yolo.pt` | spikelet | Comptage des épillets |
| `bag_digits_yolo.pt` | 1-20 | OCR chiffres sachets |
| `bag_opening_yolo.pt` | left, right | Orientation sachet |

### Entraînement

Les datasets d'entraînement sont dans les dossiers `training_*/`. Pour réentraîner :

```python
from ultralytics import YOLO

model = YOLO('yolo11s-obb.pt')
model.train(data='training_obb_angled/data.yaml', epochs=100, imgsz=640)
```

---

## 📊 Images de debug

L'analyseur génère des images de debug dans `output/<image>/debug/` :

| Fichier | Description |
|---------|-------------|
| `01_detections_obb.png` | Toutes les détections OBB |
| `02_ruler_calibration.png` | Calibration avec la règle |
| `03_spikes_analysis.png` | Épis avec mesures et épillets |
| `04_bag_identification.png` | Sachet avec OCR |
| `05_final_result.png` | Résultat final annoté |

---

## ⚙️ Configuration

Le fichier `config/config.yaml` permet de configurer :

```yaml
yolo:
  model_path: 'models/wheat_spike_yolo.pt'
  confidence_threshold: 0.35
  iou_threshold: 0.45

ruler_detection:
  ruler_length_mm: 300

spikelet_counting:
  enabled: true
  model_path: 'models/spikelets_yolo.pt'

bag_identification:
  enabled: true
  digits_model_path: 'models/bag_digits_yolo.pt'
  opening_model_path: 'models/bag_opening_yolo.pt'
```

---

## 📄 License

Projet de recherche - Thèse INRAE/SupAgro

---

## 👤 Auteur

Nicolas Salas - Thèse sur l'analyse phénotypique du blé
