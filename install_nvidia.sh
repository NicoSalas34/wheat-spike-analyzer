#!/bin/bash
# =============================================================================
# Script d'installation pour NVIDIA CUDA (RTX 4070)
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "Installation Wheat Spike Analyzer pour NVIDIA"
echo "=============================================="

if ! command -v python3 >/dev/null 2>&1; then
    echo "Erreur: python3 introuvable"
    exit 1
fi

# Informations GPU NVIDIA
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "✓ NVIDIA détecté:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
else
    echo "⚠ nvidia-smi introuvable. Vérifiez le driver NVIDIA."
fi

# Créer/recréer un environnement virtuel si absent ou incomplet
if [ ! -d "venv" ] || [ ! -f "venv/bin/activate" ]; then
    if [ -d "venv" ] && [ ! -f "venv/bin/activate" ]; then
        echo "⚠ Environnement virtuel incomplet détecté, recréation..."
        rm -rf venv
    fi
    echo "Création de l'environnement virtuel..."
    python3 -m venv venv
    echo "✓ Environnement virtuel créé"
fi

# Activer venv
# shellcheck disable=SC1091
source venv/bin/activate
echo "✓ Environnement virtuel activé"

# Outils Python de base
python -m pip install --upgrade pip setuptools wheel

# Installer PyTorch CUDA
echo "Installation PyTorch CUDA..."
pip uninstall -y torch torchvision torchaudio triton pytorch-triton-rocm 2>/dev/null || true

# Pour Python 3.13, cu124 propose des roues plus à jour que cu121
if ! pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124; then
    echo "⚠ torchaudio indisponible pour cette combinaison Python/CUDA, installation sans torchaudio..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi

# Dépendances du projet
echo "Installation des dépendances du projet..."
pip install -r requirements.txt

# Installer le package en mode editable
pip install -e .

# Vérification finale
echo ""
echo "=============================================="
echo "Vérification de la configuration GPU"
echo "=============================================="
python - <<'PY'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version (build): {torch.version.cuda}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU détecté: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")
else:
    print("⚠ GPU CUDA non disponible, fallback CPU")
PY

echo ""
echo "=============================================="
echo "Installation terminée"
echo "=============================================="
echo ""
echo "Utilisation:"
echo "  source venv/bin/activate"
echo "  python src/main.py data/test_sample/ --batch --low-debug"
