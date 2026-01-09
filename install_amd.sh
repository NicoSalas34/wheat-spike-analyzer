#!/bin/bash
# =============================================================================
# Script d'installation pour AMD ROCm (RX 6000/7000/9000 series)
# =============================================================================
# Ce script installe les dépendances PyTorch avec le support ROCm pour les
# GPU AMD comme la RX 9070 XT.
#
# Prérequis:
# - Ubuntu 22.04/24.04 ou autre distribution Linux supportée
# - Drivers AMD AMDGPU installés
# - ROCm 7.x installé (voir https://rocm.docs.amd.com/)
# =============================================================================

set -e

echo "=============================================="
echo "Installation Wheat Spike Analyzer pour AMD GPU"
echo "=============================================="

# Vérifier si ROCm est installé
if command -v rocminfo &> /dev/null; then
    echo "✓ ROCm détecté"
    rocminfo | grep -E "Name:|Marketing Name:" | head -4
else
    echo "⚠ ROCm n'est pas détecté. Veuillez l'installer d'abord."
    echo "  Guide: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
    echo ""
    echo "Pour Ubuntu 22.04/24.04:"
    echo "  sudo apt update"
    echo "  sudo apt install rocm-hip-runtime rocm-hip-sdk"
    echo ""
    read -p "Continuer quand même? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Créer/activer un environnement virtuel si demandé
if [ ! -d "venv" ]; then
    echo ""
    read -p "Créer un environnement virtuel? (recommandé) (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 -m venv venv
        echo "✓ Environnement virtuel créé"
    fi
fi

if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Environnement virtuel activé"
fi

# Mise à jour pip
echo ""
echo "Mise à jour de pip..."
pip install --upgrade pip

# Vérifier si PyTorch ROCm 7.1.1 est déjà installé
echo ""
echo "Vérification de PyTorch..."
PYTORCH_INSTALLED=false
PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")

if [[ "$PYTORCH_VERSION" == *"rocm7.1.1"* ]]; then
    echo "✓ PyTorch $PYTORCH_VERSION déjà installé avec ROCm 7.1.1"
    PYTORCH_INSTALLED=true
else
    if [ -n "$PYTORCH_VERSION" ]; then
        echo "⚠ PyTorch $PYTORCH_VERSION détecté (pas ROCm 7.1.1)"
    else
        echo "⚠ PyTorch non installé"
    fi
fi

if [ "$PYTORCH_INSTALLED" = false ]; then
    # Installation de PyTorch avec ROCm 7.1.1 (packages officiels AMD)
    echo ""
    echo "Installation de PyTorch avec ROCm 7.1.1..."
    echo "(Téléchargement depuis repo.radeon.com)"
    echo "(La RX 9070 XT est supportée via gfx1201)"

    # Télécharger les packages PyTorch pour ROCm 7.1.1
    echo "Téléchargement des packages..."
    wget -q --show-progress https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/torch-2.9.1%2Brocm7.1.1.lw.git351ff442-cp312-cp312-linux_x86_64.whl
    wget -q --show-progress https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/torchvision-0.24.0%2Brocm7.1.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl
    wget -q --show-progress https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/torchaudio-2.9.0%2Brocm7.1.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl
    wget -q --show-progress https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/triton-3.5.1%2Brocm7.1.1.gita272dfa8-cp312-cp312-linux_x86_64.whl

    # Désinstaller les anciennes versions
    echo "Désinstallation des anciennes versions..."
    pip uninstall -y torch torchvision torchaudio triton pytorch-triton-rocm 2>/dev/null || true

    # Installer les nouvelles versions
    echo "Installation des packages ROCm 7.1.1..."
    pip install torch-2.9.1+rocm7.1.1.lw.git351ff442-cp312-cp312-linux_x86_64.whl \
        torchvision-0.24.0+rocm7.1.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl \
        torchaudio-2.9.0+rocm7.1.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl \
        triton-3.5.1+rocm7.1.1.gita272dfa8-cp312-cp312-linux_x86_64.whl

    # Nettoyer les fichiers .whl téléchargés
    echo "Nettoyage des fichiers temporaires..."
    rm -f torch-*.whl torchvision-*.whl torchaudio-*.whl triton-*.whl
fi

# Vérifier l'installation PyTorch
echo ""
echo "Vérification de PyTorch..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm/HIP version: {torch.version.hip}')
print(f'CUDA disponible (via ROCm): {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU détecté: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'VRAM: {props.total_memory / 1024**3:.1f} GB')
"

# Installation des autres dépendances
echo ""
echo "Installation des dépendances Python..."
pip install numpy opencv-python opencv-contrib-python scikit-image scikit-learn scipy pillow
pip install pandas openpyxl pyyaml matplotlib seaborn tqdm

# Installation des dépendances ML
echo ""
echo "Installation des dépendances ML..."
pip install transformers>=4.40.0
pip install ultralytics>=8.1.0

# Installation de Segment Anything (SAM)
echo ""
read -p "Installer Segment Anything Model (SAM)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install git+https://github.com/facebookresearch/segment-anything.git
    echo "✓ SAM installé"
fi

# Installation du package wheat-spike-analyzer
echo ""
echo "Installation du package wheat-spike-analyzer..."
pip install -e .

# Test final
echo ""
echo "=============================================="
echo "Test de la configuration GPU..."
echo "=============================================="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm/HIP version: {torch.version.hip if hasattr(torch.version, \"hip\") else \"N/A\"}')
print(f'CUDA disponible (via ROCm): {torch.cuda.is_available()}')
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    print(f'GPU détecté: {device_name}')
    print(f'VRAM: {vram_gb:.1f} GB')
    print()
    print('🎉 Configuration AMD ROCm 7.1.1 réussie!')
else:
    print('⚠ GPU non détecté. Mode CPU.')
"

echo ""
echo "=============================================="
echo "Installation terminée!"
echo "=============================================="
echo ""
echo "Pour utiliser l'application:"
echo "  source venv/bin/activate  # si vous utilisez un venv"
echo "  python -m src.main --help"
echo ""
echo "Pour vérifier le GPU:"
echo "  python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\")'"
echo ""
