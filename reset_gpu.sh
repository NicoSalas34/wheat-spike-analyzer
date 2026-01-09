#!/bin/bash
# Script de reset des interfaces GPU pour AMD ROCm/HIP

echo "=== Reset GPU AMD ROCm ==="

# 1. Tuer les processus Python bloqués
echo "[1/5] Arrêt des processus Python..."
pkill -9 -f "python.*yolo" 2>/dev/null
pkill -9 -f "python.*torch" 2>/dev/null
pkill -9 -f "python.*wheat" 2>/dev/null
sleep 2

# 2. Nettoyer les caches HIP
echo "[2/5] Nettoyage des caches HIP/PyTorch..."
rm -rf ~/.cache/hip_* 2>/dev/null
rm -rf /tmp/hip* 2>/dev/null
rm -rf ~/.cache/torch_extensions 2>/dev/null
rm -rf ~/.nv 2>/dev/null
rm -rf /tmp/pytorch_* 2>/dev/null

# 3. Reset du module amdgpu (nécessite sudo)
echo "[3/5] Reset du module amdgpu..."
if [ "$EUID" -eq 0 ]; then
    # Si root, reset le GPU
    echo 1 > /sys/kernel/debug/dri/0/amdgpu_gpu_recover 2>/dev/null || true
    echo 1 > /sys/kernel/debug/dri/1/amdgpu_gpu_recover 2>/dev/null || true
else
    echo "  (skip - exécuter avec sudo pour reset hardware)"
fi

# 4. Configurer les variables d'environnement
echo "[4/5] Configuration des variables d'environnement..."
export HSA_OVERRIDE_GFX_VERSION=12.0.0
export HIP_VISIBLE_DEVICES=0
export AMD_LOG_LEVEL=0
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
export HSA_FORCE_FINE_GRAIN_PCIE=1
# Supprimer la variable dépréciée
unset PYTORCH_HIP_ALLOC_CONF

echo "  HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo "  HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo "  PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF"

# 5. Vérifier l'état du GPU
echo "[5/5] Vérification du GPU..."
rocm-smi --showmeminfo vram 2>/dev/null | head -10

echo ""
echo "=== Reset terminé ==="
echo ""
echo "Pour utiliser le GPU, exécutez:"
echo "  source reset_gpu.sh"
echo "  python src/main.py image.JPG"
echo ""
echo "Pour forcer le mode CPU:"
echo "  CUDA_VISIBLE_DEVICES=\"\" python src/main.py image.JPG"
