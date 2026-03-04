# Models

Modèles YOLO pour l'analyse d'épis de blé (stockés via Git LFS).

## Modèles disponibles

| Fichier | Type YOLO | Classes | Architecture de base | Description |
|---------|-----------|---------|---------------------|-------------|
| `wheat_spike_yolo.pt` | OBB | ruler (0), spike (1), bag (2), whole_spike (3) | yolo26s-obb | Détection principale orientée — 4 classes |
| `graduations_yolo.pt` | OBB | 0cm, 10cm, 20cm, 30cm | — | Graduations sur la règle pour calibration |
| `spikelets_yolo.pt` | Detect | spikelet | — | Comptage/détection des épillets |
| `spike_seg_yolo.pt` | Segment | spike | yolo26s-seg | Segmentation pixel-level des épis |
| `rachis_yolo.pt` | Segment | rachis | yolo26l-seg | Segmentation de l'axe central (rachis) |
| `bag_digits_yolo.pt` | Detect | 1 à 20 (20 classes) | — | OCR chiffres manuscrits sur sachets |
| `bag_opening_yolo.pt` | Detect | bag_opening | — | Détection de l'ouverture du sachet (orientation) |

Note : `spikelets_yolo_save.pt` est une sauvegarde de backup.

## Téléchargement

Les modèles sont gérés via Git LFS. Après clonage :

```bash
git lfs pull
```

## Datasets d'entraînement associés

Chaque modèle a son dataset dans un dossier `training_*/` à la racine :

| Modèle | Dataset | Nombre d'images (approx.) |
|--------|---------|--------------------------|
| `wheat_spike_yolo.pt` | `training_obb_angled/` | variable |
| `graduations_yolo.pt` | `training_graduations/` | variable |
| `spikelets_yolo.pt` | `training_spikelets/` | variable |
| `spike_seg_yolo.pt` | `training_spike_segmentation/` | 160 train + 40 val |
| `rachis_yolo.pt` | `training_rachis/` | 80-150 recommandé |
| `bag_digits_yolo.pt` | `training_bag_digits/` | 592 train + 198 val |
| `bag_opening_yolo.pt` | `training_bag_opening/` | ~100 recommandé |
