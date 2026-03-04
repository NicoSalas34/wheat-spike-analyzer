"""
Module de comptage des épillets.
Utilise YOLO comme méthode principale avec fallback sur méthode fréquentielle.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import signal
from scipy.ndimage import rotate
import logging

from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class SpikeletResult:
    """Résultat du comptage d'épillets pour un épi."""
    count: int
    method: str  # 'yolo', 'gradient', 'hybrid'
    confidence: str  # 'high', 'medium', 'low'
    yolo_count: int
    gradient_count: int
    bboxes: List[Tuple[int, int, int, int]]  # Liste des bbox des épillets
    masks: Optional[List[np.ndarray]] = None  # Masques YOLO-Seg par épillet (coords ROI)


class SpikeletCounter:
    """Compteur d'épillets avec YOLO et fallback fréquentiel."""
    
    # Seuils pour la confiance
    HIGH_CONFIDENCE_THRESHOLD = 10  # Si YOLO détecte >= 10, haute confiance
    LOW_CONFIDENCE_THRESHOLD = 5    # Si YOLO détecte < 5, basse confiance
    
    def __init__(
        self,
        model_path: str = "models/spikelets_yolo.pt",
        confidence: float = 0.25,
        use_fallback: bool = True
    ):
        """
        Initialise le compteur d'épillets.
        
        Args:
            model_path: Chemin vers le modèle YOLO pour les épillets
            confidence: Seuil de confiance YOLO
            use_fallback: Utiliser la méthode gradient en fallback
        """
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.use_fallback = use_fallback
        self.model = None
        # NMS and deduplication parameters
        self.nms_iou = 0.3
        self.dedup_radius = 10  # pixels
        
    def load_model(self):
        """Charge le modèle YOLO."""
        if self.model is None:
            if not self.model_path.exists():
                logger.warning(f"Modèle épillets non trouvé: {self.model_path}")
                return False
            logger.info(f"Chargement du modèle épillets: {self.model_path}")
            self.model = YOLO(str(self.model_path))
        return True
    
    def count_yolo(self, roi: np.ndarray) -> Tuple[int, List[Tuple[int, int, int, int]], Optional[List[np.ndarray]]]:
        """
        Compte les épillets avec YOLO (+ masques si modèle YOLO-Seg).
        
        Args:
            roi: Image du ROI de l'épi
            
        Returns:
            Tuple (nombre d'épillets, liste des bboxes, liste des masques ou None)
        """
        if self.model is None:
            if not self.load_model():
                return 0, [], None
        
        results = self.model(roi, conf=self.confidence, verbose=False)

        raw_boxes = []  # (x1,y1,x2,y2,score)
        raw_masks = []  # masques binaires ROI ou None
        has_masks = False
        
        for r in results:
            # Vérifier si le modèle produit des masques (YOLO-Seg)
            seg_masks = None
            if r.masks is not None and r.masks.xy is not None:
                seg_masks = r.masks.xy
                has_masks = True
            
            for i, box in enumerate(r.boxes):
                xy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xy)
                score = float(box.conf[0]) if hasattr(box, 'conf') else 1.0
                raw_boxes.append((x1, y1, x2, y2, score))
                
                # Extraire le masque correspondant
                if seg_masks is not None and i < len(seg_masks):
                    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                    pts = seg_masks[i].astype(np.int32)
                    if len(pts) >= 3:
                        cv2.fillPoly(mask, [pts], 255)
                    raw_masks.append(mask)
                else:
                    raw_masks.append(None)

        # Apply IoU NMS
        boxes = [b[:4] for b in raw_boxes]
        scores = [b[4] for b in raw_boxes]
        keep_idxs = self._nms(boxes, scores, iou_threshold=self.nms_iou)

        bboxes = [raw_boxes[i][:4] for i in keep_idxs]
        masks = [raw_masks[i] for i in keep_idxs] if has_masks else None

        return len(bboxes), bboxes, masks

    def _nms(self, boxes: List[Tuple[int, int, int, int]], scores: List[float], iou_threshold: float = 0.3) -> List[int]:
        """
        Simple IoU-based NMS returning indices to keep.
        """
        if not boxes:
            return []

        boxes_arr = np.array(boxes, dtype=float)
        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = boxes_arr[:, 2]
        y2 = boxes_arr[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep
    
    def count_gradient(self, roi: np.ndarray, mask: Optional[np.ndarray] = None) -> int:
        """
        Compte les épillets via la méthode du gradient.
        Analyse les oscillations du profil d'intensité le long de l'épi.
        
        Args:
            roi: Image du ROI de l'épi
            mask: Masque optionnel de l'épi (si None, utilise toute l'image)
            
        Returns:
            Nombre estimé d'épillets
        """
        # Convertir en niveaux de gris
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Créer un masque si non fourni
        if mask is None:
            # Masque basé sur la couleur du blé (jaune/doré)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower = np.array([15, 30, 60])
            upper = np.array([45, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Orienter verticalement si possible
        mask_oriented, angle = self._orient_vertically(mask)
        if angle != 0:
            gray = rotate(gray, angle, reshape=True, mode='constant', cval=0).astype(np.uint8)
            mask_oriented = (mask_oriented > 0.5).astype(np.uint8)
            # S'assurer que gray et mask ont les mêmes dimensions après rotation
            if gray.shape != mask_oriented.shape:
                mask_oriented = cv2.resize(
                    mask_oriented, (gray.shape[1], gray.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
        else:
            mask_oriented = (mask > 0).astype(np.uint8)
        
        # Extraire le profil d'intensité
        profile = self._extract_profile(gray, mask_oriented)
        
        if len(profile) < 20:
            return 0
        
        # Compter les oscillations
        count = self._count_oscillations(profile)
        
        return count
    
    def _orient_vertically(self, mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """Oriente le masque verticalement."""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return mask, 0
        
        cnt = max(contours, key=cv2.contourArea)
        
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            angle = ellipse[2]
        else:
            rect = cv2.minAreaRect(cnt)
            angle = rect[2]
        
        rotation_angle = 90 - angle if angle > 45 else -angle
        mask_rot = rotate(mask.astype(float), rotation_angle, reshape=True, mode='constant', cval=0)
        
        return mask_rot, rotation_angle
    
    def _extract_profile(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extrait le profil d'intensité moyen par ligne."""
        profile = []
        for i in range(gray.shape[0]):
            row = gray[i, :] * mask[i, :]
            valid = mask[i, :].sum()
            if valid > 5:
                profile.append(row.sum() / valid)
        return np.array(profile)
    
    def _count_oscillations(self, profile: np.ndarray, min_distance: int = 6) -> int:
        """Compte les oscillations dans le profil."""
        if len(profile) < 10:
            return 0
        
        # Lisser
        kernel_size = max(3, len(profile) // 30)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = min(kernel_size, len(profile) - 1)
        
        try:
            profile_smooth = signal.savgol_filter(profile, kernel_size, 2)
        except:
            profile_smooth = np.convolve(profile, np.ones(5)/5, mode='same')
        
        # Trouver pics et vallées
        peaks, _ = signal.find_peaks(profile_smooth, distance=min_distance, prominence=1)
        valleys, _ = signal.find_peaks(-profile_smooth, distance=min_distance, prominence=1)
        
        # Nombre d'oscillations ≈ cycles
        count = (len(peaks) + len(valleys)) // 2
        
        return count
    
    # =========================================================================
    # TTA (Test-Time Augmentation)
    # =========================================================================
    
    def _match_masks_to_tta_bboxes(
        self,
        tta_bboxes: List[Tuple[int, int, int, int]],
        orig_bboxes: List[Tuple[int, int, int, int]],
        orig_masks: List[np.ndarray],
        image_shape: Tuple[int, int],
    ) -> Optional[List[np.ndarray]]:
        """
        Pour chaque bbox TTA, trouver la bbox originale la plus proche (par IoU)
        et retourner son masque. Cela permet de récupérer les masques YOLO-Seg
        même quand le comptage a été fait par TTA.
        
        Returns:
            Liste de masques correspondant aux tta_bboxes, ou None si pas de match
        """
        if not tta_bboxes or not orig_bboxes or not orig_masks:
            return None
        
        matched_masks = []
        for tta_box in tta_bboxes:
            best_iou = 0.0
            best_idx = -1
            for j, orig_box in enumerate(orig_bboxes):
                iou = self._compute_single_iou(tta_box, orig_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            
            if best_idx >= 0 and best_iou > 0.2 and orig_masks[best_idx] is not None:
                matched_masks.append(orig_masks[best_idx])
            else:
                # Pas de masque trouvé : créer un masque à partir de la bbox
                mask = np.zeros(image_shape, dtype=np.uint8)
                x1, y1, x2, y2 = tta_box
                mask[y1:y2, x1:x2] = 255
                matched_masks.append(mask)
        
        return matched_masks if matched_masks else None
    
    def count_yolo_tta(
        self,
        roi: np.ndarray,
        augmentations: Optional[List[str]] = None,
        nms_iou_tta: float = 0.3,
    ) -> Tuple[int, List[Tuple[int, int, int, int]]]:
        """
        Comptage YOLO avec Test-Time Augmentation.
        
        Applique plusieurs transformations géométriques, exécute l'inférence
        sur chaque variante, projette les détections dans l'espace original,
        puis agrège avec Weighted Box Fusion (WBF) / NMS.
        
        Args:
            roi: Image du ROI de l'épi (BGR)
            augmentations: Liste de transformations parmi:
                'original', 'flip_h', 'flip_v', 'rot180',
                'rot90_cw', 'rot90_ccw'
                Défaut: ['original', 'flip_h', 'flip_v', 'rot180']
            nms_iou_tta: Seuil IoU pour le NMS d'agrégation TTA
            
        Returns:
            Tuple (count, bboxes) avec les détections agrégées
        """
        if augmentations is None:
            augmentations = ['original', 'flip_h', 'flip_v', 'rot180']
        
        h, w = roi.shape[:2]
        all_boxes = []  # (x1, y1, x2, y2, score)
        
        for aug_name in augmentations:
            # Appliquer la transformation
            aug_image, inverse_fn = self._apply_augmentation(roi, aug_name)
            
            if aug_image is None:
                continue
            
            # Inférence YOLO
            count, bboxes, _masks = self.count_yolo(aug_image)
            
            # Projeter les bboxes dans l'espace original
            for (bx1, by1, bx2, by2) in bboxes:
                orig_bbox = inverse_fn(bx1, by1, bx2, by2)
                if orig_bbox is not None:
                    ox1, oy1, ox2, oy2 = orig_bbox
                    # Clamp dans les limites de l'image originale
                    ox1 = max(0, min(w - 1, ox1))
                    oy1 = max(0, min(h - 1, oy1))
                    ox2 = max(0, min(w - 1, ox2))
                    oy2 = max(0, min(h - 1, oy2))
                    if ox2 > ox1 and oy2 > oy1:
                        # Score pondéré par 1/n_augmentations
                        score = 1.0 / len(augmentations)
                        all_boxes.append((ox1, oy1, ox2, oy2, score))
        
        if not all_boxes:
            return 0, []
        
        # Agrégation via Weighted NMS
        boxes = [b[:4] for b in all_boxes]
        scores = [b[4] for b in all_boxes]
        keep_idxs = self._nms(boxes, scores, iou_threshold=nms_iou_tta)
        
        # Pour chaque boîte gardée, calculer la boîte moyenne pondérée
        # des boîtes qui lui étaient proches (soft-NMS-like averaging)
        final_bboxes = []
        for idx in keep_idxs:
            anchor = all_boxes[idx]
            ax1, ay1, ax2, ay2 = anchor[:4]
            
            # Trouver toutes les boîtes proches
            cluster_boxes = []
            cluster_scores = []
            for bx1, by1, bx2, by2, sc in all_boxes:
                iou = self._compute_single_iou(
                    (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2)
                )
                if iou > 0.3:  # Même cluster
                    cluster_boxes.append((bx1, by1, bx2, by2))
                    cluster_scores.append(sc)
            
            if cluster_boxes:
                # Moyenne pondérée par les scores
                total_score = sum(cluster_scores)
                avg_x1 = sum(b[0] * s for b, s in zip(cluster_boxes, cluster_scores)) / total_score
                avg_y1 = sum(b[1] * s for b, s in zip(cluster_boxes, cluster_scores)) / total_score
                avg_x2 = sum(b[2] * s for b, s in zip(cluster_boxes, cluster_scores)) / total_score
                avg_y2 = sum(b[3] * s for b, s in zip(cluster_boxes, cluster_scores)) / total_score
                final_bboxes.append((int(avg_x1), int(avg_y1), int(avg_x2), int(avg_y2)))
            else:
                final_bboxes.append((int(ax1), int(ay1), int(ax2), int(ay2)))
        
        return len(final_bboxes), final_bboxes
    
    def _apply_augmentation(
        self, image: np.ndarray, aug_name: str
    ) -> Tuple[Optional[np.ndarray], Optional[callable]]:
        """
        Applique une augmentation et retourne l'image transformée
        ainsi que la fonction inverse pour reprojeter les bboxes.
        
        Args:
            image: Image BGR
            aug_name: Nom de l'augmentation
            
        Returns:
            (image_augmentée, fonction_inverse)
            La fonction inverse prend (x1, y1, x2, y2) et
            retourne (x1', y1', x2', y2') dans l'espace original.
        """
        h, w = image.shape[:2]
        
        if aug_name == 'original':
            return image, lambda x1, y1, x2, y2: (x1, y1, x2, y2)
        
        elif aug_name == 'flip_h':
            aug = cv2.flip(image, 1)  # Flip horizontal
            def inv_flip_h(x1, y1, x2, y2):
                return (w - 1 - x2, y1, w - 1 - x1, y2)
            return aug, inv_flip_h
        
        elif aug_name == 'flip_v':
            aug = cv2.flip(image, 0)  # Flip vertical
            def inv_flip_v(x1, y1, x2, y2):
                return (x1, h - 1 - y2, x2, h - 1 - y1)
            return aug, inv_flip_v
        
        elif aug_name == 'rot180':
            aug = cv2.flip(image, -1)  # Rotation 180° = flip both
            def inv_rot180(x1, y1, x2, y2):
                return (w - 1 - x2, h - 1 - y2, w - 1 - x1, h - 1 - y1)
            return aug, inv_rot180
        
        elif aug_name == 'rot90_cw':
            # Rotation 90° horaire: (x,y) → (h-1-y, x)
            aug = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            def inv_rot90_cw(x1, y1, x2, y2):
                # Inverse: (x,y) → (y, w_aug-1-x) avec w_aug=h_orig
                return (y1, h - 1 - x2, y2, h - 1 - x1)
            return aug, inv_rot90_cw
        
        elif aug_name == 'rot90_ccw':
            # Rotation 90° anti-horaire: (x,y) → (y, w-1-x)
            aug = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            def inv_rot90_ccw(x1, y1, x2, y2):
                return (w - 1 - y2, x1, w - 1 - y1, x2)
            return aug, inv_rot90_ccw
        
        else:
            logger.warning(f"Augmentation inconnue: {aug_name}")
            return None, None
    
    @staticmethod
    def _compute_single_iou(
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        """Calcule l'IoU entre deux bboxes axis-aligned."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0
    
    # =========================================================================
    # MÉTHODE PRINCIPALE DE COMPTAGE
    # =========================================================================
    
    def count(
        self, 
        roi: np.ndarray, 
        mask: Optional[np.ndarray] = None,
        use_tta: bool = False,
        tta_augmentations: Optional[List[str]] = None,
    ) -> SpikeletResult:
        """
        Compte les épillets avec la stratégie recommandée:
        - YOLO comme méthode principale (avec TTA optionnel)
        - Fallback sur gradient si YOLO échoue
        
        Args:
            roi: Image du ROI de l'épi
            mask: Masque optionnel de l'épi
            use_tta: Si True, utilise le TTA pour le comptage YOLO
            tta_augmentations: Liste d'augmentations TTA (défaut: 4 variantes)
            
        Returns:
            SpikeletResult avec le comptage et les métadonnées
        """
        # Comptage YOLO (avec ou sans TTA)
        spikelet_masks = None
        if use_tta:
            yolo_count, bboxes = self.count_yolo_tta(
                roi, augmentations=tta_augmentations
            )
            # TTA ne retourne pas de masques. Récupérer les masques en faisant
            # un seul passage supplémentaire sur l'image originale, puis matcher
            # les masques aux bboxes TTA par IoU.
            if yolo_count > 0:
                _, orig_bboxes, orig_masks = self.count_yolo(roi)
                if orig_masks is not None and len(orig_masks) > 0:
                    spikelet_masks = self._match_masks_to_tta_bboxes(
                        bboxes, orig_bboxes, orig_masks, roi.shape[:2]
                    )
        else:
            yolo_count, bboxes, spikelet_masks = self.count_yolo(roi)

        # Filtrer les bboxes dont le centroïde n'est pas dans le masque (si fourni)
        if mask is not None and len(bboxes) > 0:
            filtered = []
            filtered_masks = [] if spikelet_masks is not None else None
            for i, (x1, y1, x2, y2) in enumerate(bboxes):
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                # protéger index
                if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
                    if mask[cy, cx] > 0:
                        filtered.append((x1, y1, x2, y2))
                        if filtered_masks is not None:
                            filtered_masks.append(spikelet_masks[i])
                # if centroid outside mask, discard
            bboxes = filtered
            spikelet_masks = filtered_masks
            yolo_count = len(bboxes)

        # Déduplication par distance entre centroïdes (pour éviter doubles très proches)
        if len(bboxes) > 1:
            kept = []
            kept_masks = [] if spikelet_masks is not None else None
            centers = []
            for i, (x1, y1, x2, y2) in enumerate(bboxes):
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                too_close = False
                for (ox, oy) in centers:
                    if (cx - ox) ** 2 + (cy - oy) ** 2 <= (self.dedup_radius ** 2):
                        too_close = True
                        break
                if not too_close:
                    kept.append((int(x1), int(y1), int(x2), int(y2)))
                    if kept_masks is not None:
                        kept_masks.append(spikelet_masks[i])
                    centers.append((cx, cy))
            bboxes = kept
            spikelet_masks = kept_masks
            yolo_count = len(bboxes)
        
        # Comptage gradient (toujours calculé pour comparaison)
        gradient_count = 0
        if self.use_fallback:
            gradient_count = self.count_gradient(roi, mask)
        
        # Décision basée sur la confiance
        if yolo_count >= self.HIGH_CONFIDENCE_THRESHOLD:
            # Haute confiance dans YOLO
            return SpikeletResult(
                count=yolo_count,
                method='yolo',
                confidence='high',
                yolo_count=yolo_count,
                gradient_count=gradient_count,
                bboxes=bboxes,
                masks=spikelet_masks,
            )
        elif yolo_count >= self.LOW_CONFIDENCE_THRESHOLD:
            # Confiance moyenne, utiliser YOLO mais marquer
            return SpikeletResult(
                count=yolo_count,
                method='yolo',
                confidence='medium',
                yolo_count=yolo_count,
                gradient_count=gradient_count,
                bboxes=bboxes,
                masks=spikelet_masks,
            )
        else:
            # Basse confiance, utiliser fallback
            if self.use_fallback and gradient_count > yolo_count:
                # Utiliser le gradient si plus élevé
                return SpikeletResult(
                    count=gradient_count,
                    method='gradient',
                    confidence='low',
                    yolo_count=yolo_count,
                    gradient_count=gradient_count,
                    bboxes=bboxes,
                    masks=spikelet_masks,
                )
            else:
                # Garder YOLO même si faible
                return SpikeletResult(
                    count=yolo_count,
                    method='yolo',
                    confidence='low',
                    yolo_count=yolo_count,
                    gradient_count=gradient_count,
                    bboxes=bboxes,
                    masks=spikelet_masks,
                )
    
    def draw_detections(
        self, 
        image: np.ndarray, 
        result: SpikeletResult,
        offset: Tuple[int, int] = (0, 0)
    ) -> np.ndarray:
        """
        Dessine les détections sur l'image.
        
        Args:
            image: Image sur laquelle dessiner
            result: Résultat du comptage
            offset: Décalage (x, y) pour les coordonnées
            
        Returns:
            Image avec les annotations
        """
        image_viz = image.copy()
        ox, oy = offset
        
        # Dessiner les bboxes des épillets
        for (x1, y1, x2, y2) in result.bboxes:
            color = (0, 255, 0) if result.confidence == 'high' else \
                    (0, 165, 255) if result.confidence == 'medium' else \
                    (0, 0, 255)
            cv2.rectangle(image_viz, (ox + x1, oy + y1), (ox + x2, oy + y2), color, 2)
        
        return image_viz
