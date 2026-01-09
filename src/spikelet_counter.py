"""
Module de comptage des épillets.
Utilise YOLO comme méthode principale avec fallback sur méthode fréquentielle.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
    
    def count_yolo(self, roi: np.ndarray) -> Tuple[int, List[Tuple[int, int, int, int]]]:
        """
        Compte les épillets avec YOLO.
        
        Args:
            roi: Image du ROI de l'épi
            
        Returns:
            Tuple (nombre d'épillets, liste des bboxes)
        """
        if self.model is None:
            if not self.load_model():
                return 0, []
        
        results = self.model(roi, conf=self.confidence, verbose=False)

        raw_boxes = []  # (x1,y1,x2,y2,score)
        for r in results:
            for box in r.boxes:
                xy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xy)
                score = float(box.conf.cpu().numpy()) if hasattr(box, 'conf') else 1.0
                raw_boxes.append((x1, y1, x2, y2, score))

        # Apply IoU NMS
        boxes = [b[:4] for b in raw_boxes]
        scores = [b[4] for b in raw_boxes]
        keep_idxs = self._nms(boxes, scores, iou_threshold=self.nms_iou)

        bboxes = [raw_boxes[i][:4] for i in keep_idxs]

        return len(bboxes), bboxes

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
    
    def count(
        self, 
        roi: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> SpikeletResult:
        """
        Compte les épillets avec la stratégie recommandée:
        - YOLO comme méthode principale
        - Fallback sur gradient si YOLO échoue
        
        Args:
            roi: Image du ROI de l'épi
            mask: Masque optionnel de l'épi
            
        Returns:
            SpikeletResult avec le comptage et les métadonnées
        """
        # Comptage YOLO
        yolo_count, bboxes = self.count_yolo(roi)

        # Filtrer les bboxes dont le centroïde n'est pas dans le masque (si fourni)
        if mask is not None and len(bboxes) > 0:
            filtered = []
            for (x1, y1, x2, y2) in bboxes:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                # protéger index
                if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
                    if mask[cy, cx] > 0:
                        filtered.append((x1, y1, x2, y2))
                # if centroid outside mask, discard
            bboxes = filtered
            yolo_count = len(bboxes)

        # Déduplication par distance entre centroïdes (pour éviter doubles très proches)
        if len(bboxes) > 1:
            kept = []
            centers = []
            for (x1, y1, x2, y2) in bboxes:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                too_close = False
                for (ox, oy) in centers:
                    if (cx - ox) ** 2 + (cy - oy) ** 2 <= (self.dedup_radius ** 2):
                        too_close = True
                        break
                if not too_close:
                    kept.append((int(x1), int(y1), int(x2), int(y2)))
                    centers.append((cx, cy))
            bboxes = kept
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
                bboxes=bboxes
            )
        elif yolo_count >= self.LOW_CONFIDENCE_THRESHOLD:
            # Confiance moyenne, utiliser YOLO mais marquer
            return SpikeletResult(
                count=yolo_count,
                method='yolo',
                confidence='medium',
                yolo_count=yolo_count,
                gradient_count=gradient_count,
                bboxes=bboxes
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
                    bboxes=bboxes
                )
            else:
                # Garder YOLO même si faible
                return SpikeletResult(
                    count=yolo_count,
                    method='yolo',
                    confidence='low',
                    yolo_count=yolo_count,
                    gradient_count=gradient_count,
                    bboxes=bboxes
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
