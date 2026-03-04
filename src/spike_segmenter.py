#!/usr/bin/env python3
"""
Module de segmentation pixel-level des épis de blé via SAM2.

Utilise SAM2 (Segment Anything Model 2) avec les détections OBB comme
box-prompts pour obtenir des masques précis de chaque épi.

Cela permet de calculer :
- Aire et périmètre réels (contour, pas OBB)
- Circularité / facteur de forme
- Couleur moyenne de l'épi (HSV)
- Profil de largeur (apical, médian, basal)
- Courbure du squelette
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


class SpikeSegmenter:
    """
    Segmentation pixel-level des épis via SAM2 (ultralytics).
    
    Utilise les détections OBB comme box-prompts pour SAM2.
    Le modèle SAM2 est chargé via ultralytics, aucune dépendance
    supplémentaire n'est nécessaire.
    """
    
    # Modèles SAM2 disponibles (du plus léger au plus lourd)
    AVAILABLE_MODELS = [
        'sam2_t.pt',   # Tiny   ~40MB
        'sam2_s.pt',   # Small  ~46MB
        'sam2_b.pt',   # Base   ~160MB
        'sam2_l.pt',   # Large  ~340MB
    ]
    
    def __init__(
        self,
        model_name: str = 'sam2_t.pt',
        device: Optional[str] = None,
        min_mask_area: int = 1000,
    ):
        """
        Args:
            model_name: Nom du modèle SAM2 (sam2_t.pt, sam2_s.pt, sam2_b.pt, sam2_l.pt)
            device: Device pour l'inférence (None = auto)
            min_mask_area: Aire minimale du masque en pixels² pour être valide
        """
        self.model_name = model_name
        self.device = device
        self.min_mask_area = min_mask_area
        self.model = None
        
    def load_model(self) -> bool:
        """Charge le modèle SAM2 via ultralytics."""
        try:
            from ultralytics import SAM
            
            logger.info(f"Chargement du modèle SAM2: {self.model_name}")
            self.model = SAM(self.model_name)
            logger.info(f"✓ SAM2 chargé: {self.model_name}")
            return True
            
        except ImportError:
            logger.error("ultralytics n'est pas installé ou ne supporte pas SAM2")
            return False
        except Exception as e:
            logger.error(f"Erreur chargement SAM2 ({self.model_name}): {e}")
            # Essayer un modèle plus léger
            for fallback_model in self.AVAILABLE_MODELS:
                if fallback_model != self.model_name:
                    try:
                        logger.info(f"Fallback vers {fallback_model}...")
                        from ultralytics import SAM
                        self.model = SAM(fallback_model)
                        self.model_name = fallback_model
                        logger.info(f"✓ SAM2 fallback chargé: {fallback_model}")
                        return True
                    except Exception:
                        continue
            return False
    
    def is_available(self) -> bool:
        """Vérifie si le segmenteur est disponible."""
        return self.model is not None
    
    def segment_from_bbox(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """
        Segmente un objet dans l'image à partir d'une bounding box.
        
        Args:
            image: Image BGR complète
            bbox: Bounding box axis-aligned (x1, y1, x2, y2)
            
        Returns:
            Masque binaire (h, w) dtype=uint8 (255=objet, 0=fond) ou None si échec
        """
        if not self.is_available():
            if not self.load_model():
                return None
        
        try:
            # SAM2 via ultralytics attend des bboxes au format [[x1, y1, x2, y2]]
            results = self.model(
                image,
                bboxes=[list(bbox)],
                verbose=False,
            )
            
            if not results or len(results) == 0:
                return None
            
            result = results[0]
            
            if result.masks is None or len(result.masks.data) == 0:
                return None
            
            # Prendre le meilleur masque
            mask = result.masks.data[0].cpu().numpy()
            
            # Redimensionner au format de l'image originale si nécessaire
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            
            # Binariser
            mask_binary = (mask > 0.5).astype(np.uint8) * 255
            
            # Vérifier l'aire minimale
            if cv2.countNonZero(mask_binary) < self.min_mask_area:
                logger.debug(f"Masque trop petit ({cv2.countNonZero(mask_binary)} px²)")
                return None
            
            return mask_binary
            
        except Exception as e:
            logger.warning(f"Erreur segmentation SAM2: {e}")
            return None
    
    def segment_spike(
        self,
        image: np.ndarray,
        obb_detection,
        margin: int = 20,
    ) -> Optional[Dict]:
        """
        Segmente un épi à partir de sa détection OBB.
        
        Args:
            image: Image BGR complète
            obb_detection: OBBDetection de l'épi (spike ou whole_spike)
            margin: Marge autour de la bbox pour le prompt SAM
            
        Returns:
            Dict avec 'mask' (full-size), 'roi_mask', 'roi_bbox', 'contour'
            ou None si échec
        """
        h, w = image.shape[:2]
        
        # Obtenir la bbox axis-aligned avec marge
        x1, y1, x2, y2 = obb_detection.bbox
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        # Segmenter avec SAM2
        full_mask = self.segment_from_bbox(image, (x1, y1, x2, y2))
        
        if full_mask is None:
            return None
        
        # Extraire la ROI du masque
        roi_mask = full_mask[y1:y2, x1:x2]
        
        # Trouver le contour principal
        contours, _ = cv2.findContours(
            full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Prendre le plus grand contour
        main_contour = max(contours, key=cv2.contourArea)
        
        return {
            'mask': full_mask,
            'roi_mask': roi_mask,
            'roi_bbox': (x1, y1, x2, y2),
            'contour': main_contour,
            'contour_area_px': cv2.contourArea(main_contour),
            'contour_perimeter_px': cv2.arcLength(main_contour, True),
        }
    
    def compute_contour_metrics(
        self,
        segmentation_result: Dict,
        pixel_per_mm: Optional[float] = None,
    ) -> Dict:
        """
        Calcule les métriques morphologiques à partir du masque de segmentation.
        
        Args:
            segmentation_result: Résultat de segment_spike()
            pixel_per_mm: Ratio de calibration
            
        Returns:
            Dict avec area, perimeter, circularity, etc.
        """
        contour = segmentation_result['contour']
        mask = segmentation_result['mask']
        area_px = segmentation_result['contour_area_px']
        perimeter_px = segmentation_result['contour_perimeter_px']
        
        metrics = {
            'real_area_px': area_px,
            'real_perimeter_px': perimeter_px,
        }
        
        # Circularité : 4π·area / perimeter²
        if perimeter_px > 0:
            metrics['circularity'] = (4 * np.pi * area_px) / (perimeter_px ** 2)
        else:
            metrics['circularity'] = 0.0
        
        # Conversion en mm si calibré
        if pixel_per_mm and pixel_per_mm > 0:
            metrics['real_area_mm2'] = area_px / (pixel_per_mm ** 2)
            metrics['real_perimeter_mm'] = perimeter_px / pixel_per_mm
        
        # Convex hull et convexité
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            metrics['solidity'] = area_px / hull_area  # ratio area/convex_hull_area
        else:
            metrics['solidity'] = 0.0
        
        # Ellipse ajustée (si assez de points)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (ecx, ecy), (minor_axis, major_axis), angle = ellipse
            metrics['ellipse_major_axis_px'] = major_axis
            metrics['ellipse_minor_axis_px'] = minor_axis
            metrics['ellipse_angle'] = angle
            if minor_axis > 0:
                metrics['ellipse_eccentricity'] = np.sqrt(
                    1 - (minor_axis / major_axis) ** 2
                )
            if pixel_per_mm and pixel_per_mm > 0:
                metrics['ellipse_major_axis_mm'] = major_axis / pixel_per_mm
                metrics['ellipse_minor_axis_mm'] = minor_axis / pixel_per_mm
        
        return metrics
    
    def compute_skeleton_length_width(
        self,
        segmentation_result: Dict,
        pixel_per_mm: Optional[float] = None,
        n_width_samples: int = 30,
    ) -> Dict:
        """
        Mesure la longueur et la largeur de l'épi à partir du squelette
        morphologique du masque de segmentation.
        
        Longueur : plus long chemin dans le squelette (géodésique).
        Largeur  : médiane des largeurs perpendiculaires le long du squelette.
        
        Cette méthode est plus précise que l'OBB car elle suit la forme réelle
        de l'épi, y compris sa courbure.
        
        Args:
            segmentation_result: Résultat de segment_spike()
            pixel_per_mm: Ratio de calibration (optionnel)
            n_width_samples: Nombre de points du squelette où mesurer la largeur
            
        Returns:
            Dict avec seg_length_px, seg_width_px, seg_length_mm, seg_width_mm, etc.
        """
        mask = segmentation_result['mask']
        
        result = {}
        
        # ---- Squelettisation morphologique ----
        mask_binary = (mask > 0).astype(np.uint8)
        
        # Appliquer une légère ouverture morphologique pour lisser
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_clean = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
        
        # Squelettisation via thin itéré (Zhang-Suen)
        skeleton = self._skeletonize(mask_clean)
        
        if skeleton.sum() < 10:
            # Pas assez de pixels dans le squelette
            return result
        
        # ---- Longueur : plus long chemin (géodésique) ----
        endpoints = self._find_skeleton_endpoints(skeleton)
        
        if len(endpoints) < 2:
            # Pas assez de points terminaux, utiliser la longueur de la courbe
            ys, xs = np.where(skeleton > 0)
            if len(ys) < 2:
                return result
            # Fallback : distance entre les deux points squelette les plus éloignés
            pts = np.column_stack((xs, ys))
            d_max, p1, p2 = self._farthest_pair(pts)
            result['seg_length_px'] = float(d_max)
        else:
            # Trouver le plus long chemin géodésique entre endpoints
            length_px, path = self._longest_geodesic_path(skeleton, endpoints)
            result['seg_length_px'] = float(length_px)
        
        # ---- Largeur : perpendiculaires le long du squelette ----
        # Ordonner les points du squelette selon l'axe principal
        ys, xs = np.where(skeleton > 0)
        skel_points = np.column_stack((xs, ys))
        
        if len(skel_points) > n_width_samples:
            # Ordonner par projection sur l'axe principal (PCA)
            ordered_points = self._order_skeleton_points(skel_points)
            
            # Échantillonner n_width_samples points uniformément
            indices = np.linspace(0, len(ordered_points) - 1, n_width_samples, dtype=int)
            sample_points = ordered_points[indices]
            
            # Mesurer la largeur perpendiculaire à chaque point
            widths = []
            for i, pt in enumerate(sample_points):
                # Direction locale du squelette (tangente)
                if i == 0:
                    tangent = sample_points[min(1, len(sample_points)-1)] - pt
                elif i == len(sample_points) - 1:
                    tangent = pt - sample_points[max(0, i-1)]
                else:
                    tangent = sample_points[i+1] - sample_points[i-1]
                
                norm = np.linalg.norm(tangent)
                if norm < 1e-6:
                    continue
                tangent = tangent / norm
                
                # Direction perpendiculaire
                perp = np.array([-tangent[1], tangent[0]])
                
                # Mesurer la largeur dans les deux sens
                w = self._measure_perpendicular_width(mask_binary, pt, perp)
                if w > 0:
                    widths.append(w)
            
            if widths:
                widths = np.array(widths)
                result['seg_width_px'] = float(np.median(widths))
                result['seg_width_mean_px'] = float(np.mean(widths))
                result['seg_width_max_px'] = float(np.max(widths))
                result['seg_width_min_px'] = float(np.min(widths))
                result['seg_width_std_px'] = float(np.std(widths))
        
        # ---- Conversion en mm ----
        if pixel_per_mm and pixel_per_mm > 0:
            if 'seg_length_px' in result:
                result['seg_length_mm'] = result['seg_length_px'] / pixel_per_mm
            if 'seg_width_px' in result:
                result['seg_width_mm'] = result['seg_width_px'] / pixel_per_mm
                result['seg_width_mean_mm'] = result['seg_width_mean_px'] / pixel_per_mm
                result['seg_width_max_mm'] = result['seg_width_max_px'] / pixel_per_mm
                result['seg_width_min_mm'] = result['seg_width_min_px'] / pixel_per_mm
            if 'seg_length_px' in result and 'seg_width_px' in result and result['seg_width_px'] > 0:
                result['seg_aspect_ratio'] = result['seg_length_px'] / result['seg_width_px']
        
        return result
    
    # =========================================================================
    # Méthodes utilitaires pour squelette et mesures
    # =========================================================================
    
    @staticmethod
    def _skeletonize(mask: np.ndarray) -> np.ndarray:
        """
        Squelettisation morphologique par amincissement itératif.
        Implémentation via OpenCV (thinning).
        """
        try:
            # OpenCV >= 4.5 a cv2.ximgproc.thinning
            skeleton = cv2.ximgproc.thinning(
                mask * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
            )
            return (skeleton > 0).astype(np.uint8)
        except AttributeError:
            # Fallback : squelettisation par érosion itérative
            skel = np.zeros_like(mask)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            temp = mask.copy()
            while True:
                eroded = cv2.erode(temp, element)
                opened = cv2.dilate(eroded, element)
                diff = cv2.subtract(temp, opened)
                skel = cv2.bitwise_or(skel, diff)
                temp = eroded.copy()
                if cv2.countNonZero(temp) == 0:
                    break
            return (skel > 0).astype(np.uint8)
    
    @staticmethod
    def _find_skeleton_endpoints(skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """
        Trouve les points terminaux du squelette (voisinage = 1 seul voisin).
        """
        endpoints = []
        ys, xs = np.where(skeleton > 0)
        
        for x, y in zip(xs, ys):
            # Compter les voisins 8-connectés
            patch = skeleton[max(0, y-1):y+2, max(0, x-1):x+2]
            n_neighbors = patch.sum() - 1  # -1 pour le pixel central
            if n_neighbors == 1:
                endpoints.append((x, y))
        
        return endpoints
    
    @staticmethod
    def _farthest_pair(
        points: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Trouve la paire de points la plus éloignée (diamètre)."""
        # Utilise le convex hull pour accélérer
        if len(points) > 4:
            hull = cv2.convexHull(points.astype(np.float32))
            hull = hull.reshape(-1, 2)
        else:
            hull = points
        
        max_dist = 0
        p1, p2 = hull[0], hull[-1]
        for i in range(len(hull)):
            for j in range(i + 1, len(hull)):
                d = np.linalg.norm(hull[i] - hull[j])
                if d > max_dist:
                    max_dist = d
                    p1, p2 = hull[i], hull[j]
        return max_dist, p1, p2
    
    def _longest_geodesic_path(
        self,
        skeleton: np.ndarray,
        endpoints: List[Tuple[int, int]],
    ) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Trouve le plus long chemin géodésique entre endpoints du squelette
        via BFS sur les pixels du squelette.
        """
        from collections import deque
        
        # Construire un graphe implicite sur les pixels du squelette
        # BFS depuis chaque endpoint, garder le plus long chemin
        h, w = skeleton.shape
        
        best_length = 0
        best_path = []
        
        # Limiter à 6 endpoints max pour éviter explosion combinatoire
        test_endpoints = endpoints[:6] if len(endpoints) > 6 else endpoints
        
        for start in test_endpoints:
            # BFS depuis ce point
            dist = np.full((h, w), -1, dtype=np.int32)
            parent = np.full((h, w, 2), -1, dtype=np.int32)
            sx, sy = start
            dist[sy, sx] = 0
            queue = deque([(sx, sy)])
            farthest = (sx, sy)
            max_d = 0
            
            while queue:
                cx, cy = queue.popleft()
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < w and 0 <= ny < h and skeleton[ny, nx] > 0 and dist[ny, nx] == -1:
                            step = 1.414 if (dx != 0 and dy != 0) else 1.0
                            dist[ny, nx] = dist[cy, cx] + int(step * 100)  # centipixels
                            parent[ny, nx] = [cx, cy]
                            queue.append((nx, ny))
                            if dist[ny, nx] > max_d:
                                max_d = dist[ny, nx]
                                farthest = (nx, ny)
            
            length = max_d / 100.0  # convertir centipixels en pixels
            if length > best_length:
                best_length = length
                # Reconstruire le chemin
                path = []
                cx, cy = farthest
                while parent[cy, cx, 0] >= 0:
                    path.append((cx, cy))
                    px, py = parent[cy, cx]
                    cx, cy = int(px), int(py)
                path.append((sx, sy))
                best_path = path[::-1]
        
        return best_length, best_path
    
    @staticmethod
    def _order_skeleton_points(points: np.ndarray) -> np.ndarray:
        """
        Ordonne les points du squelette le long de l'axe principal (PCA).
        """
        # PCA pour trouver l'axe principal
        mean = points.mean(axis=0)
        centered = points - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # L'axe principal est le dernier eigenvector (plus grande eigenvalue)
        principal_axis = eigenvectors[:, -1]
        
        # Projeter sur l'axe principal et trier
        projections = centered @ principal_axis
        order = np.argsort(projections)
        return points[order]
    
    @staticmethod
    def _measure_perpendicular_width(
        mask: np.ndarray,
        point: np.ndarray,
        direction: np.ndarray,
        max_dist: int = 500,
    ) -> float:
        """
        Mesure la largeur du masque perpendiculairement au squelette en un point.
        """
        h, w = mask.shape
        px, py = float(point[0]), float(point[1])
        dx, dy = float(direction[0]), float(direction[1])
        
        # Scanner dans la direction positive
        dist_pos = 0
        for t in range(1, max_dist):
            nx, ny = int(round(px + t * dx)), int(round(py + t * dy))
            if 0 <= nx < w and 0 <= ny < h:
                if mask[ny, nx] == 0:
                    dist_pos = t - 1
                    break
                dist_pos = t
            else:
                dist_pos = t - 1
                break
        
        # Scanner dans la direction négative
        dist_neg = 0
        for t in range(1, max_dist):
            nx, ny = int(round(px - t * dx)), int(round(py - t * dy))
            if 0 <= nx < w and 0 <= ny < h:
                if mask[ny, nx] == 0:
                    dist_neg = t - 1
                    break
                dist_neg = t
            else:
                dist_neg = t - 1
                break
        
        return float(dist_pos + dist_neg)
    
    # =========================================================================
    # Segmentation individuelle des épillets
    # =========================================================================
    
    def segment_spikelets(
        self,
        image: np.ndarray,
        spike_segmentation: Dict,
        spikelet_bboxes: List[Tuple[int, int, int, int]],
        pixel_per_mm: Optional[float] = None,
        bbox_offset: Tuple[int, int] = (0, 0),
    ) -> List[Dict]:
        """
        Segmente individuellement chaque épillet détecté par YOLO en utilisant
        SAM2 avec des box-prompts, puis calcule les métriques morphologiques
        par épillet.
        
        Args:
            image: Image BGR complète
            spike_segmentation: Résultat de segment_spike() pour l'épi parent
            spikelet_bboxes: Bboxes des épillets en coordonnées globales
                             [(x1, y1, x2, y2), ...]
            pixel_per_mm: Ratio de calibration (optionnel)
            bbox_offset: Offset (ox, oy) si bboxes sont en coordonnées ROI
            
        Returns:
            Liste de dicts, un par épillet avec métriques morphologiques
        """
        if not self.is_available() or not spikelet_bboxes:
            return []
        
        spike_mask = spike_segmentation.get('mask')
        results = []
        ox, oy = bbox_offset
        
        for idx, (bx1, by1, bx2, by2) in enumerate(spikelet_bboxes):
            # Coordonnées globales
            gx1, gy1, gx2, gy2 = bx1 + ox, by1 + oy, bx2 + ox, by2 + oy
            
            # Ajouter une marge de 5px pour le prompt SAM
            margin = 5
            h, w = image.shape[:2]
            sx1 = max(0, gx1 - margin)
            sy1 = max(0, gy1 - margin)
            sx2 = min(w, gx2 + margin)
            sy2 = min(h, gy2 + margin)
            
            spikelet_info = {
                'id': idx + 1,
                'bbox_global': (gx1, gy1, gx2, gy2),
                'center_x': (gx1 + gx2) / 2,
                'center_y': (gy1 + gy2) / 2,
            }
            
            # Segmenter cet épillet avec SAM2 box-prompt
            try:
                spikelet_mask = self.segment_from_bbox(image, (sx1, sy1, sx2, sy2))
            except Exception as e:
                logger.debug(f"  Épillet #{idx+1}: erreur SAM2: {e}")
                spikelet_mask = None
            
            if spikelet_mask is not None:
                # Restreindre au masque de l'épi parent si disponible
                if spike_mask is not None:
                    spikelet_mask = cv2.bitwise_and(spikelet_mask, spike_mask)
                
                area_px = float(cv2.countNonZero(spikelet_mask))
                
                if area_px < 50:  # trop petit, ignorer la segmentation
                    spikelet_info['segmented'] = False
                    results.append(spikelet_info)
                    continue
                
                spikelet_info['segmented'] = True
                spikelet_info['area_px'] = area_px
                
                # Trouver le contour
                contours, _ = cv2.findContours(
                    spikelet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    perimeter_px = cv2.arcLength(cnt, True)
                    spikelet_info['perimeter_px'] = float(perimeter_px)
                    
                    # Circularité
                    if perimeter_px > 0:
                        spikelet_info['circularity'] = (4 * np.pi * area_px) / (perimeter_px ** 2)
                    
                    # Convexité
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        spikelet_info['solidity'] = area_px / hull_area
                    
                    # minAreaRect pour longueur/largeur
                    rect = cv2.minAreaRect(cnt)
                    (rcx, rcy), (rw, rh), rangle = rect
                    # rw, rh : le plus grand = longueur, le plus petit = largeur
                    length_px = max(rw, rh)
                    width_px = min(rw, rh)
                    spikelet_info['length_px'] = float(length_px)
                    spikelet_info['width_px'] = float(width_px)
                    spikelet_info['angle'] = float(rangle)
                    
                    if width_px > 0:
                        spikelet_info['aspect_ratio'] = float(length_px / width_px)
                    
                    # Ellipse ajustée
                    if len(cnt) >= 5:
                        ell = cv2.fitEllipse(cnt)
                        (_, _), (ell_minor, ell_major), ell_angle = ell
                        spikelet_info['ellipse_major_px'] = float(max(ell_minor, ell_major))
                        spikelet_info['ellipse_minor_px'] = float(min(ell_minor, ell_major))
                        if min(ell_minor, ell_major) > 0:
                            spikelet_info['ellipse_eccentricity'] = float(np.sqrt(
                                1 - (min(ell_minor, ell_major) / max(ell_minor, ell_major)) ** 2
                            ))
                    
                    # Stocker le contour pour visualisation
                    spikelet_info['contour'] = cnt
                    spikelet_info['mask_roi'] = spikelet_mask[gy1:gy2, gx1:gx2]
                    
                    # Conversion en mm
                    if pixel_per_mm and pixel_per_mm > 0:
                        pp2 = pixel_per_mm ** 2
                        spikelet_info['area_mm2'] = area_px / pp2
                        spikelet_info['perimeter_mm'] = perimeter_px / pixel_per_mm
                        spikelet_info['length_mm'] = length_px / pixel_per_mm
                        spikelet_info['width_mm'] = width_px / pixel_per_mm
                        if 'ellipse_major_px' in spikelet_info:
                            spikelet_info['ellipse_major_mm'] = spikelet_info['ellipse_major_px'] / pixel_per_mm
                            spikelet_info['ellipse_minor_mm'] = spikelet_info['ellipse_minor_px'] / pixel_per_mm
                
            else:
                spikelet_info['segmented'] = False
                # Fallback : utiliser la bbox YOLO pour les mesures de base
                bw = gx2 - gx1
                bh = gy2 - gy1
                spikelet_info['length_px'] = float(max(bw, bh))
                spikelet_info['width_px'] = float(min(bw, bh))
                if min(bw, bh) > 0:
                    spikelet_info['aspect_ratio'] = float(max(bw, bh) / min(bw, bh))
                if pixel_per_mm and pixel_per_mm > 0:
                    spikelet_info['length_mm'] = max(bw, bh) / pixel_per_mm
                    spikelet_info['width_mm'] = min(bw, bh) / pixel_per_mm
            
            results.append(spikelet_info)
        
        # Statistiques agrégées sur les épillets
        return results
    
    @staticmethod
    def compute_spikelet_stats(
        spikelet_results: List[Dict],
        pixel_per_mm: Optional[float] = None,
    ) -> Dict:
        """
        Calcule des statistiques agrégées sur les épillets segmentés.
        
        Args:
            spikelet_results: Liste des résultats de segment_spikelets()
            pixel_per_mm: Ratio de calibration
            
        Returns:
            Dict avec mean/std/min/max pour longueur, largeur, aire, etc.
        """
        if not spikelet_results:
            return {}
        
        segmented = [s for s in spikelet_results if s.get('segmented', False)]
        n_total = len(spikelet_results)
        n_segmented = len(segmented)
        
        stats = {
            'n_total': n_total,
            'n_segmented': n_segmented,
            'segmentation_rate': n_segmented / n_total if n_total > 0 else 0,
        }
        
        if not segmented:
            return stats
        
        # Collecte par métrique
        for metric_key, display_key in [
            ('length_px', 'spikelet_length_px'),
            ('width_px', 'spikelet_width_px'),
            ('area_px', 'spikelet_area_px'),
            ('aspect_ratio', 'spikelet_aspect_ratio'),
            ('circularity', 'spikelet_circularity'),
            ('solidity', 'spikelet_solidity'),
        ]:
            values = [s[metric_key] for s in segmented if metric_key in s]
            if values:
                arr = np.array(values)
                stats[f'{display_key}_mean'] = float(np.mean(arr))
                stats[f'{display_key}_std'] = float(np.std(arr))
                stats[f'{display_key}_min'] = float(np.min(arr))
                stats[f'{display_key}_max'] = float(np.max(arr))
        
        # Uniformité (CV = std/mean) – indicateur de la régularité des épillets
        if stats.get('spikelet_length_px_mean', 0) > 0:
            stats['spikelet_length_cv'] = (
                stats.get('spikelet_length_px_std', 0) / stats['spikelet_length_px_mean']
            )
        if stats.get('spikelet_area_px_mean', 0) > 0:
            stats['spikelet_area_cv'] = (
                stats.get('spikelet_area_px_std', 0) / stats['spikelet_area_px_mean']
            )
        
        # Versions mm
        if pixel_per_mm and pixel_per_mm > 0:
            pp = pixel_per_mm
            pp2 = pixel_per_mm ** 2
            for suffix in ['_mean', '_std', '_min', '_max']:
                if f'spikelet_length_px{suffix}' in stats:
                    stats[f'spikelet_length_mm{suffix}'] = stats[f'spikelet_length_px{suffix}'] / pp
                if f'spikelet_width_px{suffix}' in stats:
                    stats[f'spikelet_width_mm{suffix}'] = stats[f'spikelet_width_px{suffix}'] / pp
                if f'spikelet_area_px{suffix}' in stats:
                    stats[f'spikelet_area_mm2{suffix}'] = stats[f'spikelet_area_px{suffix}'] / pp2
        
        return stats
    
    def compute_width_profile(
        self,
        segmentation_result: Dict,
        pixel_per_mm: Optional[float] = None,
        n_slices: int = 10,
    ) -> Dict:
        """
        Calcule le profil de largeur le long de l'épi (apex → base).
        
        Divise l'épi en n_slices tranches perpendiculaires à l'axe principal,
        et mesure la largeur de chaque tranche.
        
        Args:
            segmentation_result: Résultat de segment_spike()
            pixel_per_mm: Ratio de calibration
            n_slices: Nombre de tranches
            
        Returns:
            Dict avec les largeurs par tranche et les stats par tiers
        """
        mask = segmentation_result['mask']
        contour = segmentation_result['contour']
        
        if len(contour) < 5:
            return {}
        
        # Ajuster une ellipse pour l'orientation
        ellipse = cv2.fitEllipse(contour)
        (ecx, ecy), (minor_axis, major_axis), angle_deg = ellipse
        
        # Rotation du masque pour aligner l'épi verticalement
        h, w = mask.shape[:2]
        rotation_angle = 90 - angle_deg if angle_deg > 90 else -angle_deg
        
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rotation_angle, 1.0)
        
        # Calculer la nouvelle taille pour ne rien couper
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        
        mask_rotated = cv2.warpAffine(
            mask, M, (new_w, new_h),
            flags=cv2.INTER_NEAREST,
        )
        
        # Trouver la bounding box du masque aligné
        ys, xs = np.where(mask_rotated > 0)
        if len(ys) == 0:
            return {}
        
        y_min, y_max = ys.min(), ys.max()
        total_height = y_max - y_min + 1
        
        if total_height < n_slices:
            return {}
        
        # Mesurer la largeur à chaque tranche
        slice_height = total_height / n_slices
        widths_px = []
        
        for i in range(n_slices):
            slice_y_start = int(y_min + i * slice_height)
            slice_y_end = int(y_min + (i + 1) * slice_height)
            
            # Largeur = nombre de pixels non-nuls par ligne, prendre le max
            slice_region = mask_rotated[slice_y_start:slice_y_end, :]
            if slice_region.size == 0:
                widths_px.append(0)
                continue
            
            row_widths = []
            for row in range(slice_region.shape[0]):
                nonzero = np.where(slice_region[row, :] > 0)[0]
                if len(nonzero) > 0:
                    row_widths.append(nonzero[-1] - nonzero[0] + 1)
            
            widths_px.append(np.mean(row_widths) if row_widths else 0)
        
        widths_px = np.array(widths_px)
        
        # Stats par tiers (apical, médian, basal)
        third = n_slices // 3
        remainder = n_slices - 3 * third
        
        apical_widths = widths_px[:third]
        medial_widths = widths_px[third:2 * third + remainder]
        basal_widths = widths_px[2 * third + remainder:]
        
        result = {
            'widths_px': widths_px.tolist(),
            'n_slices': n_slices,
            'apical_width_px': float(np.mean(apical_widths)) if len(apical_widths) > 0 else 0,
            'medial_width_px': float(np.mean(medial_widths)) if len(medial_widths) > 0 else 0,
            'basal_width_px': float(np.mean(basal_widths)) if len(basal_widths) > 0 else 0,
            'max_width_px': float(np.max(widths_px)) if len(widths_px) > 0 else 0,
            'max_width_position': float(np.argmax(widths_px) / n_slices) if len(widths_px) > 0 else 0.5,
        }
        
        if pixel_per_mm and pixel_per_mm > 0:
            result['widths_mm'] = (widths_px / pixel_per_mm).tolist()
            result['apical_width_mm'] = result['apical_width_px'] / pixel_per_mm
            result['medial_width_mm'] = result['medial_width_px'] / pixel_per_mm
            result['basal_width_mm'] = result['basal_width_px'] / pixel_per_mm
            result['max_width_mm'] = result['max_width_px'] / pixel_per_mm
        
        # Classification de forme basée sur le profil
        if result['medial_width_px'] > 0:
            apical_ratio = result['apical_width_px'] / result['medial_width_px']
            basal_ratio = result['basal_width_px'] / result['medial_width_px']
            
            if apical_ratio < 0.7 and basal_ratio < 0.7:
                shape_class = 'fusiforme'     # étroit aux deux extrémités
            elif apical_ratio < 0.7 and basal_ratio >= 0.7:
                shape_class = 'claviforme'    # large à la base, étroit au sommet
            elif apical_ratio >= 0.7 and basal_ratio >= 0.7:
                shape_class = 'parallele'     # largeur uniforme
            else:
                shape_class = 'obovale'       # large au sommet, étroit à la base
            
            result['shape_class'] = shape_class
            result['apical_medial_ratio'] = apical_ratio
            result['basal_medial_ratio'] = basal_ratio
        
        return result
    
    def compute_color_stats(
        self,
        image: np.ndarray,
        segmentation_result: Dict,
    ) -> Dict:
        """
        Calcule les statistiques de couleur de l'épi à partir du masque.
        
        Args:
            image: Image BGR complète
            segmentation_result: Résultat de segment_spike()
            
        Returns:
            Dict avec stats HSV et RGB moyennes
        """
        mask = segmentation_result['mask']
        
        # Convertir en HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Appliquer le masque
        mask_bool = mask > 0
        
        if not np.any(mask_bool):
            return {}
        
        # Stats HSV
        h_vals = hsv[:, :, 0][mask_bool].astype(float)
        s_vals = hsv[:, :, 1][mask_bool].astype(float)
        v_vals = hsv[:, :, 2][mask_bool].astype(float)
        
        # Stats BGR
        b_vals = image[:, :, 0][mask_bool].astype(float)
        g_vals = image[:, :, 1][mask_bool].astype(float)
        r_vals = image[:, :, 2][mask_bool].astype(float)
        
        return {
            # HSV
            'hue_mean': float(np.mean(h_vals)),
            'hue_std': float(np.std(h_vals)),
            'saturation_mean': float(np.mean(s_vals)),
            'saturation_std': float(np.std(s_vals)),
            'value_mean': float(np.mean(v_vals)),
            'value_std': float(np.std(v_vals)),
            # RGB
            'red_mean': float(np.mean(r_vals)),
            'green_mean': float(np.mean(g_vals)),
            'blue_mean': float(np.mean(b_vals)),
            # Indices dérivés
            'greenness_index': float(np.mean(g_vals) / (np.mean(r_vals) + 1e-6)),
            'yellowing_index': float(
                np.mean(r_vals + g_vals) / (2 * np.mean(b_vals) + 1e-6)
            ),
        }
    
    def draw_segmentation(
        self,
        image: np.ndarray,
        segmentation_result: Dict,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.3,
    ) -> np.ndarray:
        """
        Dessine la segmentation sur l'image.
        
        Args:
            image: Image BGR
            segmentation_result: Résultat de segment_spike()
            color: Couleur du masque superposé (BGR)
            alpha: Transparence du masque
            
        Returns:
            Image avec superposition du masque
        """
        viz = image.copy()
        mask = segmentation_result['mask']
        contour = segmentation_result['contour']
        
        # Superposition semi-transparente
        overlay = viz.copy()
        overlay[mask > 0] = color
        cv2.addWeighted(overlay, alpha, viz, 1 - alpha, 0, viz)
        
        # Contour
        cv2.drawContours(viz, [contour], -1, color, 2)
        
        return viz
