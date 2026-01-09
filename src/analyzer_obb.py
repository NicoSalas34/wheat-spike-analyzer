#!/usr/bin/env python3
"""
Analyseur d'épis de blé utilisant YOLO OBB (Oriented Bounding Boxes)

Pipeline:
1. Détection YOLO OBB : règle, épis (spike), épis avec barbes (whole_spike), sachets
2. Calibration : calcul du ratio px/mm depuis l'OBB de la règle
3. Mesures morphométriques : longueur, largeur, aire, périmètre depuis les OBB
4. Comptage des épillets : modèle YOLO dédié
5. Identification du sachet : OCR des chiffres (bac-ligne-colonne)
6. Export : JSON + images annotées
"""

import cv2
import numpy as np
import logging
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OBBDetection:
    """Représente une détection avec bounding box orientée"""
    class_id: int
    class_name: str
    confidence: float
    obb_points: np.ndarray  # 4x2 points du polygone orienté
    center: Tuple[float, float]
    width: float   # Dimension la plus petite
    height: float  # Dimension la plus grande (longueur)
    angle: float   # Angle en degrés
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Bounding box axis-aligned englobante"""
        xs = self.obb_points[:, 0]
        ys = self.obb_points[:, 1]
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    
    @property
    def area(self) -> float:
        """Aire du rectangle orienté"""
        return self.width * self.height
    
    @property
    def perimeter(self) -> float:
        """Périmètre du rectangle orienté"""
        return 2 * (self.width + self.height)
    
    @property
    def aspect_ratio(self) -> float:
        """Ratio longueur/largeur"""
        return self.height / self.width if self.width > 0 else 0


# Mapping des classes
CLASS_NAMES = {
    0: 'ruler',
    1: 'spike',
    2: 'bag',
    3: 'whole_spike',
}


class WheatSpikeAnalyzerOBB:
    """
    Analyseur d'épis de blé utilisant YOLO OBB
    
    Pipeline complet:
    1. Détection OBB (règle, épis, sachets)
    2. Calibration (px/mm)
    3. Mesures morphométriques
    4. Comptage épillets
    5. Identification sachet (OCR)
    6. Export résultats
    """
    
    def __init__(self, config: Dict, output_dir: str = "output", debug: bool = True):
        """
        Initialise l'analyseur
        
        Args:
            config: Configuration YAML chargée
            output_dir: Dossier de sortie
            debug: Activer les images de debug
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.debug = debug
        
        # Charger le modèle YOLO OBB principal
        self.detector = self._load_yolo_model()
        
        # Charger les modèles auxiliaires
        self.spikelet_counter = self._load_spikelet_counter()
        self.bag_digit_detector = self._load_bag_digit_detector()
        
        # Ratio de calibration (calculé à chaque image)
        self.pixel_per_mm: Optional[float] = None
        
        logger.info("✓ WheatSpikeAnalyzerOBB initialisé")
    
    def _load_yolo_model(self):
        """Charge le modèle YOLO OBB principal"""
        from ultralytics import YOLO
        
        model_path = self.config.get('yolo', {}).get('model_path', 
            'runs/train_obb_angled/test_run/weights/best.pt')
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Modèle YOLO OBB non trouvé: {model_path}")
        
        logger.info(f"Chargement modèle YOLO OBB: {model_path}")
        model = YOLO(model_path)
        logger.info(f"✓ Modèle chargé - Classes: {model.names}")
        
        return model
    
    def _load_spikelet_counter(self):
        """Charge le compteur d'épillets YOLO"""
        spikelet_config = self.config.get('spikelet_counting', {})
        
        if not spikelet_config.get('enabled', True):
            return None
        
        model_path = spikelet_config.get('yolo', {}).get('model_path', 
            'models/spikelets_yolo.pt')
        
        if not Path(model_path).exists():
            logger.warning(f"Modèle épillets non trouvé: {model_path}")
            return None
        
        try:
            from .spikelet_counter import SpikeletCounter
        except ImportError:
            from spikelet_counter import SpikeletCounter
        
        counter = SpikeletCounter(
            model_path=model_path,
            confidence=spikelet_config.get('yolo', {}).get('confidence_threshold', 0.25)
        )
        counter.load_model()
        logger.info(f"✓ Compteur épillets chargé: {model_path}")
        
        return counter
    
    def _load_bag_digit_detector(self):
        """Charge le détecteur de chiffres sur sachets"""
        bag_config = self.config.get('bag_digits', {})
        
        if not bag_config.get('enabled', True):
            return None
        
        model_path = bag_config.get('model_path', 'models/bag_digits_yolo.pt')
        
        if not Path(model_path).exists():
            logger.warning(f"Modèle bag_digits non trouvé: {model_path}")
            return None
        
        try:
            from .bag_digit_detector import BagDigitDetector
        except ImportError:
            from bag_digit_detector import BagDigitDetector
        
        detector = BagDigitDetector(
            model_path=model_path,
            confidence_threshold=bag_config.get('confidence_threshold', 0.35),
            opening_model_path=bag_config.get('opening_model_path', 'models/bag_opening_yolo.pt'),
            opening_confidence_threshold=bag_config.get('opening_confidence_threshold', 0.3)
        )
        logger.info(f"✓ Détecteur bag_digits chargé")
        
        return detector
    
    # =========================================================================
    # ÉTAPE 1: DÉTECTION YOLO OBB
    # =========================================================================
    
    def detect_objects_obb(self, image: np.ndarray) -> Dict[str, List[OBBDetection]]:
        """
        Détecte tous les objets avec YOLO OBB
        
        Args:
            image: Image BGR
            
        Returns:
            Dict avec clés 'ruler', 'spikes', 'whole_spikes', 'bags'
        """
        yolo_config = self.config.get('yolo', {})
        conf_threshold = yolo_config.get('confidence_threshold', 0.35)
        iou_threshold = yolo_config.get('iou_threshold', 0.45)
        
        # Inférence YOLO OBB
        results = self.detector.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
            half=False,  # Compatibilité ROCm
        )
        
        detections = {
            'ruler': [],
            'spikes': [],
            'whole_spikes': [],
            'bags': [],
        }
        
        for result in results:
            # Vérifier si c'est un résultat OBB
            if hasattr(result, 'obb') and result.obb is not None:
                obb_data = result.obb.data
                
                for i in range(len(obb_data)):
                    row = obb_data[i].cpu().numpy()
                    # Format OBB: [cx, cy, w, h, angle_rad, conf, cls]
                    cx, cy, w, h, angle_rad, confidence, class_id = row
                    class_id = int(class_id)
                    
                    # Convertir angle en degrés
                    angle_deg = float(angle_rad) * 180.0 / np.pi
                    
                    # Calculer les 4 coins du rectangle orienté
                    rect = ((float(cx), float(cy)), (float(w), float(h)), angle_deg)
                    box_pts = cv2.boxPoints(rect).astype(np.float32)
                    
                    # Assurer que height > width (height = longueur)
                    length = max(w, h)
                    width = min(w, h)
                    
                    det = OBBDetection(
                        class_id=class_id,
                        class_name=CLASS_NAMES.get(class_id, f"class_{class_id}"),
                        confidence=float(confidence),
                        obb_points=box_pts,
                        center=(float(cx), float(cy)),
                        width=float(width),
                        height=float(length),
                        angle=angle_deg,
                    )
                    
                    # Classer par type
                    if class_id == 0:
                        detections['ruler'].append(det)
                    elif class_id == 1:
                        detections['spikes'].append(det)
                    elif class_id == 2:
                        detections['bags'].append(det)
                    elif class_id == 3:
                        detections['whole_spikes'].append(det)
            
            # Fallback pour résultats non-OBB (boxes standard)
            elif hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    
                    w = x2 - x1
                    h = y2 - y1
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    
                    # Créer un rectangle "aligné" comme si c'était un OBB à angle 0
                    box_pts = np.array([
                        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                    ], dtype=np.float32)
                    
                    det = OBBDetection(
                        class_id=class_id,
                        class_name=CLASS_NAMES.get(class_id, f"class_{class_id}"),
                        confidence=float(confidence),
                        obb_points=box_pts,
                        center=(cx, cy),
                        width=float(min(w, h)),
                        height=float(max(w, h)),
                        angle=0.0,
                    )
                    
                    if class_id == 0:
                        detections['ruler'].append(det)
                    elif class_id == 1:
                        detections['spikes'].append(det)
                    elif class_id == 2:
                        detections['bags'].append(det)
                    elif class_id == 3:
                        detections['whole_spikes'].append(det)
        
        # Log des détections
        logger.info(f"Détections OBB: "
                   f"ruler={len(detections['ruler'])}, "
                   f"spikes={len(detections['spikes'])}, "
                   f"whole_spikes={len(detections['whole_spikes'])}, "
                   f"bags={len(detections['bags'])}")
        
        return detections
    
    # =========================================================================
    # ÉTAPE 2: CALIBRATION (ratio px/mm)
    # =========================================================================
    
    def calibrate_from_ruler(self, ruler_det: OBBDetection) -> Optional[float]:
        """
        Calcule le ratio pixels/mm depuis l'OBB de la règle
        
        Args:
            ruler_det: Détection OBB de la règle
            
        Returns:
            Ratio pixels par millimètre ou None si échec
        """
        ruler_length_mm = self.config.get('ruler_detection', {}).get('ruler_length_mm', 300)
        
        # La longueur de la règle est la dimension la plus grande de l'OBB
        ruler_length_px = ruler_det.height  # height = dimension la plus grande
        
        if ruler_length_px < 100:
            logger.warning(f"Règle trop petite: {ruler_length_px}px")
            return None
        
        pixel_per_mm = ruler_length_px / ruler_length_mm
        
        logger.info(f"Calibration: {ruler_length_px:.1f}px / {ruler_length_mm}mm = "
                   f"{pixel_per_mm:.3f} px/mm")
        
        return pixel_per_mm
    
    # =========================================================================
    # ÉTAPE 3: MESURES MORPHOMÉTRIQUES
    # =========================================================================
    
    def measure_spike_pair(self, whole_spike_det: Optional[OBBDetection], 
                            spike_det: Optional[OBBDetection]) -> Dict:
        """
        Calcule les mesures morphométriques d'un épi depuis les OBB
        
        Si whole_spike ET spike sont disponibles:
        - Longueur spike = longueur de la bbox spike
        - Longueur barbes = longueur whole_spike - longueur spike
        
        Args:
            whole_spike_det: Détection OBB du whole_spike (épi + barbes)
            spike_det: Détection OBB du spike seul (épi sans barbes)
            
        Returns:
            Dict avec toutes les mesures
        """
        # Déterminer quelle détection utiliser pour les mesures principales
        # Priorité au spike (mesure plus précise de l'épi seul)
        primary_det = spike_det if spike_det else whole_spike_det
        
        if primary_det is None:
            return {}
        
        measurements = {
            # Type de détection
            'detection_type': primary_det.class_name,
            'has_whole_spike': whole_spike_det is not None,
            'has_spike': spike_det is not None,
            
            # Mesures en pixels de l'épi (spike)
            'spike_length_pixels': spike_det.height if spike_det else None,
            'spike_width_pixels': spike_det.width if spike_det else None,
            
            # Mesures en pixels du whole_spike
            'whole_spike_length_pixels': whole_spike_det.height if whole_spike_det else None,
            'whole_spike_width_pixels': whole_spike_det.width if whole_spike_det else None,
            
            # Mesures principales (depuis spike si dispo, sinon whole_spike)
            'length_pixels': primary_det.height,
            'width_pixels': primary_det.width,
            'area_pixels': primary_det.area,
            'perimeter_pixels': primary_det.perimeter,
            'aspect_ratio': primary_det.aspect_ratio,
            'angle_degrees': primary_det.angle,
            'center_x': primary_det.center[0],
            'center_y': primary_det.center[1],
            'confidence': primary_det.confidence,
            'measurement_source': 'obb',
        }
        
        # Conversion en mm si calibré
        if self.pixel_per_mm:
            if spike_det:
                measurements['spike_length_mm'] = spike_det.height / self.pixel_per_mm
                measurements['spike_width_mm'] = spike_det.width / self.pixel_per_mm
            if whole_spike_det:
                measurements['whole_spike_length_mm'] = whole_spike_det.height / self.pixel_per_mm
                measurements['whole_spike_width_mm'] = whole_spike_det.width / self.pixel_per_mm
            
            measurements['length_mm'] = primary_det.height / self.pixel_per_mm
            measurements['width_mm'] = primary_det.width / self.pixel_per_mm
            measurements['area_mm2'] = primary_det.area / (self.pixel_per_mm ** 2)
            measurements['perimeter_mm'] = primary_det.perimeter / self.pixel_per_mm
        
        # Calcul des barbes (awns) = whole_spike - spike
        measurements['has_awns'] = (whole_spike_det is not None and spike_det is not None)
        
        if whole_spike_det and spike_det:
            awns_length_px = whole_spike_det.height - spike_det.height
            measurements['awns_length_pixels'] = awns_length_px
            
            if self.pixel_per_mm:
                measurements['awns_length_mm'] = awns_length_px / self.pixel_per_mm
            
            logger.debug(f"  Barbes: {awns_length_px:.1f}px = "
                        f"{whole_spike_det.height:.1f} - {spike_det.height:.1f}")
        
        return measurements
    
    # =========================================================================
    # ÉTAPE 4: COMPTAGE DES ÉPILLETS
    # =========================================================================
    
    def count_spikelets(self, image: np.ndarray, spike_det: OBBDetection) -> Dict:
        """
        Compte les épillets dans un épi
        
        Args:
            image: Image complète
            spike_det: Détection OBB de l'épi
            
        Returns:
            Dict avec count, positions, confidence
        """
        if self.spikelet_counter is None:
            return {'count': None, 'method': 'unavailable', 'positions': []}
        
        # Extraire la ROI de l'épi
        x1, y1, x2, y2 = spike_det.bbox
        margin = 10
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(image.shape[1], x2 + margin)
        y2 = min(image.shape[0], y2 + margin)
        
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return {'count': None, 'method': 'error', 'positions': []}
        
        # Compter avec YOLO
        count, bboxes = self.spikelet_counter.count_yolo(roi)
        
        # Convertir les positions en coordonnées globales
        positions = []
        for bx1, by1, bx2, by2 in bboxes:
            cx = x1 + (bx1 + bx2) / 2
            cy = y1 + (by1 + by2) / 2
            positions.append((cx, cy))
        
        confidence = 'high' if count >= 10 else ('medium' if count >= 5 else 'low')
        
        return {
            'count': count,
            'method': 'yolo',
            'confidence': confidence,
            'positions': positions,
            'bboxes': [(x1 + bx1, y1 + by1, x1 + bx2, y1 + by2) for bx1, by1, bx2, by2 in bboxes],
        }
    
    # =========================================================================
    # ÉTAPE 5: IDENTIFICATION DU SACHET (OCR)
    # =========================================================================
    
    def identify_bag(self, image: np.ndarray, bag_det: OBBDetection) -> Dict:
        """
        Identifie le sachet via OCR des chiffres manuscrits
        
        Args:
            image: Image complète
            bag_det: Détection OBB du sachet
            
        Returns:
            Dict avec sample_id, bac, ligne, colonne
        """
        if self.bag_digit_detector is None or not self.bag_digit_detector.is_available():
            return {
                'detected': True,
                'sample_id': None,
                'bac': None,
                'ligne': None,
                'colonne': None,
                'confidence': 0,
                'complete': False,
            }
        
        # Utiliser la bbox axis-aligned pour l'extraction
        bbox = bag_det.bbox
        
        try:
            sample_info = self.bag_digit_detector.detect_from_full_image(image, bbox)
            
            return {
                'detected': True,
                'sample_id': sample_info.get('sample_id'),
                'bac': sample_info.get('bac'),
                'ligne': sample_info.get('ligne'),
                'colonne': sample_info.get('colonne'),
                'confidence': sample_info.get('confidence', 0),
                'complete': sample_info.get('complete', False),
                'detections': sample_info.get('detections', []),
                'orientation': sample_info.get('orientation', {}),
            }
        except Exception as e:
            logger.warning(f"Erreur OCR sachet: {e}")
            return {
                'detected': True,
                'sample_id': None,
                'error': str(e),
            }
    
    # =========================================================================
    # IMAGES DE DEBUG
    # =========================================================================
    
    def _draw_obb(self, image: np.ndarray, det: OBBDetection, 
                  color: Tuple[int, int, int], thickness: int = 2,
                  label: str = None, fill: bool = False) -> np.ndarray:
        """Dessine un OBB sur l'image"""
        pts = det.obb_points.astype(np.int32)
        
        # Optionnellement remplir avec transparence
        if fill:
            overlay = image.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Dessiner le contour
        cv2.polylines(image, [pts], True, color, thickness)
        
        if label:
            # Position du label au-dessus de l'OBB
            y_min = int(pts[:, 1].min())
            x_center = int(det.center[0])
            
            # Fond pour le texte
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, 
                         (x_center - text_w//2 - 2, y_min - text_h - 8),
                         (x_center + text_w//2 + 2, y_min - 2),
                         color, -1)
            cv2.putText(image, label, (x_center - text_w//2, y_min - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def save_debug_01_detections(self, image: np.ndarray, 
                                  detections: Dict[str, List[OBBDetection]],
                                  session_dir: Path) -> None:
        """Sauvegarde l'image des détections OBB avec toutes les boîtes"""
        debug_dir = session_dir / 'debug'
        debug_dir.mkdir(exist_ok=True)
        
        viz = image.copy()
        colors = {
            'ruler': (0, 0, 255),      # Rouge
            'spikes': (0, 255, 0),     # Vert
            'whole_spikes': (0, 255, 255),  # Jaune
            'bags': (255, 0, 0),       # Bleu
        }
        
        # Compter le total des détections
        total_dets = sum(len(dets) for dets in detections.values())
        
        # Dessiner chaque type de détection
        for det_type, dets in detections.items():
            color = colors.get(det_type, (128, 128, 128))
            for i, det in enumerate(dets):
                # Label avec dimensions
                label = f"{det.class_name} {det.confidence:.0%}"
                self._draw_obb(viz, det, color, 3, label)
                
                # Ajouter les dimensions dans l'OBB
                cx, cy = int(det.center[0]), int(det.center[1])
                dim_text = f"{det.height:.0f}x{det.width:.0f}px"
                cv2.putText(viz, dim_text, (cx - 40, cy + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                cv2.putText(viz, dim_text, (cx - 40, cy + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Ajouter un résumé en haut de l'image
        summary = f"Detections: ruler={len(detections['ruler'])}, "
        summary += f"spikes={len(detections['spikes'])}, "
        summary += f"whole_spikes={len(detections['whole_spikes'])}, "
        summary += f"bags={len(detections['bags'])}"
        cv2.putText(viz, summary, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(viz, summary, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        cv2.imwrite(str(debug_dir / '01_detections_obb.png'), viz)
        logger.debug(f"  [DEBUG] 01_detections_obb.png sauvegardé ({total_dets} détections)")
    
    def save_debug_02_calibration(self, image: np.ndarray,
                                   ruler_det: Optional[OBBDetection],
                                   pixel_per_mm: Optional[float],
                                   session_dir: Path) -> None:
        """Sauvegarde l'image de calibration"""
        debug_dir = session_dir / 'debug'
        debug_dir.mkdir(exist_ok=True)
        
        viz = image.copy()
        
        if ruler_det:
            # Dessiner l'OBB de la règle
            self._draw_obb(viz, ruler_det, (0, 0, 255), 3)
            
            # Ajouter les informations de calibration
            info_text = []
            info_text.append(f"Longueur regle: {ruler_det.height:.1f} px")
            if pixel_per_mm:
                info_text.append(f"Calibration: {pixel_per_mm:.3f} px/mm")
                info_text.append(f"1 cm = {pixel_per_mm * 10:.1f} px")
            
            y_offset = 50
            for text in info_text:
                cv2.putText(viz, text, (50, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                cv2.putText(viz, text, (50, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                y_offset += 40
        else:
            cv2.putText(viz, "REGLE NON DETECTEE", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        cv2.imwrite(str(debug_dir / '02_ruler_calibration.png'), viz)
        logger.debug("  [DEBUG] 02_ruler_calibration.png sauvegardé")
    
    def save_debug_03_spikes(self, image: np.ndarray,
                              spike_results: List[Dict],
                              session_dir: Path) -> None:
        """Sauvegarde l'image des épis avec mesures et OBB distincts"""
        debug_dir = session_dir / 'debug'
        debug_dir.mkdir(exist_ok=True)
        
        viz = image.copy()
        
        for result in spike_results:
            ws_det = result.get('whole_spike_det')
            sp_det = result.get('spike_det')
            measurements = result.get('measurements', {})
            spikelets = result.get('spikelets', {})
            
            # Dessiner le whole_spike en jaune (si présent)
            if ws_det:
                self._draw_obb(viz, ws_det, (0, 255, 255), 2)  # Jaune
            
            # Dessiner le spike en vert (si présent)
            if sp_det:
                self._draw_obb(viz, sp_det, (0, 255, 0), 3)  # Vert, plus épais
            
            # Détection principale pour les annotations
            primary_det = sp_det if sp_det else ws_det
            if primary_det is None:
                continue
            
            cx, cy = int(primary_det.center[0]), int(primary_det.center[1])
            
            # Construire les lignes d'annotation
            lines = [f"#{result['id']}"]
            
            # Longueur du spike
            if measurements.get('spike_length_mm'):
                lines.append(f"Spike: {measurements['spike_length_mm']:.1f}mm")
            elif measurements.get('spike_length_pixels'):
                lines.append(f"Spike: {measurements['spike_length_pixels']:.0f}px")
            elif measurements.get('length_mm'):
                lines.append(f"L: {measurements['length_mm']:.1f}mm")
            
            # Longueur des barbes si disponible
            if measurements.get('has_awns'):
                if measurements.get('awns_length_mm'):
                    lines.append(f"Barbes: {measurements['awns_length_mm']:.1f}mm")
                elif measurements.get('awns_length_pixels'):
                    lines.append(f"Barbes: {measurements['awns_length_pixels']:.0f}px")
            
            # Largeur
            if measurements.get('width_mm'):
                lines.append(f"W: {measurements['width_mm']:.1f}mm")
            elif measurements.get('width_pixels'):
                lines.append(f"W: {measurements['width_pixels']:.0f}px")
            
            # Épillets
            spikelet_count = spikelets.get('count')
            if spikelet_count:
                lines.append(f"Ep: {spikelet_count}")
            
            # Dessiner les annotations
            y_offset = cy - 30
            for line in lines:
                cv2.putText(viz, line, (cx - 40, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(viz, line, (cx - 40, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                y_offset += 18
            
            # Dessiner les positions des épillets
            for pos in spikelets.get('positions', []):
                px, py = int(pos[0]), int(pos[1])
                cv2.circle(viz, (px, py), 5, (255, 0, 255), -1)
        
        cv2.imwrite(str(debug_dir / '03_spikes_analysis.png'), viz)
        logger.debug("  [DEBUG] 03_spikes_analysis.png sauvegardé")
    
    def save_debug_04_bag(self, image: np.ndarray,
                           bag_det: Optional[OBBDetection],
                           bag_info: Dict,
                           session_dir: Path) -> None:
        """Sauvegarde l'image d'identification du sachet"""
        debug_dir = session_dir / 'debug'
        debug_dir.mkdir(exist_ok=True)
        
        viz = image.copy()
        
        if bag_det:
            self._draw_obb(viz, bag_det, (255, 0, 0), 3)
            
            # Afficher l'ID de l'échantillon
            sample_id = bag_info.get('sample_id')
            cx, cy = int(bag_det.center[0]), int(bag_det.center[1])
            
            if sample_id:
                text = f"ID: {sample_id}"
                cv2.putText(viz, text, (cx - 50, cy - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            # Afficher bac/ligne/colonne
            bac = bag_info.get('bac', '?')
            ligne = bag_info.get('ligne', '?')
            colonne = bag_info.get('colonne', '?')
            
            info_text = f"Bac:{bac} Ligne:{ligne} Col:{colonne}"
            cv2.putText(viz, info_text, (cx - 80, cy + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Dessiner les détections de chiffres
            for det_info in bag_info.get('detections', []):
                bbox = det_info.get('bbox')
                value = det_info.get('value')
                if bbox and value:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(viz, str(value), (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imwrite(str(debug_dir / '04_bag_identification.png'), viz)
        logger.debug("  [DEBUG] 04_bag_identification.png sauvegardé")
    
    def save_debug_05_final(self, image: np.ndarray,
                             results: Dict,
                             spike_results: List[Dict],
                             detections: Dict[str, List[OBBDetection]],
                             session_dir: Path) -> None:
        """Sauvegarde l'image finale annotée avec toutes les OBB"""
        debug_dir = session_dir / 'debug'
        debug_dir.mkdir(exist_ok=True)
        
        viz = image.copy()
        h, w = viz.shape[:2]
        
        # 1. Dessiner la règle
        for ruler in detections.get('ruler', []):
            self._draw_obb(viz, ruler, (0, 0, 255), 2)
        
        # 2. Dessiner tous les épis (whole_spike en jaune, spike en vert)
        for result in spike_results:
            ws_det = result.get('whole_spike_det')
            sp_det = result.get('spike_det')
            measurements = result.get('measurements', {})
            
            if ws_det:
                self._draw_obb(viz, ws_det, (0, 255, 255), 2)  # Jaune
            if sp_det:
                self._draw_obb(viz, sp_det, (0, 255, 0), 3)  # Vert
            
            # Annotation
            primary_det = sp_det if sp_det else ws_det
            if primary_det:
                cx, cy = int(primary_det.center[0]), int(primary_det.center[1])
                
                # Texte récapitulatif
                parts = [f"#{result['id']}"]
                if measurements.get('spike_length_mm'):
                    parts.append(f"L:{measurements['spike_length_mm']:.1f}mm")
                elif measurements.get('length_mm'):
                    parts.append(f"L:{measurements['length_mm']:.1f}mm")
                
                if measurements.get('awns_length_mm'):
                    parts.append(f"Barbes:{measurements['awns_length_mm']:.1f}mm")
                
                spikelet_count = result.get('spikelets', {}).get('count')
                if spikelet_count:
                    parts.append(f"Ep:{spikelet_count}")
                
                text = " ".join(parts)
                cv2.putText(viz, text, (cx - 60, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(viz, text, (cx - 60, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 3. Dessiner le sachet
        for bag in detections.get('bags', []):
            self._draw_obb(viz, bag, (255, 0, 0), 2)
        
        # 4. Ajouter les infos en superposition
        # Calibration
        if results.get('calibration', {}).get('ruler_detected'):
            cv2.putText(viz, f"Calibration: {results['calibration']['pixel_per_mm']:.2f} px/mm",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            cv2.putText(viz, f"Calibration: {results['calibration']['pixel_per_mm']:.2f} px/mm",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Identification sachet
        bag_info = results.get('bag', {})
        sample_id = bag_info.get('sample_id')
        if sample_id:
            cv2.putText(viz, f"Echantillon: {sample_id}",
                       (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(viz, f"Echantillon: {sample_id}",
                       (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # Légende
        legend_y = h - 80
        cv2.putText(viz, "Legende: ", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(viz, (100, legend_y - 12), (120, legend_y + 2), (0, 255, 0), -1)
        cv2.putText(viz, "spike", (125, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(viz, (180, legend_y - 12), (200, legend_y + 2), (0, 255, 255), -1)
        cv2.putText(viz, "whole_spike", (205, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imwrite(str(debug_dir / '05_final_result.png'), viz)
        cv2.imwrite(str(session_dir / 'result_annotated.png'), viz)
        logger.debug("  [DEBUG] 05_final_result.png sauvegardé")
    
    # =========================================================================
    # PIPELINE PRINCIPAL
    # =========================================================================
    
    def analyze_image(self, image_path: str) -> Optional[Dict]:
        """
        Analyse une image d'épis de blé
        
        Pipeline:
        1. Détection YOLO OBB
        2. Calibration (règle)
        3. Mesures morphométriques
        4. Comptage épillets
        5. Identification sachet
        6. Export
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Dict avec tous les résultats ou None si erreur
        """
        try:
            logger.info(f"{'='*60}")
            logger.info(f"Analyse: {image_path}")
            logger.info(f"{'='*60}")
            
            # Charger l'image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Impossible de charger: {image_path}")
                return None
            
            logger.info(f"Image: {image.shape[1]}x{image.shape[0]} pixels")
            
            # Créer le dossier de session
            session_name = Path(image_path).stem
            session_dir = self.output_dir / session_name
            session_dir.mkdir(parents=True, exist_ok=True)
            
            # =====================================================================
            # ÉTAPE 1: DÉTECTION YOLO OBB
            # =====================================================================
            logger.info("[1/5] Détection YOLO OBB...")
            detections = self.detect_objects_obb(image)
            
            if self.debug:
                self.save_debug_01_detections(image, detections, session_dir)
            
            # =====================================================================
            # ÉTAPE 2: CALIBRATION
            # =====================================================================
            logger.info("[2/5] Calibration...")
            ruler_det = None
            self.pixel_per_mm = None
            
            if detections['ruler']:
                # Prendre la règle avec la meilleure confiance
                ruler_det = max(detections['ruler'], key=lambda d: d.confidence)
                self.pixel_per_mm = self.calibrate_from_ruler(ruler_det)
            else:
                logger.warning("Aucune règle détectée - mesures en pixels uniquement")
            
            if self.debug:
                self.save_debug_02_calibration(image, ruler_det, self.pixel_per_mm, session_dir)
            
            # =====================================================================
            # ÉTAPE 3: MESURES MORPHOMÉTRIQUES DES ÉPIS
            # =====================================================================
            logger.info("[3/5] Mesures morphométriques...")
            
            # Combiner spikes et whole_spikes, en appariant si possible
            all_spike_dets = []
            
            # D'abord traiter les whole_spikes
            for ws_det in detections['whole_spikes']:
                # Chercher un spike associé (contenu dans le whole_spike)
                associated_spike = None
                for sp_det in detections['spikes']:
                    # Vérifier si le spike est à l'intérieur du whole_spike
                    sp_cx, sp_cy = sp_det.center
                    ws_x1, ws_y1, ws_x2, ws_y2 = ws_det.bbox
                    if ws_x1 <= sp_cx <= ws_x2 and ws_y1 <= sp_cy <= ws_y2:
                        associated_spike = sp_det
                        break
                
                all_spike_dets.append((ws_det, associated_spike))
            
            # Ajouter les spikes non associés (ceux qui n'ont pas de whole_spike)
            used_spikes = {id(sp) for ws, sp in all_spike_dets if sp is not None}
            for sp_det in detections['spikes']:
                if id(sp_det) not in used_spikes:
                    all_spike_dets.append((None, sp_det))  # (whole_spike=None, spike)
            
            spike_results = []
            for idx, (ws_det, sp_det) in enumerate(all_spike_dets):
                # Mesurer avec le nouveau système qui calcule les barbes
                measurements = self.measure_spike_pair(ws_det, sp_det)
                
                # Stocker les deux détections pour le debug
                primary_det = sp_det if sp_det else ws_det
                
                spike_result = {
                    'id': idx + 1,
                    'detection': primary_det,
                    'whole_spike_det': ws_det,
                    'spike_det': sp_det,
                    'measurements': measurements,
                    'spikelets': {},
                }
                spike_results.append(spike_result)
                
                # Log des mesures
                if measurements.get('has_awns'):
                    awns = measurements.get('awns_length_mm') or measurements.get('awns_length_pixels', 0)
                    unit = 'mm' if 'awns_length_mm' in measurements else 'px'
                    logger.debug(f"  Épi #{idx+1}: spike={measurements.get('spike_length_pixels', 0):.0f}px, "
                               f"whole_spike={measurements.get('whole_spike_length_pixels', 0):.0f}px, "
                               f"barbes={awns:.1f}{unit}")
            
            logger.info(f"  {len(spike_results)} épi(s) mesuré(s)")
            
            # =====================================================================
            # ÉTAPE 4: COMPTAGE DES ÉPILLETS
            # =====================================================================
            logger.info("[4/5] Comptage des épillets...")
            
            for result in spike_results:
                det = result['detection']
                spikelets = self.count_spikelets(image, det)
                result['spikelets'] = spikelets
                
                if spikelets['count']:
                    logger.info(f"  Épi #{result['id']}: {spikelets['count']} épillets "
                               f"({spikelets['confidence']})")
            
            if self.debug:
                self.save_debug_03_spikes(image, spike_results, session_dir)
            
            # =====================================================================
            # ÉTAPE 5: IDENTIFICATION DU SACHET
            # =====================================================================
            logger.info("[5/5] Identification du sachet...")
            
            bag_det = None
            bag_info = {'detected': False}
            
            if detections['bags']:
                bag_det = max(detections['bags'], key=lambda d: d.confidence)
                bag_info = self.identify_bag(image, bag_det)
                
                if bag_info.get('sample_id'):
                    logger.info(f"  Échantillon identifié: {bag_info['sample_id']}")
                else:
                    logger.info("  Chiffres non détectés sur le sachet")
            else:
                logger.info("  Aucun sachet détecté")
            
            if self.debug:
                self.save_debug_04_bag(image, bag_det, bag_info, session_dir)
            
            # =====================================================================
            # EXPORT DES RÉSULTATS
            # =====================================================================
            logger.info("Export des résultats...")
            
            # Construire le résumé
            results = {
                'image': image_path,
                'image_size': {'width': image.shape[1], 'height': image.shape[0]},
                'calibration': {
                    'ruler_detected': ruler_det is not None,
                    'pixel_per_mm': self.pixel_per_mm,
                    'ruler_length_px': ruler_det.height if ruler_det else None,
                },
                'bag': bag_info,
                'spike_count': len(spike_results),
                'spikes': [
                    {
                        'id': r['id'],
                        'measurements': r['measurements'],
                        'spikelet_count': r['spikelets'].get('count'),
                        'spikelet_method': r['spikelets'].get('method'),
                        'spikelet_confidence': r['spikelets'].get('confidence'),
                    }
                    for r in spike_results
                ],
            }
            
            # Sauvegarder JSON
            with open(session_dir / 'results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Image finale
            if self.debug:
                self.save_debug_05_final(image, results, spike_results, detections, session_dir)
            
            logger.info(f"✓ Résultats sauvegardés: {session_dir}")
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur analyse: {e}")
            logger.exception("Détails:")
            return None
    
    def analyze_batch(self, image_paths: List[str], include_existing: bool = True) -> List[Dict]:
        """
        Analyse un lot d'images et génère un CSV récapitulatif
        
        Args:
            image_paths: Liste des chemins d'images à traiter
            include_existing: Si True, inclure les résultats existants dans le CSV final
        """
        results = []
        
        for i, path in enumerate(image_paths):
            logger.info(f"\n[{i+1}/{len(image_paths)}] {Path(path).name}")
            result = self.analyze_image(path)
            if result:
                results.append(result)
        
        # Collecter aussi les résultats existants pour le CSV final
        if include_existing:
            existing_results = []
            for results_file in self.output_dir.glob('**/results.json'):
                try:
                    with open(results_file, 'r') as f:
                        existing_results.append(json.load(f))
                except Exception as e:
                    logger.warning(f"Erreur lecture {results_file}: {e}")
            
            # Dédupliquer par image_id
            all_results = {Path(r.get('image', '')).stem: r for r in existing_results}
            for r in results:
                all_results[Path(r.get('image', '')).stem] = r
            
            csv_results = list(all_results.values())
            logger.info(f"\nGénération CSV avec {len(csv_results)} résultats au total")
        else:
            csv_results = results
        
        # Générer le CSV récapitulatif
        if csv_results:
            self.export_batch_csv(csv_results)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch terminé: {len(results)}/{len(image_paths)} nouvelles images traitées")
        
        return results
    
    def export_batch_csv(self, results: List[Dict]) -> None:
        """
        Exporte les résultats en CSV récapitulatif
        
        Args:
            results: Liste des résultats d'analyse
        """
        rows = []
        
        for result in results:
            if not result:
                continue
            
            # Infos de base de l'image
            image_path = result.get('image', '')
            image_name = Path(image_path).stem
            
            base_row = {
                'image_id': image_name,
                'image_path': image_path,
                'image_width': result.get('image_size', {}).get('width', ''),
                'image_height': result.get('image_size', {}).get('height', ''),
                'pixel_per_mm': result.get('calibration', {}).get('pixel_per_mm', ''),
                'ruler_detected': result.get('calibration', {}).get('ruler_detected', False),
                'ruler_length_px': result.get('calibration', {}).get('ruler_length_px', ''),
                'spike_count': result.get('spike_count', 0),
            }
            
            # Infos du sachet
            bag = result.get('bag', {})
            base_row.update({
                'bag_detected': bag.get('detected', False),
                'sample_id': bag.get('sample_id', ''),
                'bac': bag.get('bac', ''),
                'ligne': bag.get('ligne', ''),
                'colonne': bag.get('colonne', ''),
                'bag_confidence': bag.get('confidence', ''),
                'bag_complete': bag.get('complete', ''),
            })
            
            spikes = result.get('spikes', [])
            
            if not spikes:
                # Pas d'épis - une ligne vide
                row = base_row.copy()
                row.update({
                    'spike_id': '',
                    'spike_length_px': '',
                    'spike_length_mm': '',
                    'spike_width_px': '',
                    'spike_width_mm': '',
                    'whole_spike_length_px': '',
                    'whole_spike_length_mm': '',
                    'awns_length_px': '',
                    'awns_length_mm': '',
                    'has_awns': '',
                    'area_px': '',
                    'area_mm2': '',
                    'perimeter_px': '',
                    'perimeter_mm': '',
                    'aspect_ratio': '',
                    'angle_degrees': '',
                    'spikelet_count': '',
                    'spikelet_method': '',
                    'spikelet_confidence': '',
                })
                rows.append(row)
            else:
                # Une ligne par épi
                for spike in spikes:
                    row = base_row.copy()
                    m = spike.get('measurements', {})
                    
                    row.update({
                        'spike_id': spike.get('id', ''),
                        # Longueurs spike (sans barbes)
                        'spike_length_px': m.get('spike_length_pixels', m.get('length_pixels', '')),
                        'spike_length_mm': m.get('spike_length_mm', m.get('length_mm', '')),
                        'spike_width_px': m.get('spike_width_pixels', m.get('width_pixels', '')),
                        'spike_width_mm': m.get('spike_width_mm', m.get('width_mm', '')),
                        # Longueurs whole_spike (avec barbes)
                        'whole_spike_length_px': m.get('whole_spike_length_pixels', ''),
                        'whole_spike_length_mm': m.get('whole_spike_length_mm', ''),
                        # Barbes (différence)
                        'awns_length_px': m.get('awns_length_pixels', ''),
                        'awns_length_mm': m.get('awns_length_mm', ''),
                        'has_awns': m.get('has_awns', False),
                        # Autres mesures
                        'area_px': m.get('area_pixels', ''),
                        'area_mm2': m.get('area_mm2', ''),
                        'perimeter_px': m.get('perimeter_pixels', ''),
                        'perimeter_mm': m.get('perimeter_mm', ''),
                        'aspect_ratio': m.get('aspect_ratio', ''),
                        'angle_degrees': m.get('angle_degrees', ''),
                        # Épillets
                        'spikelet_count': spike.get('spikelet_count', ''),
                        'spikelet_method': spike.get('spikelet_method', ''),
                        'spikelet_confidence': spike.get('spikelet_confidence', ''),
                        # Coordonnées
                        'center_x': m.get('center_x', ''),
                        'center_y': m.get('center_y', ''),
                        'confidence': m.get('confidence', ''),
                    })
                    rows.append(row)
        
        if not rows:
            logger.warning("Aucun résultat à exporter en CSV")
            return
        
        # Collecter TOUS les champs de toutes les lignes
        all_fieldnames = set()
        for row in rows:
            all_fieldnames.update(row.keys())
        
        # Ordonner les champs de manière logique
        priority_fields = [
            'image_id', 'image_path', 'image_width', 'image_height',
            'pixel_per_mm', 'ruler_detected', 'ruler_length_px', 'spike_count',
            'bag_detected', 'sample_id', 'bac', 'ligne', 'colonne', 'bag_confidence', 'bag_complete',
            'spike_id', 'spike_length_px', 'spike_length_mm', 'spike_width_px', 'spike_width_mm',
            'whole_spike_length_px', 'whole_spike_length_mm', 'awns_length_px', 'awns_length_mm', 'has_awns',
            'area_px', 'area_mm2', 'perimeter_px', 'perimeter_mm', 'aspect_ratio', 'angle_degrees',
            'spikelet_count', 'spikelet_method', 'spikelet_confidence',
            'center_x', 'center_y', 'confidence',
        ]
        fieldnames = [f for f in priority_fields if f in all_fieldnames]
        # Ajouter les champs restants non listés
        fieldnames += [f for f in sorted(all_fieldnames) if f not in fieldnames]
        
        # S'assurer que chaque ligne a tous les champs
        for row in rows:
            for field in fieldnames:
                if field not in row:
                    row[field] = ''
        
        # Écrire les CSV
        
        # CSV avec timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path_timestamp = self.output_dir / f'results_summary_{timestamp}.csv'
        
        with open(csv_path_timestamp, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"CSV sauvegardé: {csv_path_timestamp} ({len(rows)} lignes)")
        
        # CSV sans timestamp (latest)
        csv_path_latest = self.output_dir / 'results_summary.csv'
        with open(csv_path_latest, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"CSV latest: {csv_path_latest}")


def create_analyzer_from_config(config_path: str = "config/config_obb.yaml",
                                 output_dir: str = "output",
                                 debug: bool = True) -> WheatSpikeAnalyzerOBB:
    """
    Crée un analyseur depuis un fichier de configuration
    
    Args:
        config_path: Chemin vers le fichier YAML
        output_dir: Dossier de sortie
        debug: Activer le mode debug
        
    Returns:
        WheatSpikeAnalyzerOBB configuré
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return WheatSpikeAnalyzerOBB(config, output_dir, debug)
