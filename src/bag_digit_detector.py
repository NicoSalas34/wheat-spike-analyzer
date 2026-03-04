#!/usr/bin/env python3
"""
Détecteur de chiffres manuscrits sur les sachets basé sur YOLO

Utilise un modèle YOLO entraîné pour détecter les nombres 1-20
écrits sur les sachets d'échantillons.

Les 3 chiffres sur chaque sachet représentent (en partant de l'ouverture):
1. Bac: Numéro du bac de culture
2. Ligne: Numéro de la ligne dans le bac
3. Colonne: Numéro de la colonne dans le bac

L'ouverture du sachet est détectée automatiquement pour ordonner correctement les chiffres.
"""

import logging
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class BagDigitDetector:
    """
    Détecteur de chiffres manuscrits sur les sachets basé sur YOLO
    """
    
    # Mapping des classes YOLO vers les valeurs numériques
    # Classes: 0="1", 1="2", ..., 19="20"
    CLASS_TO_VALUE = {i: i + 1 for i in range(20)}  # {0: 1, 1: 2, ..., 19: 20}
    
    def __init__(
        self,
        model_path: str = "models/bag_digits_yolo.pt",
        confidence_threshold: float = 0.5,
        device: str = None,
        opening_model_path: str = "models/bag_opening_yolo.pt",
        opening_confidence_threshold: float = 0.3,
    ):
        """
        Initialise le détecteur
        
        Args:
            model_path: Chemin vers le modèle YOLO pour les chiffres
            confidence_threshold: Seuil de confiance minimum pour les chiffres
            device: Device (cuda:0, cpu, ou None pour auto)
            opening_model_path: Chemin vers le modèle YOLO pour l'ouverture
            opening_confidence_threshold: Seuil de confiance pour l'ouverture
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = device
        
        # Modèle pour détecter l'ouverture du sachet
        self.opening_model_path = Path(opening_model_path)
        self.opening_model = None
        self.opening_confidence_threshold = opening_confidence_threshold
        
        if self.model_path.exists():
            self._load_model()
        else:
            logger.warning(f"Modèle non trouvé: {model_path}")
            logger.warning("Utilisez scripts/train_yolo_bag_digits.py pour entraîner le modèle")
        
        # Charger le modèle d'ouverture
        self._load_opening_model()
    
    def _load_model(self):
        """Charge le modèle YOLO"""
        try:
            from ultralytics import YOLO
            import torch
            
            # Déterminer le device
            if self.device is None:
                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Chargement du modèle bag_digits: {self.model_path}")
            self.model = YOLO(str(self.model_path))
            
            # Vérifier les classes
            if hasattr(self.model, 'names'):
                logger.info(f"Classes: {self.model.names}")
            
            logger.info(f"✓ Modèle bag_digits chargé sur {self.device}")
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            self.model = None
    
    def _load_opening_model(self):
        """Charge le modèle YOLO pour détecter l'ouverture du sachet"""
        if not self.opening_model_path.exists():
            logger.info(f"Modèle bag_opening non trouvé: {self.opening_model_path}")
            logger.info("Fallback vers détection heuristique de l'orientation")
            return
        
        try:
            from ultralytics import YOLO
            import torch
            
            # Déterminer le device
            if self.device is None:
                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Chargement du modèle bag_opening: {self.opening_model_path}")
            self.opening_model = YOLO(str(self.opening_model_path))
            logger.info(f"✓ Modèle bag_opening chargé sur {self.device}")
            
        except Exception as e:
            logger.warning(f"Erreur chargement modèle bag_opening: {e}")
            logger.info("Fallback vers détection heuristique de l'orientation")
            self.opening_model = None
    
    def is_available(self) -> bool:
        """Vérifie si le détecteur est disponible"""
        return self.model is not None
    
    def is_opening_detector_available(self) -> bool:
        """Vérifie si le détecteur d'ouverture YOLO est disponible"""
        return self.opening_model is not None
    
    def _detect_bag_orientation_yolo(self, bag_roi: np.ndarray) -> Optional[Dict]:
        """
        Détecte l'orientation du sachet via YOLO (modèle bag_opening).
        
        Args:
            bag_roi: Image du sachet (BGR)
            
        Returns:
            Dict avec orientation ou None si échec
        """
        if not self.is_opening_detector_available():
            return None
        
        try:
            h, w = bag_roi.shape[:2]
            is_vertical = h > w
            
            # Inférence YOLO
            results = self.opening_model(
                bag_roi,
                verbose=False,
                conf=self.opening_confidence_threshold,
                device=self.device,
            )
            
            # Chercher la détection bag_opening avec la meilleure confiance
            best_detection = None
            best_conf = 0
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        best_detection = {
                            'bbox': (x1, y1, x2, y2),
                            'center_x': center_x,
                            'center_y': center_y,
                            'confidence': conf,
                        }
                        best_conf = conf
            
            if best_detection is None:
                logger.debug("Aucune ouverture détectée par YOLO")
                return None
            
            # Déterminer le côté de l'ouverture basé sur la position du centre
            center_x = best_detection['center_x']
            center_y = best_detection['center_y']
            
            if is_vertical:
                # Sachet vertical: comparer position Y
                if center_y < h / 2:
                    opening_side = 'top'
                else:
                    opening_side = 'bottom'
            else:
                # Sachet horizontal: comparer position X
                if center_x < w / 2:
                    opening_side = 'left'
                else:
                    opening_side = 'right'
            
            result = {
                'opening_side': opening_side,
                'is_vertical': is_vertical,
                'confidence': best_detection['confidence'],
                'detection_method': 'yolo',
                'opening_bbox': best_detection['bbox'],
            }
            
            logger.debug(f"Orientation sachet (YOLO): {opening_side} (conf: {best_conf:.2f})")
            return result
            
        except Exception as e:
            logger.warning(f"Erreur détection ouverture YOLO: {e}")
            return None
    
    def _detect_bag_orientation_heuristic(self, bag_roi: np.ndarray) -> Dict:
        """
        Détecte l'orientation du sachet et la position de l'ouverture.
        
        L'ouverture du sachet est généralement:
        - Plus large que le fond
        - A une forme plus irrégulière (plis, ouverture)
        
        Stratégie: Analyser la largeur du sachet à différentes hauteurs.
        L'ouverture est du côté où le sachet est plus large ou plus irrégulier.
        
        Args:
            bag_roi: Image du sachet (BGR)
            
        Returns:
            Dict avec:
            - 'opening_side': 'top', 'bottom', 'left', 'right'
            - 'is_vertical': bool (True si sachet vertical)
            - 'confidence': float
        """
        h, w = bag_roi.shape[:2]
        
        # Convertir en niveaux de gris et binariser
        gray = cv2.cvtColor(bag_roi, cv2.COLOR_BGR2GRAY)
        
        # Appliquer un flou pour réduire le bruit
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Binarisation adaptative pour détecter le sachet
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 21, 5)
        
        # Déterminer si le sachet est vertical ou horizontal
        # basé sur le ratio d'aspect
        is_vertical = h > w
        
        result = {
            'opening_side': 'top',  # Par défaut
            'is_vertical': is_vertical,
            'confidence': 0.5,
            'width_profile': [],
        }
        
        if is_vertical:
            # Sachet vertical: analyser la largeur à chaque hauteur
            # Diviser en bandes horizontales
            n_bands = 10
            band_height = h // n_bands
            widths = []
            
            for i in range(n_bands):
                y_start = i * band_height
                y_end = min((i + 1) * band_height, h)
                band = binary[y_start:y_end, :]
                
                # Calculer la largeur moyenne du contenu dans cette bande
                col_sums = np.sum(band, axis=0)
                non_zero = np.where(col_sums > 0)[0]
                
                if len(non_zero) > 0:
                    width = non_zero[-1] - non_zero[0]
                else:
                    width = 0
                widths.append(width)
            
            result['width_profile'] = widths
            
            # Comparer les largeurs du haut vs bas
            top_avg = np.mean(widths[:3]) if len(widths) >= 3 else np.mean(widths[:len(widths)//2])
            bottom_avg = np.mean(widths[-3:]) if len(widths) >= 3 else np.mean(widths[len(widths)//2:])
            
            # L'ouverture est du côté le plus large
            if top_avg > bottom_avg * 1.1:  # 10% plus large = ouverture en haut
                result['opening_side'] = 'top'
                result['confidence'] = min(0.9, 0.5 + (top_avg - bottom_avg) / (top_avg + 1) * 0.5)
            elif bottom_avg > top_avg * 1.1:  # Ouverture en bas
                result['opening_side'] = 'bottom'
                result['confidence'] = min(0.9, 0.5 + (bottom_avg - top_avg) / (bottom_avg + 1) * 0.5)
            else:
                # Pas de différence claire, analyser la variance (plus de variance = ouverture)
                top_var = np.var(widths[:3]) if len(widths) >= 3 else 0
                bottom_var = np.var(widths[-3:]) if len(widths) >= 3 else 0
                
                if top_var > bottom_var:
                    result['opening_side'] = 'top'
                else:
                    result['opening_side'] = 'bottom'
                result['confidence'] = 0.6
        else:
            # Sachet horizontal: analyser la hauteur à chaque position x
            n_bands = 10
            band_width = w // n_bands
            heights = []
            
            for i in range(n_bands):
                x_start = i * band_width
                x_end = min((i + 1) * band_width, w)
                band = binary[:, x_start:x_end]
                
                row_sums = np.sum(band, axis=1)
                non_zero = np.where(row_sums > 0)[0]
                
                if len(non_zero) > 0:
                    height = non_zero[-1] - non_zero[0]
                else:
                    height = 0
                heights.append(height)
            
            result['width_profile'] = heights
            
            left_avg = np.mean(heights[:3]) if len(heights) >= 3 else np.mean(heights[:len(heights)//2])
            right_avg = np.mean(heights[-3:]) if len(heights) >= 3 else np.mean(heights[len(heights)//2:])
            
            if left_avg > right_avg * 1.1:
                result['opening_side'] = 'left'
                result['confidence'] = min(0.9, 0.5 + (left_avg - right_avg) / (left_avg + 1) * 0.5)
            elif right_avg > left_avg * 1.1:
                result['opening_side'] = 'right'
                result['confidence'] = min(0.9, 0.5 + (right_avg - left_avg) / (right_avg + 1) * 0.5)
            else:
                result['opening_side'] = 'left'  # Par défaut
                result['confidence'] = 0.5
        
        logger.debug(f"Orientation sachet (heuristique): {result['opening_side']} (conf: {result['confidence']:.2f})")
        
        result['detection_method'] = 'heuristic'
        return result
    
    def _detect_bag_orientation(self, bag_roi: np.ndarray) -> Dict:
        """
        Détecte l'orientation du sachet et la position de l'ouverture.
        
        Utilise le modèle YOLO bag_opening en priorité, avec fallback
        sur la méthode heuristique si YOLO n'est pas disponible ou échoue.
        
        Args:
            bag_roi: Image du sachet (BGR)
            
        Returns:
            Dict avec:
            - 'opening_side': 'top', 'bottom', 'left', 'right'
            - 'is_vertical': bool (True si sachet vertical)
            - 'confidence': float
            - 'detection_method': 'yolo' ou 'heuristic'
        """
        # Essayer d'abord avec YOLO
        if self.is_opening_detector_available():
            yolo_result = self._detect_bag_orientation_yolo(bag_roi)
            if yolo_result is not None:
                return yolo_result
            logger.debug("YOLO bag_opening a échoué, fallback vers heuristique")
        
        # Fallback vers la méthode heuristique
        return self._detect_bag_orientation_heuristic(bag_roi)
    
    def _sort_detections_by_opening(self, detections: List[Dict], 
                                     orientation: Dict, 
                                     roi_shape: Tuple[int, int]) -> List[Dict]:
        """
        Trie les détections en fonction de leur distance à l'ouverture du sachet.
        Le premier chiffre (bac) est le plus proche de l'ouverture.
        
        Args:
            detections: Liste des détections
            orientation: Résultat de _detect_bag_orientation
            roi_shape: (height, width) de la ROI
            
        Returns:
            Liste triée (du plus proche au plus loin de l'ouverture)
        """
        if not detections:
            return detections
        
        h, w = roi_shape
        opening_side = orientation.get('opening_side', 'top')
        
        if opening_side == 'top':
            # Trier par Y croissant (haut vers bas)
            return sorted(detections, key=lambda d: d['center_y'])
        elif opening_side == 'bottom':
            # Trier par Y décroissant (bas vers haut)
            return sorted(detections, key=lambda d: -d['center_y'])
        elif opening_side == 'left':
            # Trier par X croissant (gauche vers droite)
            return sorted(detections, key=lambda d: d['center_x'])
        elif opening_side == 'right':
            # Trier par X décroissant (droite vers gauche)
            return sorted(detections, key=lambda d: -d['center_x'])
        else:
            # Par défaut: haut vers bas
            return sorted(detections, key=lambda d: d['center_y'])
    
    def detect(
        self,
        bag_roi: np.ndarray,
    ) -> List[Dict]:
        """
        Détecte les chiffres sur une ROI de sachet
        
        Args:
            bag_roi: Image du sachet (BGR)
            
        Returns:
            Liste de dictionnaires avec les détections triées de haut en bas:
            [{'class': 0, 'value': 1, 'center_x': 123, 'center_y': 45, 'confidence': 0.95}, ...]
        """
        if not self.is_available():
            logger.warning("Modèle bag_digits non disponible")
            return []
        
        try:
            # Inférence YOLO
            results = self.model(
                bag_roi,
                verbose=False,
                conf=self.confidence_threshold,
                device=self.device,
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    detections.append({
                        'class': cls,
                        'value': self.CLASS_TO_VALUE.get(cls, cls + 1),
                        'center_x': center_x,
                        'center_y': center_y,
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                    })
            
            # Trier de haut en bas (par center_y croissant)
            detections.sort(key=lambda d: d['center_y'])
            
            logger.debug(f"Détecté {len(detections)} chiffres: {[d['value'] for d in detections]}")
            
            return detections
            
        except Exception as e:
            logger.error(f"Erreur détection bag_digits: {e}")
            return []
    
    def detect_tta(
        self,
        bag_roi: np.ndarray,
        augmentations: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Détecte les chiffres avec TTA (Test-Time Augmentation).
        
        Exécute la détection sur plusieurs versions augmentées de l'image,
        regroupe les détections par position et vote sur les valeurs.
        
        Args:
            bag_roi: Image du sachet (BGR)
            augmentations: Liste d'augmentations (défaut: geometric + photometric)
            
        Returns:
            Liste de détections avec vote majoritaire sur les valeurs
        """
        if not self.is_available():
            return []
        
        try:
            from .tta import tta_detect_digits
        except ImportError:
            from tta import tta_detect_digits
        
        return tta_detect_digits(
            model=self.model,
            bag_roi=bag_roi,
            conf=self.confidence_threshold,
            class_to_value=self.CLASS_TO_VALUE,
            augmentations=augmentations,
            nms_iou=0.5,
            device=self.device,
        )
    
    def detect_sample_id_tta(
        self,
        bag_roi: np.ndarray,
        augmentations: Optional[List[str]] = None,
    ) -> Dict:
        """
        Version TTA de detect_sample_id.
        Utilise TTA pour la détection des chiffres, puis la même logique
        d'orientation et d'assignation bac/ligne/colonne.
        """
        # Détecter l'orientation du sachet (pas de TTA ici, c'est robuste)
        orientation = self._detect_bag_orientation(bag_roi)
        
        # Détecter les chiffres avec TTA
        detections = self.detect_tta(bag_roi, augmentations=augmentations)
        
        # Trier les détections en fonction de la position de l'ouverture
        h, w = bag_roi.shape[:2]
        sorted_detections = self._sort_detections_by_opening(detections, orientation, (h, w))
        
        result = {
            'bac': None,
            'ligne': None,
            'colonne': None,
            'sample_id': None,
            'detections': sorted_detections,
            'confidence': 0.0,
            'complete': False,
            'orientation': orientation,
        }
        
        if not sorted_detections:
            return result
        
        if len(sorted_detections) >= 1:
            result['bac'] = sorted_detections[0]['value']
        if len(sorted_detections) >= 2:
            result['ligne'] = sorted_detections[1]['value']
        if len(sorted_detections) >= 3:
            result['colonne'] = sorted_detections[2]['value']
        
        if detections:
            result['confidence'] = sum(d['confidence'] for d in detections) / len(detections)
        
        result['complete'] = all([result['bac'], result['ligne'], result['colonne']])
        
        if result['complete']:
            result['sample_id'] = f"{result['bac']}-{result['ligne']}-{result['colonne']}"
        elif result['bac'] or result['ligne'] or result['colonne']:
            parts = []
            parts.append(str(result['bac']) if result['bac'] else '?')
            parts.append(str(result['ligne']) if result['ligne'] else '?')
            parts.append(str(result['colonne']) if result['colonne'] else '?')
            result['sample_id'] = '-'.join(parts)
        
        return result

    def detect_sample_id(
        self,
        bag_roi: np.ndarray,
    ) -> Dict:
        """
        Détecte l'identifiant complet d'un échantillon (bac, ligne, colonne)
        
        L'ordre est déterminé en détectant l'ouverture du sachet.
        Le premier chiffre sous l'ouverture est le bac, puis ligne, puis colonne.
        
        Args:
            bag_roi: Image du sachet (BGR)
            
        Returns:
            Dictionnaire avec:
            {
                'bac': int ou None,
                'ligne': int ou None,
                'colonne': int ou None,
                'sample_id': str (format "bac-ligne-colonne"),
                'detections': list des détections brutes,
                'confidence': float (confiance moyenne),
                'complete': bool (True si les 3 valeurs sont détectées),
                'orientation': dict (informations sur l'orientation du sachet)
            }
        """
        # Détecter l'orientation du sachet
        orientation = self._detect_bag_orientation(bag_roi)
        
        # Détecter les chiffres
        detections = self.detect(bag_roi)
        
        # Trier les détections en fonction de la position de l'ouverture
        h, w = bag_roi.shape[:2]
        sorted_detections = self._sort_detections_by_opening(detections, orientation, (h, w))
        
        result = {
            'bac': None,
            'ligne': None,
            'colonne': None,
            'sample_id': None,
            'detections': sorted_detections,
            'confidence': 0.0,
            'complete': False,
            'orientation': orientation,
        }
        
        if not sorted_detections:
            return result
        
        # Les 3 premiers chiffres (depuis l'ouverture) sont bac, ligne, colonne
        if len(sorted_detections) >= 1:
            result['bac'] = sorted_detections[0]['value']
        if len(sorted_detections) >= 2:
            result['ligne'] = sorted_detections[1]['value']
        if len(sorted_detections) >= 3:
            result['colonne'] = sorted_detections[2]['value']
        
        # Calculer la confiance moyenne
        if detections:
            result['confidence'] = sum(d['confidence'] for d in detections) / len(detections)
        
        # Vérifier si complet
        result['complete'] = all([result['bac'], result['ligne'], result['colonne']])
        
        # Construire l'ID de l'échantillon
        if result['complete']:
            result['sample_id'] = f"{result['bac']}-{result['ligne']}-{result['colonne']}"
        elif result['bac'] or result['ligne'] or result['colonne']:
            # ID partiel
            parts = []
            parts.append(str(result['bac']) if result['bac'] else '?')
            parts.append(str(result['ligne']) if result['ligne'] else '?')
            parts.append(str(result['colonne']) if result['colonne'] else '?')
            result['sample_id'] = '-'.join(parts)
        
        return result
    
    def detect_from_full_image(
        self,
        image: np.ndarray,
        bag_bbox: Tuple[int, int, int, int],
        margin: int = 20,
    ) -> Dict:
        """
        Détecte les chiffres sur un sachet à partir de l'image complète
        
        Args:
            image: Image complète (BGR)
            bag_bbox: Bounding box du sachet (x1, y1, x2, y2)
            margin: Marge autour du sachet
            
        Returns:
            Résultat de detect_sample_id
        """
        x1, y1, x2, y2 = bag_bbox
        h, w = image.shape[:2]
        
        # Ajouter la marge
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        # Extraire la ROI
        bag_roi = image[y1:y2, x1:x2]
        
        # Détecter
        result = self.detect_sample_id(bag_roi)
        
        # Ajuster les coordonnées des détections par rapport à l'image complète
        for det in result['detections']:
            det['center_x'] += x1
            det['center_y'] += y1
            bx1, by1, bx2, by2 = det['bbox']
            det['bbox'] = (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)
            det['bbox_in_bag'] = (bx1, by1, bx2, by2)  # Conserver aussi les coords dans la ROI
        
        return result


def create_bag_digit_detector(config: dict) -> Optional[BagDigitDetector]:
    """
    Crée un détecteur de chiffres sur sachets à partir de la configuration
    
    Args:
        config: Configuration globale
        
    Returns:
        BagDigitDetector ou None si non disponible
    """
    bag_config = config.get('bag_digits', {})
    
    model_path = bag_config.get('model_path', 'models/bag_digits_yolo.pt')
    confidence = bag_config.get('confidence_threshold', 0.5)
    
    # Configuration pour le modèle d'ouverture
    opening_model_path = bag_config.get('opening_model_path', 'models/bag_opening_yolo.pt')
    opening_confidence = bag_config.get('opening_confidence_threshold', 0.3)
    
    if not Path(model_path).exists():
        logger.info(f"Modèle bag_digits non trouvé: {model_path}")
        return None
    
    try:
        detector = BagDigitDetector(
            model_path=model_path,
            confidence_threshold=confidence,
            opening_model_path=opening_model_path,
            opening_confidence_threshold=opening_confidence,
        )
        return detector if detector.is_available() else None
    except Exception as e:
        logger.warning(f"Impossible de créer BagDigitDetector: {e}")
        return None
