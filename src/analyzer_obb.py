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

import gc
import cv2
import numpy as np
import logging
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from .spike_matcher import match_spikes_hungarian
except ImportError:
    from spike_matcher import match_spikes_hungarian

try:
    from .spike_segmenter import SpikeSegmenter
except ImportError:
    from spike_segmenter import SpikeSegmenter

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
    source: str = 'unknown'  # Origine: 'full' (image entière), 'tile', ou 'unknown'
    
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
            debug: Niveau de debug:
                   - False/0: aucune image de debug
                   - 'low'/1: uniquement result_annotated.png
                   - True/2: toutes les images de debug (défaut)
        """
        self.config = config
        self.output_dir = Path(output_dir)
        
        # Normaliser le niveau de debug
        # Note: vérifier True/False avec 'is' avant '==' car True == 1 en Python
        if debug is True or debug == 2:
            self.debug_level = 2
        elif debug == 'low' or debug == 1:
            self.debug_level = 1
        else:  # False, 0, None
            self.debug_level = 0
        
        # Compatibilité: self.debug = True si debug_level > 0
        self.debug = self.debug_level > 0
        
        # Charger le modèle YOLO OBB principal
        self.detector = self._load_yolo_model()
        
        # Charger les modèles auxiliaires
        self.spikelet_counter = self._load_spikelet_counter()
        self.bag_digit_detector = self._load_bag_digit_detector()
        self.graduation_detector = self._load_graduation_detector()
        self.spike_segmenter = self._load_spike_segmenter()
        self.spikelet_segmenter = self._load_spikelet_segmenter()
        self.rachis_detector = self._load_rachis_detector()
        
        # TTA configuration — par étape
        # Étape 1: OBB detection (augment=True Ultralytics)
        obb_tta = self.config.get('yolo', {}).get('tta', {})
        self.use_tta_obb = obb_tta.get('enabled', False)
        
        # Étape 5: Spike segmentation
        seg_tta = self.config.get('segmentation', {}).get('tta', {})
        self.use_tta_spike_seg = seg_tta.get('enabled', False)
        self.tta_spike_seg_augs = seg_tta.get(
            'augmentations', ['original', 'flip_h', 'flip_v', 'rot180']
        )
        
        # Étape 6: Spikelet counting (existant)
        spk_tta = self.config.get('spikelet_counting', {}).get('tta', {})
        self.use_tta = spk_tta.get('enabled', False)
        self.tta_augmentations = spk_tta.get(
            'augmentations', ['original', 'flip_h', 'flip_v', 'rot180']
        )
        
        # Étape 8: Rachis detection
        rachis_tta = self.config.get('rachis_detection', {}).get('tta', {})
        self.use_tta_rachis = rachis_tta.get('enabled', False)
        self.tta_rachis_augs = rachis_tta.get(
            'augmentations', ['original', 'flip_h', 'flip_v', 'rot180', 'brightness_up', 'contrast_up']
        )
        self.tta_rachis_consensus = rachis_tta.get('consensus_threshold', 0.4)
        
        # Étape 9: Bag digit OCR
        bag_tta = self.config.get('bag_digits', {}).get('tta', {})
        self.use_tta_bag = bag_tta.get('enabled', False)
        self.tta_bag_augs = bag_tta.get(
            'augmentations', ['original', 'flip_h', 'flip_v', 'rot180', 'brightness_up', 'contrast_up']
        )
        
        # Spike matching configuration
        matching_config = self.config.get('spike_matching', {})
        self.matching_min_iou = matching_config.get('min_iou', 0.15)
        self.matching_min_containment = matching_config.get('min_containment', 0.5)
        
        # Ratio de calibration (calculé à chaque image)
        self.pixel_per_mm: Optional[float] = None
        self.calibration_method: Optional[str] = None  # 'graduations', 'ruler_obb', or None
        self.calibration_details: Dict = {}  # Details about calibration
        
        # Log TTA status
        tta_steps = []
        if self.use_tta_obb: tta_steps.append("OBB")
        if self.use_tta_spike_seg: tta_steps.append("SpikeSeg")
        if self.use_tta: tta_steps.append("Spikelets")
        if self.use_tta_rachis: tta_steps.append("Rachis")
        if self.use_tta_bag: tta_steps.append("BagOCR")
        if tta_steps:
            logger.info(f"✓ TTA activé pour: {', '.join(tta_steps)}")
        
        logger.info("✓ WheatSpikeAnalyzerOBB initialisé")

    # =========================================================================
    # GESTION MÉMOIRE
    # =========================================================================

    @staticmethod
    def clear_memory(log: bool = True):
        """Libère la RAM (garbage collector) et la VRAM GPU (torch cache).

        Appeler cette méthode entre les images en mode batch pour éviter
        les crashs OOM sur de longues séries.

        Gère proprement les erreurs HIP (GPU AMD / ROCm) : si le GPU est
        dans un état d'erreur (hipErrorLaunchFailure, etc.), on tente un
        ``torch.cuda.synchronize()`` puis ``reset_peak_memory_stats()``
        avant de réessayer. Si ça échoue toujours, on continue sans crash.
        """
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                # --- Tentative de nettoyage GPU ---
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except RuntimeError as e:
                    # HIP/CUDA kernel failure : le GPU est dans un mauvais
                    # état. On tente de synchroniser puis de purger.
                    logger.warning(f"GPU cache cleanup failed ({e}), attempting recovery...")
                    try:
                        torch.cuda.synchronize()
                    except RuntimeError:
                        pass
                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except RuntimeError:
                        pass
                    try:
                        torch.cuda.empty_cache()
                    except RuntimeError as e2:
                        logger.warning(f"GPU recovery failed, continuing without GPU cleanup: {e2}")

                if log:
                    try:
                        allocated = torch.cuda.memory_allocated() / 1024**2
                        reserved = torch.cuda.memory_reserved() / 1024**2
                        logger.info(
                            f"  GPU: {allocated:.0f} MB alloc / {reserved:.0f} MB reserved"
                        )
                    except RuntimeError:
                        logger.info("  GPU: unable to query memory (device error)")
        except ImportError:
            pass

        if log:
            try:
                import psutil
                proc = psutil.Process()
                rss = proc.memory_info().rss / 1024**2
                logger.info(f"  RAM process: {rss:.0f} MB (RSS)")
            except ImportError:
                pass
            logger.info("  Mémoire libérée (gc.collect + GPU cleanup)")

    @staticmethod
    def _recover_gpu():
        """Tente de récupérer le GPU après une erreur HIP/CUDA fatale.

        Après un ``hipErrorLaunchFailure`` (code 719) ou autre erreur
        noyau, le device est dans un état invalide. On tente de le
        réinitialiser pour pouvoir continuer le batch.
        """
        gc.collect()
        try:
            import torch
            if not torch.cuda.is_available():
                return
            # synchronize() force la propagation de l'erreur asynchrone
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass
            # Vider le cache
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass
            # Réinitialiser les compteurs d'allocation
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
            except (RuntimeError, AttributeError):
                pass
            logger.info("  GPU recovery: done")
        except ImportError:
            pass

    def _clear_detection_caches(self):
        """Supprime les caches de détection entre les images (sliced inference debug, etc.)."""
        for attr in ('_last_raw_full_dets', '_last_raw_tile_dets', '_last_tile_grid'):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass

    @staticmethod
    def _strip_heavy_arrays(result: Dict) -> Dict:
        """Supprime les gros tableaux numpy/masques d'un résultat déjà sauvé en JSON.

        Permet de garder le dict léger dans la liste ``results`` de
        ``analyze_batch`` sans manger toute la RAM.
        """
        if result is None:
            return result
        for spike in result.get('spikes', []):
            # Les masques, contours et positions bruts ne sont pas dans le JSON
            spike.pop('segmentation', None)
            spike.pop('spikelet_details', None)
            spike.pop('insertion_angles', None)
            spikelets = spike.get('spikelets', {})
            if isinstance(spikelets, dict):
                spikelets.pop('masks_roi', None)
                spikelets.pop('positions', None)
        return result

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
    
    def _load_graduation_detector(self):
        """Charge le détecteur de graduations sur la règle"""
        from ultralytics import YOLO
        
        grad_config = self.config.get('graduation_detection', {})
        
        if not grad_config.get('enabled', True):
            return None
        
        model_path = grad_config.get('model_path', 'runs/graduations_obb/weights/best.pt')
        
        if not Path(model_path).exists():
            logger.warning(f"Modèle graduations non trouvé: {model_path}")
            return None
        
        try:
            model = YOLO(model_path)
            logger.info(f"✓ Détecteur graduations chargé: {model_path}")
            logger.info(f"  Classes: {model.names}")
            return model
        except Exception as e:
            logger.warning(f"Erreur chargement modèle graduations: {e}")
            return None
    
    def _load_spike_segmenter(self):
        """Charge le modèle YOLO-Seg pour la segmentation des épis.
        
        Remplace SAM2 par un modèle YOLO-Seg entraîné spécifiquement.
        Retourne un tuple (yolo_model, metrics_helper) ou (None, None).
        """
        from ultralytics import YOLO
        
        seg_config = self.config.get('segmentation', {})
        
        if not seg_config.get('enabled', False):
            logger.info("Segmentation des épis désactivée dans la config")
            return None
        
        model_path = seg_config.get('spike_seg_model', 'models/spike_seg_yolo.pt')
        min_mask_area = seg_config.get('min_mask_area', 1000)
        conf = seg_config.get('confidence_threshold', 0.5)
        
        if not Path(model_path).exists():
            logger.warning(f"Modèle spike-seg non trouvé: {model_path}")
            return None
        
        try:
            model = YOLO(model_path)
            logger.info(f"✓ Segmenteur YOLO-Seg épis chargé: {model_path}")
            logger.info(f"  Classes: {model.names}, conf={conf}")
            # Stocker la config pour l'inférence
            self._spike_seg_conf = conf
            self._spike_seg_min_area = min_mask_area
            # Créer un SpikeSegmenter (sans SAM) pour les méthodes de métriques
            self._spike_metrics_helper = SpikeSegmenter(min_mask_area=min_mask_area)
            return model
        except Exception as e:
            logger.warning(f"Erreur chargement YOLO-Seg épis: {e}")
            return None
    
    def _load_spikelet_segmenter(self):
        """Placeholder — les épillets utilisent désormais YOLO-Seg (étape 7)."""
        return None
    
    def _segment_spike_yolo(
        self, image: np.ndarray, obb_detection, margin: int = 30,
        use_tta: bool = False, tta_augmentations: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        """Segmente un épi avec le modèle YOLO-Seg.
        
        Utilise un crop warp-affine redressé (épi vertical) comme à l'entraînement,
        puis re-projette le masque dans les coordonnées de l'image originale.
        
        Args:
            image: Image complète BGR
            obb_detection: Détection OBB de l'épi
            margin: Marge autour de l'OBB pour le crop
            use_tta: Si True, utilise TTA pour un masque consensus
            tta_augmentations: Augmentations TTA (défaut: flip_h, flip_v, rot180)
        
        Produit un dict compatible avec l'ancien format SAM2 :
        {'mask', 'roi_mask', 'roi_bbox', 'contour', 'contour_area_px', 'contour_perimeter_px'}
        """
        if self.spike_segmenter is None:
            return None
        
        h, w = image.shape[:2]
        
        # --- Crop OBB redressé (même procédure que l'entraînement) ---
        cx, cy = obb_detection.center
        rw, rh = obb_detection.width, obb_detection.height  # width < height garanti
        angle = obb_detection.angle  # déjà ajusté pour le swap w/h
        
        out_w = int(rw + 2 * margin)
        out_h = int(rh + 2 * margin)
        
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        M[0, 2] += out_w / 2 - cx
        M[1, 2] += out_h / 2 - cy
        
        crop = cv2.warpAffine(
            image, M, (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        
        # --- Inférence YOLO-Seg (avec ou sans TTA) ---
        if use_tta:
            try:
                from .tta import tta_yolo_segment
            except ImportError:
                from tta import tta_yolo_segment
            tta_result = tta_yolo_segment(
                model=self.spike_segmenter,
                crop=crop,
                conf=self._spike_seg_conf,
                augmentations=tta_augmentations,
                min_mask_area=self._spike_seg_min_area,
                consensus_threshold=0.5,
            )
            if tta_result is None:
                return None
            mask_crop_bin, _ = tta_result
        else:
            results = self.spike_segmenter(crop, conf=self._spike_seg_conf, verbose=False)
            
            if not results or results[0].masks is None:
                return None
            
            res = results[0]
            if res.masks.data is None or len(res.masks.data) == 0:
                return None
            
            # Prendre le masque avec la plus grande confiance
            best_idx = int(res.boxes.conf.argmax())
            mask_tensor = res.masks.data[best_idx]
            
            # Convertir en masque binaire à la taille du crop
            mask_np = mask_tensor.cpu().numpy()
            mask_crop = cv2.resize(
                mask_np, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            mask_crop_bin = (mask_crop > 0.5).astype(np.uint8) * 255
        
        if cv2.countNonZero(mask_crop_bin) < self._spike_seg_min_area:
            return None
        
        # --- Re-projeter le masque vers l'image originale ---
        M_inv = cv2.invertAffineTransform(M)
        full_mask = cv2.warpAffine(
            mask_crop_bin, M_inv, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        
        # Trouver le contour principal dans l'image originale
        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        main_contour = max(contours, key=cv2.contourArea)
        
        area_px = cv2.contourArea(main_contour)
        if area_px < self._spike_seg_min_area:
            return None
        
        # Bbox axis-aligned du masque
        bx, by, bw, bh = cv2.boundingRect(main_contour)
        x1, y1 = bx, by
        x2, y2 = bx + bw, by + bh
        roi_mask = full_mask[y1:y2, x1:x2]
        
        return {
            'mask': full_mask,
            'roi_mask': roi_mask,
            'roi_bbox': (x1, y1, x2, y2),
            'contour': main_contour,
            'contour_area_px': float(area_px),
            'contour_perimeter_px': float(cv2.arcLength(main_contour, True)),
        }
    
    @staticmethod
    def _spike_border_near_whole_spike(
        sp_det: 'OBBDetection',
        ws_det: 'OBBDetection',
        threshold_px: float = 40.0,
    ) -> bool:
        """Vérifie si au moins un bord de la bbox du spike est proche d'un bord
        de la bbox du whole_spike (axis-aligned).

        Si c'est le cas, le spike est probablement « coupé » à cet endroit et un
        raffinement est justifié.

        Args:
            sp_det: Détection OBB du spike.
            ws_det: Détection OBB du whole_spike.
            threshold_px: Distance (px) en dessous de laquelle on considère qu'un
                bord est « proche ».

        Returns:
            True si au moins un bord du spike est proche d'un bord du whole_spike.
        """
        sp_x1, sp_y1, sp_x2, sp_y2 = sp_det.bbox
        ws_x1, ws_y1, ws_x2, ws_y2 = ws_det.bbox

        near = (
            abs(sp_x1 - ws_x1) <= threshold_px
            or abs(sp_x2 - ws_x2) <= threshold_px
            or abs(sp_y1 - ws_y1) <= threshold_px
            or abs(sp_y2 - ws_y2) <= threshold_px
        )
        return near

    def _refine_spike_with_whole_spike(
        self,
        image: np.ndarray,
        sp_det: 'OBBDetection',
        ws_det: 'OBBDetection',
        border_threshold_px: float = 40.0,
        margin: int = 60,
    ) -> Optional[Tuple['OBBDetection', Dict]]:
        """Raffine la détection d'un spike en utilisant le whole_spike.

        Stratégie en deux temps :
        1. **Vérification de bordure** : on vérifie qu'au moins un bord de la
           bbox du spike est loin (> border_threshold_px) d'un bord de la bbox
           du whole_spike.  Si tous les bords sont proches, le spike semble
           correctement détecté et aucun raffinement n'est nécessaire.
        2. **Consensus detection ↔ segmentation** : on fait un crop redressé
           couvrant la zone whole_spike, on y lance le modèle de segmentation,
           on dérive un OBB « seg » du masque, puis on le compare à l'OBB
           « det » original :
           - Si l'OBB seg est nettement plus grand (longueur +10 %+) → on prend
             l'OBB seg (le modèle de détection était trop court).
           - Si les deux sont proches (± 10 %) → consensus par union convexe
             de leurs contours.
           - Sinon on garde l'OBB det original.

        Args:
            image: Image complète BGR
            sp_det: Détection OBB du spike (potentiellement incomplète)
            ws_det: Détection OBB du whole_spike associé
            border_threshold_px: Distance (px) pour la vérification de bordure
            margin: Marge en pixels autour du whole_spike pour le crop

        Returns:
            Tuple (OBBDetection raffinée, dict segmentation) si réussi, None sinon.
        """
        if sp_det is None or ws_det is None:
            return None
        if self.spike_segmenter is None:
            return None

        # =================================================================
        # 1) Vérification de bordure
        # =================================================================
        if self._spike_border_near_whole_spike(sp_det, ws_det,
                                                    threshold_px=border_threshold_px):
            logger.debug("    Spike OK: bords proches du whole_spike → pas de raffinement")
            return None

        logger.info(f"    Bord du spike loin du whole_spike → lancement du raffinement")
        logger.info(f"    (spike={sp_det.height:.0f}px vs whole_spike={ws_det.height:.0f}px)")

        h_img, w_img = image.shape[:2]

        # =================================================================
        # 2) Crop redressé basé sur le whole_spike OBB
        # =================================================================
        cx, cy = ws_det.center
        rw, rh = ws_det.width, ws_det.height
        angle = ws_det.angle

        out_w = int(rw + 2 * margin)
        out_h = int(rh + 2 * margin)

        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        M[0, 2] += out_w / 2 - cx
        M[1, 2] += out_h / 2 - cy

        crop = cv2.warpAffine(
            image, M, (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # =================================================================
        # 3) Segmentation YOLO-Seg sur le crop whole_spike
        # =================================================================
        results = self.spike_segmenter(crop, conf=self._spike_seg_conf, verbose=False)

        if not results or results[0].masks is None:
            logger.warning("    Raffinement: aucun masque obtenu")
            return None

        res = results[0]
        if res.masks.data is None or len(res.masks.data) == 0:
            logger.warning("    Raffinement: masque vide")
            return None

        best_idx = int(res.boxes.conf.argmax())
        mask_tensor = res.masks.data[best_idx]

        mask_np = mask_tensor.cpu().numpy()
        mask_crop = cv2.resize(
            mask_np, (crop.shape[1], crop.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        mask_crop_bin = (mask_crop > 0.5).astype(np.uint8) * 255

        if cv2.countNonZero(mask_crop_bin) < self._spike_seg_min_area:
            logger.warning("    Raffinement: masque trop petit")
            return None

        # Re-projection vers l'image originale
        M_inv = cv2.invertAffineTransform(M)
        full_mask = cv2.warpAffine(
            mask_crop_bin, M_inv, (w_img, h_img),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.warning("    Raffinement: aucun contour trouvé")
            return None
        seg_contour = max(contours, key=cv2.contourArea)

        area_px = cv2.contourArea(seg_contour)
        if area_px < self._spike_seg_min_area:
            logger.warning("    Raffinement: contour trop petit")
            return None

        if len(seg_contour) < 5:
            logger.warning("    Raffinement: pas assez de points")
            return None

        # =================================================================
        # 4) OBB issu de la segmentation
        # =================================================================
        seg_rect = cv2.minAreaRect(seg_contour)
        (seg_cx, seg_cy), (seg_d1, seg_d2), seg_angle = seg_rect
        if seg_d1 > seg_d2:
            seg_w, seg_h = seg_d2, seg_d1
            seg_angle += 90
        else:
            seg_w, seg_h = seg_d1, seg_d2
        while seg_angle >= 90:
            seg_angle -= 180
        while seg_angle < -90:
            seg_angle += 180

        # =================================================================
        # 5) Consensus detection ↔ segmentation
        # =================================================================
        det_h = sp_det.height
        seg_h_val = seg_h

        if seg_h_val > det_h * 1.10:
            # Segmentation nettement plus grande  → prendre seg
            chosen = 'seg'
            final_contour = seg_contour
            logger.info(f"    Consensus: seg plus grand (seg={seg_h_val:.0f} vs det={det_h:.0f}) → seg retenu")
        elif det_h > seg_h_val * 1.10:
            # Détection nettement plus grande → garder la détection (pas de raffinement)
            logger.info(f"    Consensus: det plus grand (det={det_h:.0f} vs seg={seg_h_val:.0f}) → pas de raffinement")
            return None
        else:
            # Les deux sont proches → union convexe pour consensus
            chosen = 'consensus'
            # Construire les points des deux OBBs et du contour de segmentation
            det_pts = sp_det.obb_points.reshape(-1, 1, 2).astype(np.int32)
            all_pts = np.concatenate([seg_contour, det_pts], axis=0)
            final_contour = cv2.convexHull(all_pts)
            logger.info(f"    Consensus: tailles proches (det={det_h:.0f}, seg={seg_h_val:.0f}) → union convexe")

        # =================================================================
        # 6) Construire le nouvel OBB à partir du contour final
        # =================================================================
        final_rect = cv2.minAreaRect(final_contour)
        (new_cx, new_cy), (dim1, dim2), new_angle = final_rect

        if dim1 > dim2:
            new_w, new_h = dim2, dim1
            new_angle += 90
        else:
            new_w, new_h = dim1, dim2

        while new_angle >= 90:
            new_angle -= 180
        while new_angle < -90:
            new_angle += 180

        new_points = cv2.boxPoints(final_rect).astype(np.float32)

        refined_det = OBBDetection(
            class_id=sp_det.class_id,
            class_name=sp_det.class_name,
            confidence=sp_det.confidence,
            obb_points=new_points,
            center=(float(new_cx), float(new_cy)),
            width=float(new_w),
            height=float(new_h),
            angle=float(new_angle),
            source=sp_det.source,
        )

        # =================================================================
        # 7) Résultat de segmentation (réutilisé à l'étape 5)
        # =================================================================
        # Recalculer le masque si on a utilisé le consensus (union convexe)
        if chosen == 'consensus':
            # On garde le masque seg comme base — l'union convexe n'ajoute
            # que les bords du det OBB, pas de la surface masquée en plus.
            pass

        bx, by, bw, bh = cv2.boundingRect(seg_contour)
        x1, y1 = bx, by
        x2, y2 = bx + bw, by + bh
        roi_mask = full_mask[y1:y2, x1:x2]

        seg_result = {
            'mask': full_mask,
            'roi_mask': roi_mask,
            'roi_bbox': (x1, y1, x2, y2),
            'contour': seg_contour,
            'contour_area_px': float(area_px),
            'contour_perimeter_px': float(cv2.arcLength(seg_contour, True)),
        }

        logger.info(f"    ✓ Spike raffiné ({chosen}): {sp_det.height:.0f}px → {new_h:.0f}px "
                     f"(gain={new_h - sp_det.height:.0f}px, "
                     f"{(new_h/sp_det.height - 1)*100:+.0f}%)")

        return (refined_det, seg_result)
    
    def _load_rachis_detector(self):
        """Charge le détecteur/segmenteur de rachis (YOLO-Seg)"""
        from ultralytics import YOLO
        
        rachis_config = self.config.get('rachis_detection', {})
        
        if not rachis_config.get('enabled', False):
            logger.info("Détection du rachis désactivée dans la config")
            return None
        
        model_path = rachis_config.get('model_path', 'models/rachis_yolo.pt')
        
        if not Path(model_path).exists():
            logger.warning(f"Modèle rachis non trouvé: {model_path}")
            return None
        
        try:
            model = YOLO(model_path)
            logger.info(f"✓ Détecteur rachis chargé: {model_path}")
            return model
        except Exception as e:
            logger.warning(f"Erreur chargement modèle rachis: {e}")
            return None
    
    def _extract_rachis_from_yolo(
        self,
        yolo_results,
        crop: np.ndarray,
        M: np.ndarray,
        full_image_shape: Tuple[int, ...],
        pixel_per_mm: Optional[float],
        precomputed_mask: Optional[np.ndarray] = None,
        precomputed_conf: Optional[float] = None,
    ) -> Optional[Dict]:
        """
        Extrait le masque et la ligne centrale (skeleton) du rachis
        depuis les résultats YOLO-Seg sur un crop redressé.
        
        Args:
            yolo_results: Résultats YOLO sur le crop (ignoré si precomputed_mask fourni)
            crop: Image du crop redressé
            M: Matrice affine utilisée pour le crop (pour inverser)
            full_image_shape: Shape de l'image complète (pour reprojection)
            pixel_per_mm: Ratio px/mm ou None
            precomputed_mask: Masque binaire pré-calculé (ex: consensus TTA), uint8
            precomputed_conf: Confiance associée au masque pré-calculé
            
        Returns:
            Dict avec mask_crop, skeleton_pts_crop, skeleton_pts_global,
            length_px, length_mm, confidence  — ou None si pas de détection
        """
        from skimage.morphology import skeletonize
        
        if precomputed_mask is not None:
            # Utiliser le masque pré-calculé (ex: TTA consensus)
            mask_crop = (precomputed_mask > 127).astype(np.uint8)
            conf = precomputed_conf if precomputed_conf is not None else 0.5
        else:
            res = yolo_results[0]
            if res.masks is None or len(res.masks) == 0:
                return None
            
            # Prendre la détection la plus confiante
            best_idx = int(res.boxes.conf.argmax())
            mask_tensor = res.masks.data[best_idx]
            conf = float(res.boxes.conf[best_idx])
            
            # Convertir en masque binaire à la taille du crop
            mask_np = mask_tensor.cpu().numpy()
            mask_crop = cv2.resize(
                mask_np, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
        
        # Squelettiser le masque pour obtenir la ligne centrale
        skeleton = skeletonize(mask_crop > 0).astype(np.uint8) * 255
        
        # Extraire les points du squelette ordonnés de haut en bas
        pts_y, pts_x = np.where(skeleton > 0)
        if len(pts_y) < 2:
            return None
        
        # Ordonner les points le long du squelette (par y croissant)
        order = np.argsort(pts_y)
        skeleton_pts_crop = np.column_stack([pts_x[order], pts_y[order]])
        
        # Calculer la longueur du squelette (somme des segments)
        diffs = np.diff(skeleton_pts_crop, axis=0)
        segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        length_px = float(segment_lengths.sum())
        
        length_mm = None
        if pixel_per_mm and pixel_per_mm > 0:
            length_mm = length_px / pixel_per_mm
        
        # Reprojeter les points du squelette vers l'image originale
        M_inv = cv2.invertAffineTransform(M)
        pts_hom = np.hstack([
            skeleton_pts_crop.astype(np.float64),
            np.ones((len(skeleton_pts_crop), 1)),
        ])
        skeleton_pts_global = (M_inv @ pts_hom.T).T  # Nx2
        
        # Reprojeter le contour du masque
        contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_contour_global = None
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            pts_c = biggest.reshape(-1, 2).astype(np.float64)
            pts_c_hom = np.hstack([pts_c, np.ones((len(pts_c), 1))])
            mask_contour_global = (M_inv @ pts_c_hom.T).T.astype(np.int32)
        
        return {
            'confidence': conf,
            'mask_crop': mask_crop,
            'skeleton_pts_crop': skeleton_pts_crop,
            'skeleton_pts_global': skeleton_pts_global.astype(np.int32),
            'mask_contour_global': mask_contour_global,
            'length_px': length_px,
            'length_mm': length_mm,
        }
    
    def _compute_spikelet_details_from_yolo_masks(
        self,
        masks_roi: List[np.ndarray],
        bboxes_global: List[Tuple[int, int, int, int]],
        roi_offset: Tuple[int, int],
        spike_mask: Optional[np.ndarray] = None,
        pixel_per_mm: Optional[float] = None,
    ) -> List[Dict]:
        """
        Calcule les métriques morphologiques par épillet à partir des masques YOLO-Seg.
        
        Produit la même structure de données que SpikeSegmenter.segment_spikelets(),
        mais utilise directement les masques YOLO-Seg au lieu de SAM2.
        
        Args:
            masks_roi: Liste de masques binaires (uint8, 255=fg) en coordonnées ROI
            bboxes_global: Bboxes en coordonnées image globale [(x1,y1,x2,y2), ...]
            roi_offset: (ox, oy) offset du ROI dans l'image globale
            spike_mask: Masque de l'épi parent (coordonnées globales, optionnel)
            pixel_per_mm: Calibration (optionnel)
            
        Returns:
            Liste de dicts avec métriques morphologiques par épillet
        """
        ox, oy = roi_offset
        results = []
        
        for idx, (bbox, mask_roi) in enumerate(zip(bboxes_global, masks_roi)):
            gx1, gy1, gx2, gy2 = bbox
            
            spikelet_info = {
                'id': idx + 1,
                'bbox_global': (gx1, gy1, gx2, gy2),
                'center_x': (gx1 + gx2) / 2,
                'center_y': (gy1 + gy2) / 2,
            }
            
            if mask_roi is None or cv2.countNonZero(mask_roi) < 50:
                spikelet_info['segmented'] = False
                bw, bh = gx2 - gx1, gy2 - gy1
                spikelet_info['length_px'] = float(max(bw, bh))
                spikelet_info['width_px'] = float(min(bw, bh))
                if min(bw, bh) > 0:
                    spikelet_info['aspect_ratio'] = float(max(bw, bh) / min(bw, bh))
                if pixel_per_mm and pixel_per_mm > 0:
                    spikelet_info['length_mm'] = max(bw, bh) / pixel_per_mm
                    spikelet_info['width_mm'] = min(bw, bh) / pixel_per_mm
                results.append(spikelet_info)
                continue
            
            # Le mask_roi est aux dimensions du ROI
            h_roi, w_roi = mask_roi.shape[:2]
            
            # Restreindre au masque de l'épi parent si disponible
            if spike_mask is not None:
                spike_roi = spike_mask[oy:oy + h_roi, ox:ox + w_roi]
                if spike_roi.shape == mask_roi.shape:
                    mask_roi = cv2.bitwise_and(mask_roi, spike_roi)
            
            area_px = float(cv2.countNonZero(mask_roi))
            
            if area_px < 50:
                spikelet_info['segmented'] = False
                results.append(spikelet_info)
                continue
            
            spikelet_info['segmented'] = True
            spikelet_info['area_px'] = area_px
            
            # Trouver le contour
            contours, _ = cv2.findContours(
                mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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
                
                # Décaler le contour en coordonnées globales
                cnt_global = cnt.copy()
                cnt_global[:, :, 0] += ox
                cnt_global[:, :, 1] += oy
                spikelet_info['contour'] = cnt_global
                
                # Masque ROI local (zone de la bbox)
                local_x1 = max(0, gx1 - ox)
                local_y1 = max(0, gy1 - oy)
                local_x2 = min(w_roi, gx2 - ox)
                local_y2 = min(h_roi, gy2 - oy)
                if local_x2 > local_x1 and local_y2 > local_y1:
                    spikelet_info['mask_roi'] = mask_roi[local_y1:local_y2, local_x1:local_x2]
                
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
            
            results.append(spikelet_info)
        
        return results
    
    @staticmethod
    def _get_principal_axis(contour: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Calcule l'axe principal (direction du grand axe) d'un contour.
        
        Utilise fitEllipse en priorité (plus robuste que minAreaRect, surtout
        quand l'aspect-ratio est faible). Repli sur minAreaRect si le
        contour a moins de 5 points.
        
        Returns:
            (dx, dy) vecteur unitaire de l'axe principal, ou None si impossible.
        """
        if contour is None or len(contour) < 3:
            return None
        
        if len(contour) >= 5:
            try:
                ell = cv2.fitEllipse(contour)
                (_, _), (ell_w, ell_h), ell_angle = ell
                # ell_angle est l'angle du premier axe (ell_w) par rapport à l'horizontal
                # Le grand axe est celui de max(ell_w, ell_h)
                if ell_w < ell_h:
                    # Le grand axe est le second → ajouter 90°
                    axis_rad = np.radians(ell_angle + 90)
                else:
                    axis_rad = np.radians(ell_angle)
                return (float(np.cos(axis_rad)), float(np.sin(axis_rad)))
            except cv2.error:
                pass
        
        # Repli: minAreaRect
        try:
            rect = cv2.minAreaRect(contour)
            (_, _), (rw, rh), rangle = rect
            if rw > rh:
                axis_rad = np.radians(rangle)
            else:
                axis_rad = np.radians(rangle + 90)
            return (float(np.cos(axis_rad)), float(np.sin(axis_rad)))
        except cv2.error:
            return None
    
    def _compute_spikelet_insertion_angles(
        self,
        image_shape: Tuple[int, ...],
        rachis_info: Dict,
        spikelet_details: List[Dict],
        pixel_per_mm: Optional[float] = None,
        tangent_window: int = 30,
        max_attach_ratio: float = 1.5,
    ) -> List[Dict]:
        """
        Calcule les points de rattachement et angles d'insertion des épillets.
        
        Méthode :
        1. Axe principal de l'épillet = direction du grand axe (fitEllipse).
        2. Point d'attache = intersection de cet axe avec le squelette du
           rachis, recherchée UNIQUEMENT parmi les points du rachis dans un
           voisinage du centre de l'épillet (rayon = max_attach_ratio ×
           longueur de l'épillet). Cela évite les intersections aberrantes
           avec des portions éloignées du rachis.
        3. Le point d'attache exact est la projection orthogonale du point
           rachis candidat sur la droite de l'axe.
        4. Tangente du rachis = PCA locale sur le squelette au point
           d'intersection.
        5. Angle d'insertion = angle entre la tangente et l'axe (0–90°).
        
        Args:
            image_shape: (H, W, C) de l'image originale
            rachis_info: Dict issu de _extract_rachis_from_yolo
            spikelet_details: Liste de dicts avec 'contour', 'center_x/y'
            pixel_per_mm: Ratio px/mm (optionnel)
            tangent_window: Demi-fenêtre (en points) pour la tangente locale
            max_attach_ratio: Rayon de recherche = ratio × longueur épillet
            
        Returns:
            Liste de dicts par épillet avec attachment_point, insertion_angle_deg, etc.
        """
        skeleton_pts = rachis_info.get('skeleton_pts_global')
        mask_contour = rachis_info.get('mask_contour_global')
        
        if skeleton_pts is None or len(skeleton_pts) < 5 or mask_contour is None:
            return []
        
        # Ordonner le squelette le long du rachis (par arc-length)
        skel = self._order_skeleton_by_path(skeleton_pts.astype(np.float64))
        
        # Lisser le squelette avec une spline pour une tangente robuste
        skel_smooth = self._smooth_skeleton(skel, n_out=max(200, len(skel)))
        
        angle_results = []
        
        for sp in spikelet_details:
            angle_info = {
                'spikelet_id': sp.get('id'),
                'attachment_point': None,
                'insertion_angle_deg': None,
                'rachis_tangent': None,
                'spikelet_direction': None,
                'side': None,
            }
            
            if not sp.get('segmented') or 'contour' not in sp:
                angle_results.append(angle_info)
                continue
            
            contour = sp['contour']
            sp_cx = sp.get('center_x', 0)
            sp_cy = sp.get('center_y', 0)
            center = np.array([sp_cx, sp_cy], dtype=np.float64)
            sp_length = sp.get('length_px', 0)
            
            # =================================================================
            # 1. Axe principal de l'épillet (fitEllipse > minAreaRect)
            # =================================================================
            axis_dir = self._get_principal_axis(contour)
            if axis_dir is None:
                angle_results.append(angle_info)
                continue
            
            spikelet_axis = np.array(axis_dir, dtype=np.float64)
            
            # =================================================================
            # 2. Intersection axe épillet × rachis (recherche locale)
            #
            #    Droite de l'axe : P(t) = center + t * spikelet_axis
            #    On cherche le point du squelette le plus proche de cette
            #    droite, mais UNIQUEMENT parmi les points du rachis dans
            #    un rayon raisonnable autour du centre de l'épillet.
            # =================================================================
            d = spikelet_axis  # vecteur unitaire
            
            # Distance euclidienne centre → chaque point du squelette
            v = skel_smooth - center  # (N, 2)
            dist_to_center = np.sqrt((v ** 2).sum(axis=1))  # (N,)
            
            # Rayon de recherche (en pixels)
            search_radius = max(100.0, max_attach_ratio * sp_length) if sp_length > 0 else 200.0
            
            # Masque des points du rachis dans le voisinage
            candidates_mask = dist_to_center <= search_radius
            
            # Distance point-droite pour chaque point du squelette
            t_proj = v[:, 0] * d[0] + v[:, 1] * d[1]  # projection sur l'axe
            proj = np.outer(t_proj, d)  # composante parallèle
            perp = v - proj  # composante perpendiculaire
            dist_to_line = np.sqrt((perp ** 2).sum(axis=1))  # distance à la droite
            
            if candidates_mask.any():
                # Parmi les candidats proches, prendre celui le plus
                # proche de la droite de l'axe (= meilleure intersection)
                score = np.where(candidates_mask, dist_to_line, np.inf)
                skel_proj_idx = int(score.argmin())
            else:
                # Fallback : point du rachis le plus proche du centre
                skel_proj_idx = int(dist_to_center.argmin())
            
            rachis_pt = skel_smooth[skel_proj_idx]
            
            # =================================================================
            # 3. Point d'attache = projection du point rachis sur l'axe
            #
            #    Le point exact d'intersection est la projection orthogonale
            #    du point rachis sur la droite. On vérifie que ce point
            #    reste à distance raisonnable du centre de l'épillet.
            # =================================================================
            t_intersect = t_proj[skel_proj_idx]
            proj_on_axis = center + t_intersect * d
            
            # Distance entre la projection sur l'axe et le point rachis
            proj_error = float(np.linalg.norm(proj_on_axis - rachis_pt))
            proj_dist_from_center = abs(t_intersect)
            
            # Si la projection est cohérente (pas trop loin du rachis ni
            # du centre), l'utiliser. Sinon, garder le point rachis.
            max_proj_error = max(50.0, sp_length * 0.4) if sp_length > 0 else 60.0
            
            if proj_error < max_proj_error and proj_dist_from_center < search_radius:
                attach_pt = proj_on_axis
            else:
                attach_pt = rachis_pt
            
            angle_info['attachment_point'] = (int(round(attach_pt[0])), int(round(attach_pt[1])))
            
            # =================================================================
            # 4. Tangente locale du rachis (PCA sur fenêtre)
            # =================================================================
            half_w = tangent_window
            t_lo = max(0, skel_proj_idx - half_w)
            t_hi = min(len(skel_smooth) - 1, skel_proj_idx + half_w)
            
            if t_hi - t_lo < 3:
                angle_results.append(angle_info)
                continue
            
            segment = skel_smooth[t_lo:t_hi + 1]
            
            mean_pt = segment.mean(axis=0)
            centered_seg = segment - mean_pt
            cov = centered_seg.T @ centered_seg
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            tangent_unit = eigenvectors[:, -1]
            
            # Orienter la tangente dans le sens croissant du squelette
            direction_check = skel_smooth[min(t_hi, len(skel_smooth)-1)] - skel_smooth[t_lo]
            if np.dot(tangent_unit, direction_check) < 0:
                tangent_unit = -tangent_unit
            
            # =================================================================
            # 5. Orienter l'axe épillet vers l'extérieur (du rachis)
            # =================================================================
            outward_vec = center - rachis_pt
            if np.dot(spikelet_axis, outward_vec) < 0:
                spikelet_axis = -spikelet_axis
            
            sp_unit = spikelet_axis / (np.linalg.norm(spikelet_axis) + 1e-9)
            
            # =================================================================
            # 6. Angle d'insertion (0–90°)
            # =================================================================
            cos_angle = np.clip(np.abs(np.dot(tangent_unit, sp_unit)), 0, 1)
            insertion_angle_rad = np.arccos(cos_angle)
            insertion_angle_deg = float(np.degrees(insertion_angle_rad))
            
            # Côté (left/right)
            cross = tangent_unit[0] * sp_unit[1] - tangent_unit[1] * sp_unit[0]
            side = 'left' if cross > 0 else 'right'
            
            angle_info['insertion_angle_deg'] = round(insertion_angle_deg, 1)
            angle_info['rachis_tangent'] = (float(tangent_unit[0]), float(tangent_unit[1]))
            angle_info['spikelet_direction'] = (float(sp_unit[0]), float(sp_unit[1]))
            angle_info['side'] = side
            
            angle_results.append(angle_info)
        
        return angle_results
    
    @staticmethod
    def _order_skeleton_by_path(pts: np.ndarray) -> np.ndarray:
        """
        Ordonne les points d'un squelette le long de leur chemin réel
        en utilisant un parcours nearest-neighbor depuis une extrémité.
        
        Le squelette issu de skeletonize() est trié par y, ce qui ne suit
        pas le chemin quand le rachis ondule. Ce réordonnancement garantit
        un parcours continu point-à-point.
        """
        if len(pts) < 3:
            return pts
        
        n = len(pts)
        visited = np.zeros(n, dtype=bool)
        
        # Démarrer depuis le point avec le plus petit y (sommet)
        start = int(pts[:, 1].argmin())
        order = [start]
        visited[start] = True
        
        for _ in range(n - 1):
            current = order[-1]
            dists = np.sqrt(((pts - pts[current]) ** 2).sum(axis=1))
            dists[visited] = np.inf
            nearest = int(dists.argmin())
            if dists[nearest] == np.inf:
                break
            order.append(nearest)
            visited[nearest] = True
        
        return pts[order]
    
    @staticmethod
    def _smooth_skeleton(pts: np.ndarray, n_out: int = 200) -> np.ndarray:
        """
        Lisse un squelette ordonné avec une spline cubique
        pour obtenir une courbe continue et une tangente régulière.
        """
        from scipy.interpolate import splprep, splev
        
        if len(pts) < 5:
            return pts
        
        # Paramètre d'arc-length cumulé
        diffs = np.diff(pts, axis=0)
        seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
        cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
        total_len = cum_len[-1]
        
        if total_len < 1:
            return pts
        
        # Normaliser pour [0, 1]
        u = cum_len / total_len
        
        # Supprimer les doublons dans u (sinon splprep échoue)
        mask = np.diff(u, prepend=-1) > 1e-8
        u_clean = u[mask]
        pts_clean = pts[mask]
        
        if len(pts_clean) < 5:
            return pts
        
        try:
            tck, _ = splprep([pts_clean[:, 0], pts_clean[:, 1]], u=u_clean, s=total_len * 0.5, k=3)
            u_new = np.linspace(0, 1, n_out)
            x_new, y_new = splev(u_new, tck)
            return np.column_stack([x_new, y_new])
        except Exception:
            return pts
    
    # =========================================================================
    # ÉTAPE 1: DÉTECTION YOLO OBB
    # =========================================================================
    
    def _obb_iou(self, det_a: OBBDetection, det_b: OBBDetection) -> float:
        """Calcule l'IoU entre deux détections OBB via cv2.rotatedRectangleIntersection."""
        rect_a = (det_a.center, (det_a.width, det_a.height), det_a.angle)
        rect_b = (det_b.center, (det_b.width, det_b.height), det_b.angle)
        try:
            ret, pts = cv2.rotatedRectangleIntersection(rect_a, rect_b)
            if ret == cv2.INTERSECT_NONE or pts is None:
                return 0.0
            inter = cv2.contourArea(pts)
            area_a = det_a.width * det_a.height
            area_b = det_b.width * det_b.height
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0
        except Exception:
            return 0.0

    def _obb_containment(self, inner: OBBDetection, outer: OBBDetection) -> float:
        """Calcule le ratio de containment = intersection / area(inner).
        
        Retourne la fraction de `inner` qui est contenue dans `outer`.
        Valeur 1.0 = inner entièrement contenu dans outer.
        Utile pour détecter un spike détecté à l'intérieur d'un whole_spike.
        """
        rect_inner = (inner.center, (inner.width, inner.height), inner.angle)
        rect_outer = (outer.center, (outer.width, outer.height), outer.angle)
        try:
            ret, pts = cv2.rotatedRectangleIntersection(rect_inner, rect_outer)
            if ret == cv2.INTERSECT_NONE or pts is None:
                return 0.0
            inter = cv2.contourArea(pts)
            area_inner = inner.width * inner.height
            return inter / area_inner if area_inner > 0 else 0.0
        except Exception:
            return 0.0

    def _filter_nested_detections(
        self,
        detections: List[OBBDetection],
        containment_threshold: float = 0.6,
        label: str = 'detection',
        max_area_ratio: float = 5.0,
    ) -> List[OBBDetection]:
        """
        Élimine les détections imbriquées de même classe.
        
        Deux cas sont distingués :
        
        1. **Containment bidirectionnel** (les deux détections se chevauchent fortement,
           tailles similaires) : on supprime la plus petite / moins confiante (NMS-like).
           
        2. **Containment unidirectionnel** (une petite détection est à l'intérieur d'une
           grande) : si le ratio d'aire dépasse `max_area_ratio`, la grande détection
           est considérée comme un faux positif englobant → on supprime la GRANDE.
           Sinon, on supprime la petite (doublon).
        
        Cela traite le cas d'un spike détecté à l'intérieur d'un autre spike
        (ou whole_spike dans whole_spike), ce qui est physiquement impossible.
        
        Args:
            detections: Liste de détections de même classe
            containment_threshold: Seuil de containment pour considérer qu'une
                détection est imbriquée dans une autre
            label: Nom de la classe pour le logging
            max_area_ratio: Au-delà de ce ratio d'aire (grand/petit), 
                la grande détection est considérée comme un faux positif englobant
                et est supprimée à la place de la petite.
                
        Returns:
            Liste filtrée sans les détections imbriquées
        """
        if len(detections) <= 1:
            return detections
        
        # Indices à supprimer
        to_remove = set()
        
        for i in range(len(detections)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(detections)):
                if j in to_remove:
                    continue
                
                # Vérifier si i est contenu dans j
                containment_i_in_j = self._obb_containment(detections[i], detections[j])
                # Vérifier si j est contenu dans i
                containment_j_in_i = self._obb_containment(detections[j], detections[i])
                
                either_contained = (containment_i_in_j > containment_threshold or 
                                    containment_j_in_i > containment_threshold)
                
                if not either_contained:
                    continue
                
                area_i = detections[i].width * detections[i].height
                area_j = detections[j].width * detections[j].height
                
                # Identifier le plus grand et le plus petit
                if area_i >= area_j:
                    larger_idx, smaller_idx = i, j
                    larger_area, smaller_area = area_i, area_j
                else:
                    larger_idx, smaller_idx = j, i
                    larger_area, smaller_area = area_j, area_i
                
                both_contained = (containment_i_in_j > containment_threshold and 
                                  containment_j_in_i > containment_threshold)
                area_ratio = larger_area / smaller_area if smaller_area > 0 else float('inf')
                
                if both_contained:
                    # Bidirectionnel : détections très similaires (forte IoU implicite).
                    # Supprimer la plus petite ou la moins confiante (NMS-like).
                    removed_idx = smaller_idx
                    kept_idx = larger_idx
                    reason = "bidirectionnel, suppression du doublon"
                elif area_ratio > max_area_ratio:
                    # Unidirectionnel avec ratio d'aire extrême :
                    # La grande détection est un faux positif englobant → la supprimer.
                    removed_idx = larger_idx
                    kept_idx = smaller_idx
                    reason = f"faux positif englobant (ratio aire={area_ratio:.1f}x > {max_area_ratio}x)"
                else:
                    # Unidirectionnel avec ratio d'aire modéré :
                    # La petite détection est un doublon → la supprimer.
                    removed_idx = smaller_idx
                    kept_idx = larger_idx
                    reason = "doublon contenu"
                
                containment_val = max(containment_i_in_j, containment_j_in_i)
                logger.debug(
                    f"  {label} #{removed_idx} supprimé ({reason}): "
                    f"containment={containment_val:.2f}, "
                    f"area={detections[removed_idx].width * detections[removed_idx].height:.0f} vs "
                    f"{detections[kept_idx].width * detections[kept_idx].height:.0f}, "
                    f"ratio={area_ratio:.1f}x"
                )
                to_remove.add(removed_idx)
        
        filtered = [d for i, d in enumerate(detections) if i not in to_remove]
        n_removed = len(detections) - len(filtered)
        if n_removed > 0:
            logger.info(f"  {n_removed} {label}(s) supprimé(s) car contenu(s) dans un autre {label}")
        
        return filtered

    def _nms_obb(
        self,
        detections: List[OBBDetection],
        iou_threshold: float = 0.45,
        prefer_source: Optional[str] = None,
    ) -> List[OBBDetection]:
        """NMS sur les détections OBB en utilisant IoU des rectangles orientés.
        
        Args:
            detections: Liste de détections
            iou_threshold: Seuil IoU pour suppression
            prefer_source: Si spécifié, les détections avec ce source sont
                prioritaires (triées avant les autres à confiance égale).
                Ex: 'full' pour prioriser les détections full-image.
        """
        if len(detections) <= 1:
            return detections

        # Trier par (source_priority, confidence) décroissante
        def sort_key(d):
            source_prio = 1 if (prefer_source and d.source == prefer_source) else 0
            return (source_prio, d.confidence)
        
        dets = sorted(detections, key=sort_key, reverse=True)
        kept: List[OBBDetection] = []

        while dets:
            best = dets.pop(0)
            kept.append(best)
            remaining = []
            for det in dets:
                if det.class_id != best.class_id:
                    remaining.append(det)
                elif self._obb_iou(best, det) < iou_threshold:
                    remaining.append(det)
                # else: suppressed (same class + high IoU)
            dets = remaining

        return kept

    def _parse_obb_results(self, results, offset_x: float = 0.0, offset_y: float = 0.0) -> List[OBBDetection]:
        """Parse les résultats YOLO OBB en liste d'OBBDetection, avec offset optionnel pour les tuiles.
        
        Args:
            results: Résultats YOLO (liste)
            offset_x, offset_y: Décalage à ajouter aux coordonnées (pour tuiles)
        """
        detections = []
        for result in results:
            if hasattr(result, 'obb') and result.obb is not None:
                obb_data = result.obb.data
                for i in range(len(obb_data)):
                    row = obb_data[i].cpu().numpy()
                    cx, cy, w, h, angle_rad, confidence, class_id = row
                    class_id = int(class_id)

                    # Appliquer l'offset (pour les tuiles)
                    cx = float(cx) + offset_x
                    cy = float(cy) + offset_y
                    
                    angle_deg = float(angle_rad) * 180.0 / np.pi
                    
                    # Calculer les 4 coins avec offset
                    rect = ((cx, cy), (float(w), float(h)), angle_deg)
                    box_pts = cv2.boxPoints(rect).astype(np.float32)

                    length = max(w, h)
                    width = min(w, h)
                    adjusted_angle = angle_deg
                    if w > h:
                        adjusted_angle += 90.0

                    detections.append(OBBDetection(
                        class_id=class_id,
                        class_name=CLASS_NAMES.get(class_id, f"class_{class_id}"),
                        confidence=float(confidence),
                        obb_points=box_pts,
                        center=(cx, cy),
                        width=float(width),
                        height=float(length),
                        angle=adjusted_angle,
                    ))
            
            elif hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    x1 += offset_x; x2 += offset_x
                    y1 += offset_y; y2 += offset_y

                    w = x2 - x1
                    h = y2 - y1
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    box_pts = np.array([
                        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                    ], dtype=np.float32)

                    detections.append(OBBDetection(
                        class_id=class_id,
                        class_name=CLASS_NAMES.get(class_id, f"class_{class_id}"),
                        confidence=float(confidence),
                        obb_points=box_pts,
                        center=(cx, cy),
                        width=float(min(w, h)),
                        height=float(max(w, h)),
                        angle=0.0,
                    ))
        return detections

    def _detect_sliced(
        self,
        image: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
        slice_size: int = 1280,
        overlap_ratio: float = 0.25,
        use_tta: bool = False,
        full_image_imgsz: int = 1024,
    ) -> List[OBBDetection]:
        """Inférence par tuiles (sliced inference) pour les grandes images.
        
        Découpe l'image en tuiles avec chevauchement, lance la détection OBB
        sur chaque tuile, puis fusionne les résultats avec NMS OBB.
        
        Combine automatiquement les résultats de :
        - L'inférence full-image (pour les très grands objets comme la règle)
        - L'inférence par tuiles (pour les détails fins des épis)
        
        Args:
            image: Image BGR complète
            conf_threshold: Seuil de confiance YOLO
            iou_threshold: Seuil IoU pour NMS
            slice_size: Taille de chaque tuile en pixels
            overlap_ratio: Ratio de chevauchement entre tuiles (0-1)
            use_tta: Activer le TTA Ultralytics (augment=True)
        """
        h, w = image.shape[:2]
        full_detections: List[OBBDetection] = []
        tile_detections: List[OBBDetection] = []
        
        # --- 1. Inférence full-image (capture les grands objets) ---
        logger.info(f"  Sliced: inférence full-image ({w}×{h}, imgsz={full_image_imgsz})...")
        full_results = self.detector.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
            half=False,
            augment=use_tta,
            imgsz=full_image_imgsz,
        )
        full_detections = self._parse_obb_results(full_results)
        n_full = len(full_detections)
        for d in full_detections:
            d.source = 'full'
        
        # --- 2. Découper en tuiles et inférer ---
        stride = int(slice_size * (1 - overlap_ratio))
        
        # Calculer les positions des tuiles
        tiles = []
        for y0 in range(0, h, stride):
            for x0 in range(0, w, stride):
                x1 = min(x0, w - slice_size) if x0 + slice_size > w else x0
                y1 = min(y0, h - slice_size) if y0 + slice_size > h else y0
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(x1 + slice_size, w)
                y2 = min(y1 + slice_size, h)
                tiles.append((x1, y1, x2, y2))
        
        # Dédupliquer les tuiles identiques (coins de l'image)
        tiles = list(set(tiles))
        tiles.sort()
        
        logger.info(f"  Sliced: {len(tiles)} tuiles de {slice_size}px (overlap={overlap_ratio:.0%})...")
        
        for tx1, ty1, tx2, ty2 in tiles:
            tile = image[ty1:ty2, tx1:tx2]
            if tile.shape[0] < 32 or tile.shape[1] < 32:
                continue
            
            tile_results = self.detector.predict(
                source=tile,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
                half=False,
                augment=False,  # Pas de TTA sur les tuiles (trop lent)
                imgsz=min(slice_size, 1024),  # Tuiles: résolution adaptée (max 1024)
            )
            
            # Parser avec offset pour ramener en coordonnées globales
            tile_dets = self._parse_obb_results(tile_results, offset_x=float(tx1), offset_y=float(ty1))
            
            # Filtrer les détections dont le centre est trop près du bord
            # (elles seront mieux captées par la tuile voisine)
            edge_margin = slice_size * 0.05  # 5% de marge
            for det in tile_dets:
                local_cx = det.center[0] - tx1
                local_cy = det.center[1] - ty1
                tile_w = tx2 - tx1
                tile_h = ty2 - ty1
                
                # Garder si le centre n'est pas trop au bord (sauf bords de l'image)
                near_left = local_cx < edge_margin and tx1 > 0
                near_right = local_cx > tile_w - edge_margin and tx2 < w
                near_top = local_cy < edge_margin and ty1 > 0
                near_bottom = local_cy > tile_h - edge_margin and ty2 < h
                
                if not (near_left or near_right or near_top or near_bottom):
                    det.source = 'tile'
                    tile_detections.append(det)
        
        n_tile = len(tile_detections)
        
        # Stocker les détections brutes pour debug (avant fusion)
        self._last_raw_full_dets = list(full_detections)
        self._last_raw_tile_dets = list(tile_detections)
        self._last_tile_grid = list(tiles)
        
        # --- 3. Fusion intelligente full + tuiles ---
        merged = self._merge_full_and_tile_detections(
            full_detections, tile_detections, iou_threshold, image_shape=(h, w)
        )
        
        logger.info(f"  Sliced: {n_full} full + {n_tile} tuiles → {len(merged)} après fusion+vérification")
        return merged

    def _merge_full_and_tile_detections(
        self,
        full_dets: List[OBBDetection],
        tile_dets: List[OBBDetection],
        iou_threshold: float,
        image_shape: Tuple[int, int],
    ) -> List[OBBDetection]:
        """Fusionne les détections full-image et tuiles avec priorité full-image.
        
        Stratégie :
        1. Les détections full-image sont prioritaires (ancres fiables)
        2. Pour chaque full-det (spike/whole_spike), les tile-dets qui matchent
           sont utilisées pour ÉTENDRE la bbox si elles capturent un bout manquant
        3. Les tile-dets restantes (non utilisées) sont filtrées par consensus
        4. NMS final avec priorité source='full'
        
        Args:
            full_dets: Détections de l'inférence full-image
            tile_dets: Détections des tuiles (déjà en coordonnées globales)
            iou_threshold: Seuil IoU pour NMS
            image_shape: (height, width) de l'image originale
        """
        sliced_config = self.config.get('yolo', {}).get('sliced_inference', {})
        tile_conf_boost = sliced_config.get('tile_only_min_confidence', 0.5)
        require_multi_tile = sliced_config.get('require_multi_tile_consensus', True)
        extend_enabled = sliced_config.get('extend_full_with_tiles', True)
        extension_min_overlap = sliced_config.get('extension_min_overlap', 0.05)
        extension_max_increase = sliced_config.get('extension_max_increase', 0.5)
        
        h, w = image_shape
        accepted: List[OBBDetection] = []
        used_tile_indices: set = set()
        
        # --- Phase 1: Détections full-image = ancres prioritaires ---
        # Pour spike/whole_spike, essayer d'étendre avec les tuiles
        for fd in full_dets:
            if not extend_enabled or fd.class_id not in (1, 3):
                # Ruler et bag : garder tel quel
                accepted.append(fd)
                continue
            
            # Chercher les tile-dets du même type qui chevauchent cette full-det
            matching_tiles = []
            for i, td in enumerate(tile_dets):
                if i in used_tile_indices:
                    continue
                if td.class_id != fd.class_id:
                    continue
                iou = self._obb_iou(td, fd)
                containment = self._obb_containment(td, fd)
                if iou > extension_min_overlap or containment > 0.3:
                    matching_tiles.append((i, td))
            
            if matching_tiles:
                # Tenter d'étendre l'OBB full avec les tuiles
                extended = self._extend_obb_with_tiles(
                    fd, [t for _, t in matching_tiles],
                    max_increase=extension_max_increase,
                )
                accepted.append(extended)
                for idx, _ in matching_tiles:
                    used_tile_indices.add(idx)
                if extended is not fd:
                    logger.debug(
                        f"  Extension {fd.class_name}: "
                        f"{fd.height:.0f}×{fd.width:.0f} → {extended.height:.0f}×{extended.width:.0f} "
                        f"(+{len(matching_tiles)} tuile(s))"
                    )
            else:
                accepted.append(fd)
        
        # --- Phase 2: Tile-dets restantes (non utilisées pour extension) ---
        remaining_tiles = [td for i, td in enumerate(tile_dets) if i not in used_tile_indices]
        
        # Marquer celles confirmées par une full-det (mais pas utilisées pour extension)
        tile_confirmed = [False] * len(remaining_tiles)
        for i, td in enumerate(remaining_tiles):
            for fd in full_dets:
                if td.class_id == fd.class_id and self._obb_iou(td, fd) > 0.1:
                    tile_confirmed[i] = True
                    break
        
        # Confirmées → ajoutées (NMS les fusionnera avec la full-det)
        for i, td in enumerate(remaining_tiles):
            if tile_confirmed[i]:
                accepted.append(td)
        
        # Non confirmées → consensus multi-tuile ou haute confiance
        unconfirmed = [td for i, td in enumerate(remaining_tiles) if not tile_confirmed[i]]
        
        if require_multi_tile and len(unconfirmed) > 0:
            groups = self._group_detections_by_overlap(unconfirmed, iou_threshold=0.2)
            for group in groups:
                n_tiles_seeing = len(group)
                best = max(group, key=lambda d: d.confidence)
                if n_tiles_seeing >= 2:
                    accepted.append(best)
                    logger.debug(f"  Tile consensus: {best.class_name} conf={best.confidence:.2f} vu par {n_tiles_seeing} tuiles → accepté")
                elif best.confidence >= tile_conf_boost:
                    accepted.append(best)
                    logger.debug(f"  Tile haute conf: {best.class_name} conf={best.confidence:.2f} → accepté")
                else:
                    logger.debug(f"  Tile rejeté: {best.class_name} conf={best.confidence:.2f}, 1 seule tuile → faux positif probable")
        else:
            for td in unconfirmed:
                if td.confidence >= tile_conf_boost:
                    accepted.append(td)
        
        # --- Phase 3: NMS global avec priorité full-image ---
        merged = self._nms_obb(accepted, iou_threshold=iou_threshold, prefer_source='full')
        
        return merged
    
    def _extend_obb_with_tiles(
        self,
        full_det: OBBDetection,
        tile_dets: List[OBBDetection],
        max_increase: float = 0.5,
    ) -> OBBDetection:
        """Étend une détection full-image avec des détections tuiles qui capturent un bout manquant.
        
        Combine les points OBB de la full-det et des tile-dets compatibles,
        puis recalcule le rectangle englobant orienté minimal.
        
        Args:
            full_det: Détection full-image (ancre)
            tile_dets: Détections tuiles du même objet
            max_increase: Augmentation maximale tolérée de l'aire (0.5 = +50%).
                Au-delà, on considère que la tuile capture un objet différent.
                
        Returns:
            OBBDetection étendue, ou full_det inchangée si l'extension n'est pas pertinente
        """
        # Filtrer les tuiles avec un angle compatible (±30°)
        compatible_tiles = []
        for td in tile_dets:
            angle_diff = abs(full_det.angle - td.angle) % 180
            if angle_diff > 90:
                angle_diff = 180 - angle_diff
            if angle_diff < 30:
                compatible_tiles.append(td)
        
        if not compatible_tiles:
            return full_det
        
        # Combiner tous les points OBB
        all_points = [full_det.obb_points.copy()]
        for td in compatible_tiles:
            all_points.append(td.obb_points.copy())
        all_pts = np.vstack(all_points).astype(np.float32)
        
        # Rectangle englobant orienté minimal
        rect = cv2.minAreaRect(all_pts)
        (cx, cy), (w, h), angle = rect
        
        # Convention : height = dimension la plus grande
        length = max(w, h)
        width_val = min(w, h)
        adjusted_angle = angle
        if w > h:
            adjusted_angle += 90.0
        
        # Vérifier que l'extension est raisonnable
        original_area = full_det.width * full_det.height
        new_area = length * width_val
        
        if new_area <= original_area * 1.02:
            # Moins de 2% d'augmentation → pas d'extension significative
            return full_det
        
        area_increase = (new_area - original_area) / original_area
        if area_increase > max_increase:
            # Extension trop importante → les tuiles capturent probablement autre chose
            logger.debug(
                f"  Extension rejetée pour {full_det.class_name}: "
                f"augmentation={area_increase:.0%} > {max_increase:.0%}"
            )
            return full_det
        
        box_pts = cv2.boxPoints(rect).astype(np.float32)
        
        logger.debug(
            f"  Extension acceptée: {full_det.height:.0f}×{full_det.width:.0f} → "
            f"{length:.0f}×{width_val:.0f} (+{area_increase:.0%})"
        )
        
        return OBBDetection(
            class_id=full_det.class_id,
            class_name=full_det.class_name,
            confidence=full_det.confidence,
            obb_points=box_pts,
            center=(float(cx), float(cy)),
            width=float(width_val),
            height=float(length),
            angle=adjusted_angle,
            source='full',  # Garde la priorité full-image
        )

    def _group_detections_by_overlap(
        self, detections: List[OBBDetection], iou_threshold: float = 0.2
    ) -> List[List[OBBDetection]]:
        """Groupe les détections qui se chevauchent (même objet vu par plusieurs tuiles).
        
        Utilise un union-find simplifié basé sur IoU OBB.
        """
        n = len(detections)
        parent = list(range(n))
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        
        for i in range(n):
            for j in range(i + 1, n):
                if detections[i].class_id == detections[j].class_id:
                    if self._obb_iou(detections[i], detections[j]) >= iou_threshold:
                        union(i, j)
        
        groups: Dict[int, List[OBBDetection]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(detections[i])
        
        return list(groups.values())

    def detect_objects_obb(self, image: np.ndarray, use_tta: bool = False) -> Dict[str, List[OBBDetection]]:
        """
        Détecte tous les objets avec YOLO OBB
        
        Supporte deux modes :
        - Standard : inférence sur l'image complète (rapide)
        - Sliced : découpage en tuiles + full-image, puis NMS (meilleur pour épis longs)
        
        Le mode sliced est activé via config.yaml → yolo.sliced_inference.enabled
        
        Args:
            image: Image BGR
            use_tta: Si True, active le TTA intégré Ultralytics (multi-scale)
            
        Returns:
            Dict avec clés 'ruler', 'spikes', 'whole_spikes', 'bags'
        """
        yolo_config = self.config.get('yolo', {})
        conf_threshold = yolo_config.get('confidence_threshold', 0.35)
        iou_threshold = yolo_config.get('iou_threshold', 0.45)
        
        # Vérifier si l'inférence par tuiles est activée
        sliced_config = yolo_config.get('sliced_inference', {})
        use_sliced = sliced_config.get('enabled', False)
        
        if use_sliced:
            slice_size = sliced_config.get('slice_size', 1280)
            overlap_ratio = sliced_config.get('overlap_ratio', 0.25)
            full_image_imgsz = sliced_config.get('full_image_imgsz', 1024)
            
            all_dets = self._detect_sliced(
                image=image,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                slice_size=slice_size,
                overlap_ratio=overlap_ratio,
                use_tta=use_tta,
                full_image_imgsz=full_image_imgsz,
            )
        else:
            # Inférence standard sur l'image complète
            full_imgsz = yolo_config.get('imgsz', 1024)
            results = self.detector.predict(
                source=image,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
                half=False,  # Compatibilité ROCm
                augment=use_tta,  # TTA Ultralytics: multi-scale + flips
                imgsz=full_imgsz,
            )
            all_dets = self._parse_obb_results(results)
        
        # Classer par type
        detections = {
            'ruler': [],
            'spikes': [],
            'whole_spikes': [],
            'bags': [],
        }
        
        for det in all_dets:
            if det.class_id == 0:
                detections['ruler'].append(det)
            elif det.class_id == 1:
                detections['spikes'].append(det)
            elif det.class_id == 2:
                detections['bags'].append(det)
            elif det.class_id == 3:
                detections['whole_spikes'].append(det)
        
        # --- Vérification géométrique post-détection ---
        detections = self._verify_detections(detections, image.shape[:2])
        
        # Log des détections
        mode_label = "SLICED" if use_sliced else "standard"
        logger.info(f"Détections OBB ({mode_label}): "
                   f"ruler={len(detections['ruler'])}, "
                   f"spikes={len(detections['spikes'])}, "
                   f"whole_spikes={len(detections['whole_spikes'])}, "
                   f"bags={len(detections['bags'])}")
        
        return detections
    
    def _verify_detections(
        self, detections: Dict[str, List[OBBDetection]], image_shape: Tuple[int, int]
    ) -> Dict[str, List[OBBDetection]]:
        """Vérifie et filtre les détections pour éliminer les faux positifs.
        
        Applique des filtres géométriques par classe :
        - **Ruler** : aspect ratio ≥ 5, aire minimale, max 1 règle
        - **Spike** : aspect ratio ≥ 1.5, taille min/max, pas de chevauchement total avec ruler
        - **Whole_spike** : aspect ratio ≥ 1.5, taille proportionnée
        - **Bag** : taille min, pas trop petit ni trop grand
        - **Cross-class** : un spike ne doit pas chevaucher une règle ou un sachet
        
        Args:
            detections: Dict des détections classées par type
            image_shape: (height, width) de l'image
        """
        h, w = image_shape
        img_area = h * w
        
        verify_config = self.config.get('yolo', {}).get('detection_verification', {})
        if not verify_config.get('enabled', True):
            return detections
        
        verified = {
            'ruler': [],
            'spikes': [],
            'whole_spikes': [],
            'bags': [],
        }
        
        n_before = {k: len(v) for k, v in detections.items()}
        
        # --- Config des filtres par classe ---
        ruler_cfg = verify_config.get('ruler', {})
        spike_cfg = verify_config.get('spike', {})
        whole_cfg = verify_config.get('whole_spike', {})
        bag_cfg = verify_config.get('bag', {})
        
        # ==== RULER ====
        ruler_min_ar = ruler_cfg.get('min_aspect_ratio', 5.0)
        ruler_min_area_ratio = ruler_cfg.get('min_area_ratio', 0.005)  # 0.5% de l'image
        ruler_max_count = ruler_cfg.get('max_count', 2)
        
        for det in detections['ruler']:
            ar = det.aspect_ratio
            area_ratio = det.area / img_area
            if ar < ruler_min_ar:
                logger.debug(f"  Ruler rejeté: AR={ar:.1f} < {ruler_min_ar}")
                continue
            if area_ratio < ruler_min_area_ratio:
                logger.debug(f"  Ruler rejeté: area_ratio={area_ratio:.4f} < {ruler_min_area_ratio}")
                continue
            verified['ruler'].append(det)
        
        # Garder seulement les N meilleures règles
        verified['ruler'] = sorted(verified['ruler'], key=lambda d: d.confidence, reverse=True)[:ruler_max_count]
        
        # ==== SPIKE ====
        spike_min_ar = spike_cfg.get('min_aspect_ratio', 1.5)
        spike_max_ar = spike_cfg.get('max_aspect_ratio', 25.0)
        spike_min_area_ratio = spike_cfg.get('min_area_ratio', 0.0005)  # 0.05% de l'image
        spike_max_area_ratio = spike_cfg.get('max_area_ratio', 0.15)    # 15% de l'image
        spike_min_height_px = spike_cfg.get('min_height_px', 80)
        
        for det in detections['spikes']:
            ar = det.aspect_ratio
            area_ratio = det.area / img_area
            
            if ar < spike_min_ar:
                logger.debug(f"  Spike rejeté: AR={ar:.1f} < {spike_min_ar}")
                continue
            if ar > spike_max_ar:
                logger.debug(f"  Spike rejeté: AR={ar:.1f} > {spike_max_ar}")
                continue
            if area_ratio < spike_min_area_ratio:
                logger.debug(f"  Spike rejeté: trop petit (area_ratio={area_ratio:.5f})")
                continue
            if area_ratio > spike_max_area_ratio:
                logger.debug(f"  Spike rejeté: trop grand (area_ratio={area_ratio:.4f})")
                continue
            if det.height < spike_min_height_px:
                logger.debug(f"  Spike rejeté: hauteur={det.height:.0f}px < {spike_min_height_px}px")
                continue
            verified['spikes'].append(det)
        
        # ==== WHOLE_SPIKE ====
        # Filtres très permissifs : les whole_spikes sont bien détectés par YOLO,
        # on retire seulement les aberrations évidentes (bruit, artefacts minuscules).
        ws_max_area_ratio = whole_cfg.get('max_area_ratio', 0.50)  # Très permissif (50%)
        ws_min_area_ratio = whole_cfg.get('min_area_ratio', 0.0002)  # Seulement les poussières
        ws_min_height_px = whole_cfg.get('min_height_px', 50)
        
        for det in detections['whole_spikes']:
            area_ratio = det.area / img_area
            
            if area_ratio < ws_min_area_ratio:
                logger.debug(f"  Whole_spike rejeté: trop petit (area_ratio={area_ratio:.5f})")
                continue
            if area_ratio > ws_max_area_ratio:
                logger.debug(f"  Whole_spike rejeté: trop grand (area_ratio={area_ratio:.4f})")
                continue
            if det.height < ws_min_height_px:
                logger.debug(f"  Whole_spike rejeté: hauteur={det.height:.0f}px < {ws_min_height_px}px")
                continue
            verified['whole_spikes'].append(det)
        
        # ==== BAG ====
        bag_min_area_ratio = bag_cfg.get('min_area_ratio', 0.002)
        bag_max_area_ratio = bag_cfg.get('max_area_ratio', 0.25)
        bag_max_count = bag_cfg.get('max_count', 3)
        
        for det in detections['bags']:
            area_ratio = det.area / img_area
            if area_ratio < bag_min_area_ratio:
                logger.debug(f"  Bag rejeté: trop petit (area_ratio={area_ratio:.5f})")
                continue
            if area_ratio > bag_max_area_ratio:
                logger.debug(f"  Bag rejeté: trop grand (area_ratio={area_ratio:.4f})")
                continue
            verified['bags'].append(det)
        
        verified['bags'] = sorted(verified['bags'], key=lambda d: d.confidence, reverse=True)[:bag_max_count]
        
        # ==== CROSS-CLASS : éliminer spikes/whole_spikes qui chevauchent la règle ====
        if verified['ruler']:
            ruler_det = verified['ruler'][0]
            verified['spikes'] = [
                s for s in verified['spikes']
                if self._obb_iou(s, ruler_det) < 0.3
            ]
            verified['whole_spikes'] = [
                s for s in verified['whole_spikes']
                if self._obb_iou(s, ruler_det) < 0.3
            ]
        
        # ==== CROSS-CLASS : éliminer les spikes contenus dans un whole_spike ====
        # Un spike détecté à l'intérieur d'un whole_spike est un doublon (le whole_spike
        # contient déjà l'épi). On utilise le ratio de containment (intersection/area_spike)
        # plutôt que l'IoU classique, car le whole_spike est beaucoup plus grand.
        spike_in_ws_threshold = verify_config.get('spike_inside_whole_spike_threshold', 0.6)
        if verified['whole_spikes']:
            filtered_spikes = []
            for spike in verified['spikes']:
                is_contained = False
                for ws in verified['whole_spikes']:
                    containment = self._obb_containment(spike, ws)
                    if containment > spike_in_ws_threshold:
                        logger.debug(
                            f"  Spike rejeté: contenu dans whole_spike "
                            f"(containment={containment:.2f} > {spike_in_ws_threshold})"
                        )
                        is_contained = True
                        break
                if not is_contained:
                    filtered_spikes.append(spike)
            n_spike_removed = len(verified['spikes']) - len(filtered_spikes)
            if n_spike_removed > 0:
                logger.info(f"  {n_spike_removed} spike(s) supprimé(s) car contenus dans un whole_spike")
            verified['spikes'] = filtered_spikes
        
        # ==== INTRA-CLASS : éliminer les spikes contenus dans d'autres spikes ====
        # Il arrive que le modèle détecte un spike à l'intérieur d'un autre spike,
        # ce qui est physiquement impossible (un épi ne contient pas un autre épi).
        # Deux cas :
        #   - Si le ratio d'aire est extrême (grand englobant petit), le grand est un FP → supprimer le grand.
        #   - Sinon, le petit est un doublon → supprimer le petit.
        spike_in_spike_threshold = verify_config.get('spike_inside_spike_threshold', 0.6)
        nested_max_area_ratio = verify_config.get('nested_max_area_ratio', 5.0)
        verified['spikes'] = self._filter_nested_detections(
            verified['spikes'], spike_in_spike_threshold, label='spike',
            max_area_ratio=nested_max_area_ratio,
        )
        
        # ==== INTRA-CLASS : éliminer les whole_spikes contenus dans d'autres whole_spikes ====
        ws_in_ws_threshold = verify_config.get('whole_spike_inside_whole_spike_threshold', 0.6)
        verified['whole_spikes'] = self._filter_nested_detections(
            verified['whole_spikes'], ws_in_ws_threshold, label='whole_spike',
            max_area_ratio=nested_max_area_ratio,
        )
        
        # Log des rejets
        n_after = {k: len(v) for k, v in verified.items()}
        total_rejected = sum(n_before[k] - n_after[k] for k in n_before)
        if total_rejected > 0:
            details = ", ".join(f"{k}: {n_before[k]}→{n_after[k]}" for k in n_before if n_before[k] != n_after[k])
            logger.info(f"  Vérification: {total_rejected} détection(s) rejetée(s) [{details}]")
        
        return verified
    
    # =========================================================================
    # ÉTAPE 2: CALIBRATION (ratio px/mm)
    # =========================================================================
    
    def calibrate_from_ruler(self, ruler_det: OBBDetection) -> Optional[float]:
        """
        Calcule le ratio pixels/mm depuis l'OBB de la règle (méthode fallback)
        
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
        
        logger.info(f"Calibration (ruler_obb): {ruler_length_px:.1f}px / {ruler_length_mm}mm = "
                   f"{pixel_per_mm:.3f} px/mm")
        
        self.calibration_method = 'ruler_obb'
        self.calibration_details = {
            'method': 'ruler_obb',
            'ruler_length_px': ruler_length_px,
            'ruler_length_mm': ruler_length_mm,
        }
        
        return pixel_per_mm
    
    def detect_graduations(self, image: np.ndarray, ruler_det: OBBDetection) -> List[Dict]:
        """
        Détecte les graduations (0, 10, 20, 30) sur la règle
        
        Args:
            image: Image complète
            ruler_det: Détection OBB de la règle
            
        Returns:
            Liste des graduations détectées avec classe, centre, bbox, confidence
        """
        if self.graduation_detector is None:
            return []
        
        # Extraire la ROI de la règle avec marge
        x1, y1, x2, y2 = ruler_det.bbox
        margin = 30
        h, w = image.shape[:2]
        x1 = max(0, int(x1 - margin))
        y1 = max(0, int(y1 - margin))
        x2 = min(w, int(x2 + margin))
        y2 = min(h, int(y2 + margin))
        
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return []
        
        # Inférence YOLO OBB sur la ROI
        grad_config = self.config.get('graduation_detection', {})
        conf_threshold = grad_config.get('confidence_threshold', 0.3)
        
        results = self.graduation_detector.predict(
            source=roi,
            conf=conf_threshold,
            verbose=False,
        )
        
        graduations = []
        class_names = self.graduation_detector.names  # {0: '0', 1: '10', 2: '20', 3: '30'}
        
        for result in results:
            if hasattr(result, 'obb') and result.obb is not None:
                obb_data = result.obb.data
                for i in range(len(obb_data)):
                    row = obb_data[i].cpu().numpy()
                    cx, cy, ww, hh, angle_rad, confidence, class_id = row
                    class_id = int(class_id)
                    class_name = class_names.get(class_id, str(class_id))
                    
                    # Convertir les coordonnées ROI vers coordonnées image globale
                    global_cx = x1 + float(cx)
                    global_cy = y1 + float(cy)
                    
                    graduations.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'value_mm': int(class_name) * 10 if class_name.isdigit() else None,  # 0->0, 10->100, 20->200, 30->300
                        'center': (global_cx, global_cy),
                        'center_roi': (float(cx), float(cy)),
                        'width': float(ww),
                        'height': float(hh),
                        'angle': float(angle_rad) * 180.0 / np.pi,
                        'confidence': float(confidence),
                        'roi_offset': (x1, y1),
                    })
            elif hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bx1, by1, bx2, by2 = map(float, box.xyxy[0].tolist())
                    cx = (bx1 + bx2) / 2
                    cy = (by1 + by2) / 2
                    class_name = class_names.get(class_id, str(class_id))
                    
                    global_cx = x1 + cx
                    global_cy = y1 + cy
                    
                    graduations.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'value_mm': int(class_name) * 10 if class_name.isdigit() else None,
                        'center': (global_cx, global_cy),
                        'center_roi': (cx, cy),
                        'width': bx2 - bx1,
                        'height': by2 - by1,
                        'angle': 0.0,
                        'confidence': confidence,
                        'roi_offset': (x1, y1),
                    })
        
        # Trier par valeur de graduation
        graduations.sort(key=lambda g: (g['value_mm'] if g['value_mm'] is not None else 999, -g['confidence']))
        
        logger.debug(f"  Graduations détectées: {[g['class_name'] for g in graduations]}")
        
        return graduations
    
    def calibrate_from_graduations(self, image: np.ndarray, ruler_det: OBBDetection) -> Optional[float]:
        """
        Calibration par détection des graduations 0, 10, 20, 30 cm
        
        Priorité:
        1. Distance entre graduations 10, 20, 30 (100mm entre chaque)
        2. Distance entre 0 et une autre graduation
        3. Fallback vers calibrate_from_ruler()
        
        Args:
            image: Image complète
            ruler_det: Détection OBB de la règle
            
        Returns:
            Ratio pixels/mm ou None
        """
        graduations = self.detect_graduations(image, ruler_det)
        
        if len(graduations) < 2:
            logger.info("  Pas assez de graduations détectées, fallback ruler_obb")
            return None
        
        # Créer un dict par valeur de graduation (garder la meilleure confiance)
        grad_by_value = {}
        for g in graduations:
            val = g['value_mm']
            if val is not None:
                if val not in grad_by_value or g['confidence'] > grad_by_value[val]['confidence']:
                    grad_by_value[val] = g
        
        available_values = sorted(grad_by_value.keys())
        logger.debug(f"  Graduations disponibles: {available_values} mm")
        
        # Priorité 1: paires parmi 10, 20, 30 (100mm entre chaque)
        preferred_pairs = [(100, 200), (200, 300), (100, 300)]
        for v1, v2 in preferred_pairs:
            if v1 in grad_by_value and v2 in grad_by_value:
                g1, g2 = grad_by_value[v1], grad_by_value[v2]
                dist_px = np.sqrt((g1['center'][0] - g2['center'][0])**2 + 
                                  (g1['center'][1] - g2['center'][1])**2)
                dist_mm = abs(v2 - v1)
                pixel_per_mm = dist_px / dist_mm
                
                logger.info(f"Calibration (graduations {v1//10}-{v2//10}cm): "
                           f"{dist_px:.1f}px / {dist_mm}mm = {pixel_per_mm:.3f} px/mm")
                
                self.calibration_method = 'graduations'
                self.calibration_details = {
                    'method': 'graduations',
                    'grad_1': {'value_mm': v1, 'center': g1['center'], 'confidence': g1['confidence']},
                    'grad_2': {'value_mm': v2, 'center': g2['center'], 'confidence': g2['confidence']},
                    'distance_px': dist_px,
                    'distance_mm': dist_mm,
                    'all_graduations': graduations,
                }
                return pixel_per_mm
        
        # Priorité 2: utiliser 0 et une autre graduation
        if 0 in grad_by_value:
            for v2 in [300, 200, 100]:
                if v2 in grad_by_value:
                    g1, g2 = grad_by_value[0], grad_by_value[v2]
                    dist_px = np.sqrt((g1['center'][0] - g2['center'][0])**2 + 
                                      (g1['center'][1] - g2['center'][1])**2)
                    dist_mm = v2
                    pixel_per_mm = dist_px / dist_mm
                    
                    logger.info(f"Calibration (graduations 0-{v2//10}cm): "
                               f"{dist_px:.1f}px / {dist_mm}mm = {pixel_per_mm:.3f} px/mm")
                    
                    self.calibration_method = 'graduations'
                    self.calibration_details = {
                        'method': 'graduations',
                        'grad_1': {'value_mm': 0, 'center': g1['center'], 'confidence': g1['confidence']},
                        'grad_2': {'value_mm': v2, 'center': g2['center'], 'confidence': g2['confidence']},
                        'distance_px': dist_px,
                        'distance_mm': dist_mm,
                        'all_graduations': graduations,
                    }
                    return pixel_per_mm
        
        # Priorité 3: n'importe quelle paire
        if len(available_values) >= 2:
            v1, v2 = available_values[0], available_values[-1]
            if v1 != v2:
                g1, g2 = grad_by_value[v1], grad_by_value[v2]
                dist_px = np.sqrt((g1['center'][0] - g2['center'][0])**2 + 
                                  (g1['center'][1] - g2['center'][1])**2)
                dist_mm = abs(v2 - v1)
                if dist_mm > 0:
                    pixel_per_mm = dist_px / dist_mm
                    
                    logger.info(f"Calibration (graduations {v1//10}-{v2//10}cm): "
                               f"{dist_px:.1f}px / {dist_mm}mm = {pixel_per_mm:.3f} px/mm")
                    
                    self.calibration_method = 'graduations'
                    self.calibration_details = {
                        'method': 'graduations',
                        'grad_1': {'value_mm': v1, 'center': g1['center'], 'confidence': g1['confidence']},
                        'grad_2': {'value_mm': v2, 'center': g2['center'], 'confidence': g2['confidence']},
                        'distance_px': dist_px,
                        'distance_mm': dist_mm,
                        'all_graduations': graduations,
                    }
                    return pixel_per_mm
        
        logger.info("  Impossible de calculer la calibration depuis les graduations")
        return None
    
    def save_debug_02b_graduations(self, image: np.ndarray,
                                    ruler_det: Optional[OBBDetection],
                                    graduations: List[Dict],
                                    calibration_details: Dict,
                                    session_dir: Path) -> None:
        """Sauvegarde l'image de debug des graduations détectées"""
        debug_dir = session_dir / 'debug'
        debug_dir.mkdir(exist_ok=True)
        
        viz = image.copy()
        
        # Dessiner la règle
        if ruler_det:
            self._draw_obb(viz, ruler_det, (0, 0, 255), 2)
        
        # Couleurs par classe de graduation
        colors = {
            '0': (0, 255, 0),    # Vert
            '10': (255, 255, 0), # Cyan
            '20': (255, 0, 255), # Magenta
            '30': (0, 165, 255), # Orange
        }
        
        # Dessiner chaque graduation détectée
        for grad in graduations:
            cx, cy = int(grad['center'][0]), int(grad['center'][1])
            class_name = grad['class_name']
            color = colors.get(class_name, (128, 128, 128))
            
            # Cercle au centre
            cv2.circle(viz, (cx, cy), 15, color, 3)
            cv2.circle(viz, (cx, cy), 5, color, -1)
            
            # Label
            label = f"{class_name}cm ({grad['confidence']:.0%})"
            cv2.putText(viz, label, (cx - 30, cy - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
            cv2.putText(viz, label, (cx - 30, cy - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Dessiner la ligne entre les graduations utilisées pour la calibration
        if calibration_details.get('grad_1') and calibration_details.get('grad_2'):
            g1 = calibration_details['grad_1']
            g2 = calibration_details['grad_2']
            pt1 = (int(g1['center'][0]), int(g1['center'][1]))
            pt2 = (int(g2['center'][0]), int(g2['center'][1]))
            cv2.line(viz, pt1, pt2, (0, 255, 0), 3)
            
            # Afficher la distance
            mid_x = (pt1[0] + pt2[0]) // 2
            mid_y = (pt1[1] + pt2[1]) // 2
            dist_text = f"{calibration_details.get('distance_px', 0):.1f}px = {calibration_details.get('distance_mm', 0)}mm"
            cv2.putText(viz, dist_text, (mid_x - 80, mid_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            cv2.putText(viz, dist_text, (mid_x - 80, mid_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Ajouter infos de calibration en haut
        info_lines = [
            f"Méthode: {calibration_details.get('method', 'N/A')}",
            f"Graduations: {len(graduations)} détectées",
        ]
        if calibration_details.get('distance_px'):
            px_per_mm = calibration_details['distance_px'] / calibration_details['distance_mm']
            info_lines.append(f"Calibration: {px_per_mm:.3f} px/mm")
        
        y_offset = 40
        for line in info_lines:
            cv2.putText(viz, line, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            cv2.putText(viz, line, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
            y_offset += 35
        
        cv2.imwrite(str(debug_dir / '02b_graduations.png'), viz)
        logger.debug("  [DEBUG] 02b_graduations.png sauvegardé")
    
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
    
    def count_spikelets(
        self,
        image: np.ndarray,
        spike_det: 'OBBDetection',
        use_tta: bool = False,
        tta_augmentations: Optional[List[str]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compte les épillets dans un épi.
        
        L'épi est d'abord redressé verticalement (warp-affine basé sur
        l'angle OBB) avant la détection, puis les résultats sont
        retransformés dans les coordonnées de l'image originale.
        
        Args:
            image: Image complète
            spike_det: Détection OBB de l'épi
            use_tta: Utiliser le TTA pour le comptage
            tta_augmentations: Augmentations TTA
            mask: Masque global de l'épi (coordonnées image, optionnel)
            
        Returns:
            Dict avec count, positions, confidence
        """
        if self.spikelet_counter is None:
            return {'count': None, 'method': 'unavailable', 'positions': []}
        
        h_img, w_img = image.shape[:2]
        
        # === Créer un crop redressé verticalement (warp-affine) ===
        cx, cy = spike_det.center
        rw, rh = spike_det.width, spike_det.height  # width < height garanti
        angle = spike_det.angle
        margin = 20
        
        out_w = int(rw + 2 * margin)
        out_h = int(rh + 2 * margin)
        
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        M[0, 2] += out_w / 2 - cx
        M[1, 2] += out_h / 2 - cy
        
        vertical_crop = cv2.warpAffine(
            image, M, (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        
        if vertical_crop.size == 0:
            return {'count': None, 'method': 'error', 'positions': []}
        
        # Transformer le masque global de l'épi vers l'espace du crop vertical
        vertical_mask = None
        if mask is not None and mask.size > 0:
            vertical_mask = cv2.warpAffine(
                mask, M, (out_w, out_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        
        # === Détection des épillets sur le crop vertical ===
        result = self.spikelet_counter.count(
            vertical_crop,
            mask=vertical_mask,
            use_tta=use_tta,
            tta_augmentations=tta_augmentations,
        )
        
        count = result.count
        method = result.method + ('+tta' if use_tta else '') + '+vert'
        confidence = result.confidence
        
        if count == 0 or not result.bboxes:
            return {
                'count': count,
                'method': method,
                'confidence': confidence,
                'positions': [],
                'bboxes': [],
                'masks_roi': None,
                'roi_offset': (0, 0),
            }
        
        # === Re-transformation vers les coordonnées originales ===
        M_inv = cv2.invertAffineTransform(M)
        
        # ROI de référence = bbox axis-aligned de l'épi + marge
        roi_x1, roi_y1, roi_x2, roi_y2 = spike_det.bbox
        roi_margin = 10
        roi_x1 = max(0, roi_x1 - roi_margin)
        roi_y1 = max(0, roi_y1 - roi_margin)
        roi_x2 = min(w_img, roi_x2 + roi_margin)
        roi_y2 = min(h_img, roi_y2 + roi_margin)
        
        # Transformer les bboxes et positions vers les coordonnées globales
        positions = []
        global_bboxes = []
        for bx1, by1, bx2, by2 in result.bboxes:
            # Centre → coordonnées globales
            crop_cx = (bx1 + bx2) / 2.0
            crop_cy = (by1 + by2) / 2.0
            pt = M_inv @ np.array([crop_cx, crop_cy, 1.0])
            positions.append((float(pt[0]), float(pt[1])))
            
            # Coins → bbox axis-aligned globale
            corners = np.array([
                [bx1, by1, 1], [bx2, by1, 1],
                [bx1, by2, 1], [bx2, by2, 1],
            ], dtype=float)
            gc = (M_inv @ corners.T).T  # (4, 2)
            gx1 = int(np.clip(gc[:, 0].min(), 0, w_img))
            gy1 = int(np.clip(gc[:, 1].min(), 0, h_img))
            gx2 = int(np.clip(gc[:, 0].max(), 0, w_img))
            gy2 = int(np.clip(gc[:, 1].max(), 0, h_img))
            global_bboxes.append((gx1, gy1, gx2, gy2))
        
        # Transformer les masques vers les coordonnées globales → extraire ROI
        global_masks_roi = None
        if result.masks is not None:
            global_masks_roi = []
            for mask_crop in result.masks:
                if mask_crop is None:
                    global_masks_roi.append(None)
                    continue
                # Projeter le masque dans l'image originale
                full_mask = cv2.warpAffine(
                    mask_crop, M_inv, (w_img, h_img),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                # Extraire la portion ROI
                roi_mask = full_mask[roi_y1:roi_y2, roi_x1:roi_x2]
                global_masks_roi.append(roi_mask)
        
        return {
            'count': count,
            'method': method,
            'confidence': confidence,
            'positions': positions,
            'bboxes': global_bboxes,
            'masks_roi': global_masks_roi,
            'roi_offset': (roi_x1, roi_y1),
        }
    
    # =========================================================================
    # ÉTAPE 5: IDENTIFICATION DU SACHET (OCR)
    # =========================================================================
    
    def identify_bag(
        self, image: np.ndarray, bag_det: OBBDetection,
        use_tta: bool = False, tta_augmentations: Optional[List[str]] = None,
    ) -> Dict:
        """
        Identifie le sachet via OCR des chiffres manuscrits
        
        Args:
            image: Image complète
            bag_det: Détection OBB du sachet
            use_tta: Si True, utilise TTA pour une détection plus robuste
            tta_augmentations: Augmentations TTA (défaut: geometric + photometric)
            
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
            if use_tta:
                # Extraire la ROI du sachet puis utiliser la méthode TTA
                x1, y1, x2, y2 = bbox
                h, w = image.shape[:2]
                margin = 20
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                bag_roi = image[y1:y2, x1:x2]
                
                sample_info = self.bag_digit_detector.detect_sample_id_tta(
                    bag_roi, augmentations=tta_augmentations
                )
                
                # Ajuster les coordonnées des détections
                for det in sample_info.get('detections', []):
                    det['center_x'] += x1
                    det['center_y'] += y1
                    if 'bbox' in det:
                        bx1, by1, bx2, by2 = det['bbox']
                        det['bbox'] = (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)
            else:
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
    
    def save_debug_00_raw_detections(
        self,
        image: np.ndarray,
        session_dir: Path,
    ) -> None:
        """Sauvegarde une image debug montrant TOUTES les détections brutes avant fusion.
        
        Affiche :
        - La grille des tuiles (rectangles gris)
        - Les détections full-image (trait épais + étiquette 'FULL')
        - Les détections tuiles (trait fin + étiquette 'TILE')
        
        Permet de diagnostiquer pourquoi une détection n'est pas étendue ou manquante.
        """
        full_dets = getattr(self, '_last_raw_full_dets', [])
        tile_dets = getattr(self, '_last_raw_tile_dets', [])
        tile_grid = getattr(self, '_last_tile_grid', [])
        
        if not full_dets and not tile_dets:
            return
        
        debug_dir = session_dir / 'debug'
        debug_dir.mkdir(exist_ok=True)
        
        viz = image.copy()
        h, w = viz.shape[:2]
        
        # --- Dessiner la grille des tuiles ---
        for tx1, ty1, tx2, ty2 in tile_grid:
            cv2.rectangle(viz, (tx1, ty1), (tx2, ty2), (80, 80, 80), 1)
        
        # --- Légende des couleurs ---
        colors_full = {
            0: (0, 0, 255),      # ruler = rouge
            1: (0, 200, 0),      # spike = vert foncé
            2: (255, 0, 0),      # bag = bleu
            3: (0, 200, 200),    # whole_spike = jaune foncé
        }
        colors_tile = {
            0: (100, 100, 255),    # ruler = rouge clair
            1: (100, 255, 100),    # spike = vert clair
            2: (255, 100, 100),    # bag = bleu clair
            3: (100, 255, 255),    # whole_spike = jaune clair
        }
        
        # --- Dessiner les détections TILE (en premier, dessous) ---
        for det in tile_dets:
            color = colors_tile.get(det.class_id, (180, 180, 180))
            label = f"TILE {det.class_name} {det.confidence:.0%} {det.height:.0f}x{det.width:.0f}"
            self._draw_obb(viz, det, color, 2, label)
        
        # --- Dessiner les détections FULL (en dernier, dessus, trait épais) ---
        for det in full_dets:
            color = colors_full.get(det.class_id, (200, 200, 200))
            label = f"FULL {det.class_name} {det.confidence:.0%} {det.height:.0f}x{det.width:.0f}"
            self._draw_obb(viz, det, color, 4, label)
        
        # --- Résumé en haut ---
        summary = (
            f"RAW detections: {len(full_dets)} full + {len(tile_dets)} tile | "
            f"Grid: {len(tile_grid)} tiles"
        )
        cv2.putText(viz, summary, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(viz, summary, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # --- Légende ---
        y_leg = 60
        for src_label, colors_map, thickness in [('FULL', colors_full, 4), ('TILE', colors_tile, 2)]:
            for cls_id, color in colors_map.items():
                cls_name = CLASS_NAMES.get(cls_id, f'cls{cls_id}')
                text = f"{src_label} {cls_name}"
                cv2.putText(viz, text, (10, y_leg),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(viz, text, (10, y_leg),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_leg += 22
        
        cv2.imwrite(str(debug_dir / '00_raw_detections.png'), viz)
        logger.debug(f"  [DEBUG] 00_raw_detections.png sauvé ({len(full_dets)} full + {len(tile_dets)} tile)")

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

            # Dessiner les bboxes d'épillets si présentes
            for i, bbox in enumerate(spikelets.get('bboxes', [])):
                try:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(viz, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    # petit label d'index
                    cv2.putText(viz, str(i+1), (x1+2, y1+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                except Exception:
                    continue
        
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
    
    @staticmethod
    def _draw_spike_segmentation(image: np.ndarray, seg: Dict,
                                  color: tuple = (0, 255, 0),
                                  alpha: float = 0.3) -> np.ndarray:
        """Dessine un masque de segmentation sur l'image (sans dépendance SAM2)."""
        viz = image.copy()
        mask = seg['mask']
        contour = seg['contour']
        overlay = viz.copy()
        overlay[mask > 0] = color
        cv2.addWeighted(overlay, alpha, viz, 1 - alpha, 0, viz)
        cv2.drawContours(viz, [contour], -1, color, 2)
        return viz

    def save_debug_05b_segmentation(self, image: np.ndarray,
                                      spike_results: List[Dict],
                                      session_dir: Path) -> None:
        """Sauvegarde l'image de debug de la segmentation des épis."""
        debug_dir = session_dir / 'debug'
        debug_dir.mkdir(exist_ok=True)
        
        viz = image.copy()
        
        spike_colors = [
            (0, 255, 0),     # Vert
            (255, 165, 0),   # Orange
            (0, 255, 255),   # Jaune
            (255, 0, 255),   # Magenta
            (0, 165, 255),   # Orange clair
            (255, 255, 0),   # Cyan
        ]
        
        segmented_count = 0
        
        for i, result in enumerate(spike_results):
            seg = result.get('segmentation')
            if seg is None:
                continue
            
            segmented_count += 1
            color = spike_colors[i % len(spike_colors)]
            
            # Superposer le masque
            viz = self._draw_spike_segmentation(viz, seg, color, alpha=0.25)
            
            # Annoter avec les métriques
            contour = seg['contour']
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                cx, cy = int(ellipse[0][0]), int(ellipse[0][1])
            else:
                cx, cy = int(result['detection'].center[0]), int(result['detection'].center[1])
            
            m = result.get('measurements', {})
            lines = [f"#{result['id']}"]
            
            if m.get('real_area_mm2'):
                lines.append(f"Aire:{m['real_area_mm2']:.1f}mm²")
            if m.get('circularity'):
                lines.append(f"Circ:{m['circularity']:.3f}")
            if m.get('solidity'):
                lines.append(f"Solid:{m['solidity']:.3f}")
            
            wp = m.get('width_profile', {})
            if wp.get('shape_class'):
                lines.append(f"Forme:{wp['shape_class']}")
            
            col = m.get('color', {})
            if col.get('hue_mean') is not None:
                lines.append(f"H:{col['hue_mean']:.0f} S:{col['saturation_mean']:.0f}")
            
            y_offset = cy - 30
            for line in lines:
                cv2.putText(viz, line, (cx - 50, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                cv2.putText(viz, line, (cx - 50, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                y_offset += 16
        
        # Header
        header = f"Segmentation YOLO-Seg: {segmented_count}/{len(spike_results)} épis segmentés"
        cv2.putText(viz, header, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        cv2.putText(viz, header, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        
        cv2.imwrite(str(debug_dir / '05b_segmentation.png'), viz)
        logger.debug("  [DEBUG] 05b_segmentation.png sauvegardé")
    
    def save_debug_06_spikelets(self, image: np.ndarray,
                                  spike_results: List[Dict],
                                  session_dir: Path) -> None:
        """Sauvegarde l'image de debug de la segmentation des épillets"""
        debug_dir = session_dir / 'debug'
        debug_dir.mkdir(exist_ok=True)
        
        viz = image.copy()
        
        # Couleurs pour différencier les épis
        spike_colors = [
            (0, 255, 0), (255, 165, 0), (0, 255, 255),
            (255, 0, 255), (0, 165, 255), (255, 255, 0),
        ]
        
        total_spikelets = 0
        segmented_spikelets = 0
        
        for i, result in enumerate(spike_results):
            details = result.get('spikelet_details', [])
            if not details:
                continue
            
            color = spike_colors[i % len(spike_colors)]
            
            for sp in details:
                bbox = sp.get('bbox_global')
                if bbox is None:
                    continue
                
                total_spikelets += 1
                gx1, gy1, gx2, gy2 = [int(v) for v in bbox]
                
                if sp.get('segmented') and 'contour' in sp:
                    segmented_spikelets += 1
                    # Dessiner le contour de segmentation
                    cv2.drawContours(viz, [sp['contour']], -1, color, 2)
                    
                    # Dessiner l'axe principal du polygone (jaune)
                    cnt = sp['contour']
                    axis_dir = self._get_principal_axis(cnt)
                    if axis_dir is not None:
                        dx, dy = axis_dir
                        rcx = sp.get('center_x', (gx1 + gx2) / 2)
                        rcy = sp.get('center_y', (gy1 + gy2) / 2)
                        rect = cv2.minAreaRect(cnt)
                        half_len = max(rect[1][0], rect[1][1]) / 2 + 10
                        ax_p1 = (int(rcx - dx * half_len), int(rcy - dy * half_len))
                        ax_p2 = (int(rcx + dx * half_len), int(rcy + dy * half_len))
                        cv2.line(viz, ax_p1, ax_p2, (0, 255, 255), 2)
                    
                    # Annotation de taille
                    cx = int(sp.get('center_x', (gx1 + gx2) / 2))
                    cy = int(sp.get('center_y', (gy1 + gy2) / 2))
                    
                    label_parts = [f"#{sp['id']}"]
                    if sp.get('length_mm'):
                        label_parts.append(f"{sp['length_mm']:.1f}x{sp.get('width_mm', 0):.1f}mm")
                    elif sp.get('length_px'):
                        label_parts.append(f"{sp['length_px']:.0f}x{sp.get('width_px', 0):.0f}px")
                    
                    label = " ".join(label_parts)
                    cv2.putText(viz, label, (gx1, gy1 - 4),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 2)
                    cv2.putText(viz, label, (gx1, gy1 - 4),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                else:
                    # Pas segmenté : uniquement la bbox en pointillé
                    cv2.rectangle(viz, (gx1, gy1), (gx2, gy2), color, 1)
            
            # Stats globales pour cet épi
            stats = result.get('measurements', {}).get('spikelet_stats', {})
            if stats.get('n_segmented', 0) > 0:
                det = result.get('detection')
                if det:
                    tx, ty = int(det.center[0]), int(det.center[1]) + 30
                    mean_l = stats.get('spikelet_length_mm_mean', stats.get('spikelet_length_px_mean', 0))
                    unit = 'mm' if 'spikelet_length_mm_mean' in stats else 'px'
                    info = f"Epi#{result['id']}: L_moy={mean_l:.1f}{unit}, CV={stats.get('spikelet_length_cv', 0):.2f}"
                    cv2.putText(viz, info, (tx - 80, ty),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                    cv2.putText(viz, info, (tx - 80, ty),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        
        # Header
        header = f"Spikelets: {segmented_spikelets}/{total_spikelets} segmentés"
        cv2.putText(viz, header, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        cv2.putText(viz, header, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)
        
        cv2.imwrite(str(debug_dir / '06_spikelets.png'), viz)
        logger.debug("  [DEBUG] 06_spikelets.png sauvegardé")
    
    def save_debug_07_rachis(self, image: np.ndarray,
                              spike_results: List[Dict],
                              session_dir: Path) -> None:
        """Sauvegarde l'image de debug de la détection du rachis"""
        debug_dir = session_dir / 'debug'
        debug_dir.mkdir(exist_ok=True)
        
        viz = image.copy()
        
        spike_colors = [
            (0, 255, 0), (255, 165, 0), (0, 255, 255),
            (255, 0, 255), (0, 165, 255), (255, 255, 0),
        ]
        
        detected_count = 0
        total_count = len(spike_results)
        
        for i, result in enumerate(spike_results):
            rachis = result.get('rachis')
            color = spike_colors[i % len(spike_colors)]
            
            # Dessiner l'OBB de l'épi (en fin)
            det = result.get('detection')
            if det is not None:
                pts = det.obb_points.astype(np.int32)
                cv2.polylines(viz, [pts], True, color, 1)
            
            if rachis is None:
                continue
            
            detected_count += 1
            
            # Dessiner le contour du masque en semi-transparent
            contour = rachis.get('mask_contour_global')
            if contour is not None:
                overlay = viz.copy()
                cv2.fillPoly(overlay, [contour.reshape(-1, 1, 2)], color)
                cv2.addWeighted(overlay, 0.3, viz, 0.7, 0, viz)
                cv2.drawContours(viz, [contour.reshape(-1, 1, 2)], -1, color, 2)
            
            # Dessiner la ligne centrale (skeleton)
            skeleton_pts = rachis.get('skeleton_pts_global')
            if skeleton_pts is not None and len(skeleton_pts) > 1:
                # Sous-échantillonner pour un tracé plus propre
                step = max(1, len(skeleton_pts) // 200)
                pts_draw = skeleton_pts[::step]
                for j in range(len(pts_draw) - 1):
                    pt1 = tuple(pts_draw[j].astype(int))
                    pt2 = tuple(pts_draw[j + 1].astype(int))
                    cv2.line(viz, pt1, pt2, (0, 0, 255), 2)
            
            # Annotation texte
            if det is not None:
                tx = int(det.center[0])
                ty = int(det.center[1]) - 15
                length = rachis.get('length_mm') or rachis.get('length_px', 0)
                unit = 'mm' if rachis.get('length_mm') else 'px'
                conf = rachis.get('confidence', 0)
                label = f"Epi#{result['id']} rachis: {length:.1f}{unit} ({conf:.0%})"
                cv2.putText(viz, label, (tx - 100, ty),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                cv2.putText(viz, label, (tx - 100, ty),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        
        # Header
        header = f"Rachis: {detected_count}/{total_count} détectés"
        cv2.putText(viz, header, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        cv2.putText(viz, header, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
        
        cv2.imwrite(str(debug_dir / '07_rachis.png'), viz)
        logger.debug("  [DEBUG] 07_rachis.png sauvegardé")
    
    def save_debug_08_insertion_angles(self, image: np.ndarray,
                                        spike_results: List[Dict],
                                        session_dir: Path) -> None:
        """Sauvegarde l'image de debug des angles d'insertion des épillets.
        
        Affiche pour chaque épi :
        - Le squelette du rachis (rouge)
        - Les contours des épillets (couleur par épi)
        - Un point au rattachement épillet ↔ rachis (jaune)
        - Une ligne depuis l'attachment vers le centre de l'épillet (cyan)
        - L'angle en texte
        """
        debug_dir = session_dir / 'debug'
        debug_dir.mkdir(exist_ok=True)
        
        viz = image.copy()
        
        spike_colors = [
            (0, 255, 0), (255, 165, 0), (0, 255, 255),
            (255, 0, 255), (0, 165, 255), (255, 255, 0),
        ]
        
        total_angles = 0
        computed_angles = 0
        angle_values = []
        
        for i, result in enumerate(spike_results):
            rachis = result.get('rachis')
            angles = result.get('insertion_angles', [])
            details = result.get('spikelet_details', [])
            color = spike_colors[i % len(spike_colors)]
            
            # Dessiner l'OBB de l'épi
            det = result.get('detection')
            if det is not None:
                pts = det.obb_points.astype(np.int32)
                cv2.polylines(viz, [pts], True, color, 1)
            
            # Dessiner le squelette du rachis
            if rachis is not None:
                skeleton_pts = rachis.get('skeleton_pts_global')
                if skeleton_pts is not None and len(skeleton_pts) > 1:
                    step = max(1, len(skeleton_pts) // 200)
                    pts_draw = skeleton_pts[::step]
                    for j in range(len(pts_draw) - 1):
                        pt1 = tuple(pts_draw[j].astype(int))
                        pt2 = tuple(pts_draw[j + 1].astype(int))
                        cv2.line(viz, pt1, pt2, (0, 0, 255), 2)
            
            # Dessiner chaque épillet + angle
            for sp, angle_info in zip(details, angles):
                total_angles += 1
                
                # Contour de l'épillet
                if sp.get('segmented') and 'contour' in sp:
                    cv2.drawContours(viz, [sp['contour']], -1, color, 1)
                
                attach = angle_info.get('attachment_point')
                angle_deg = angle_info.get('insertion_angle_deg')
                
                if attach is None:
                    continue
                
                # Point de rattachement (cercle jaune)
                cv2.circle(viz, tuple(attach), 6, (0, 255, 255), -1)
                cv2.circle(viz, tuple(attach), 6, (0, 0, 0), 1)
                
                # Ligne vers le centre de l'épillet (cyan)
                sp_center = (int(sp.get('center_x', 0)), int(sp.get('center_y', 0)))
                cv2.line(viz, tuple(attach), sp_center, (255, 255, 0), 2)
                
                # Dessiner la tangente du rachis (segment court, rouge)
                tangent = angle_info.get('rachis_tangent')
                if tangent is not None:
                    t_len = 40  # longueur du segment tangent affiché
                    t_dx, t_dy = tangent
                    t_pt1 = (int(attach[0] - t_dx * t_len), int(attach[1] - t_dy * t_len))
                    t_pt2 = (int(attach[0] + t_dx * t_len), int(attach[1] + t_dy * t_len))
                    cv2.line(viz, t_pt1, t_pt2, (0, 0, 200), 2)
                
                if angle_deg is not None:
                    computed_angles += 1
                    angle_values.append(angle_deg)
                    
                    side = angle_info.get('side', '')
                    label = f"{angle_deg:.0f}° {side[0].upper()}" if side else f"{angle_deg:.0f}°"
                    
                    # Position du texte : milieu entre attachment et centre
                    tx = (attach[0] + sp_center[0]) // 2
                    ty = (attach[1] + sp_center[1]) // 2 - 8
                    cv2.putText(viz, label, (tx, ty),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 2)
                    cv2.putText(viz, label, (tx, ty),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1)
            
            # Annotation globale par épi
            if det is not None:
                valid = [a['insertion_angle_deg'] for a in angles if a.get('insertion_angle_deg') is not None]
                if valid:
                    info = (f"Epi#{result['id']}: angle_moy={np.mean(valid):.1f}° "
                            f"± {np.std(valid):.1f}°")
                    tx = int(det.center[0]) - 100
                    ty = int(det.center[1]) + 40
                    cv2.putText(viz, info, (tx, ty),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                    cv2.putText(viz, info, (tx, ty),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        
        # Header
        if angle_values:
            header = (f"Insertion angles: {computed_angles}/{total_angles} calculés, "
                      f"moy={np.mean(angle_values):.1f}° ± {np.std(angle_values):.1f}°")
        else:
            header = f"Insertion angles: {computed_angles}/{total_angles} calculés"
        cv2.putText(viz, header, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        cv2.putText(viz, header, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)
        
        cv2.imwrite(str(debug_dir / '08_insertion_angles.png'), viz)
        logger.debug("  [DEBUG] 08_insertion_angles.png sauvegardé")
    
    def save_debug_05_final(self, image: np.ndarray,
                             results: Dict,
                             spike_results: List[Dict],
                             detections: Dict[str, List[OBBDetection]],
                             session_dir: Path) -> None:
        """Sauvegarde l'image finale annotée avec toutes les OBB
        
        En mode low-debug (debug_level=1): uniquement result_annotated.png
        En mode full-debug (debug_level=2): result_annotated.png + debug/05_final_result.png
        """
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

        # Dessiner les positions et bboxes d'épillets pour tous les épis
        for result in spike_results:
            spikelets = result.get('spikelets', {}) or {}
            # positions
            for pos in spikelets.get('positions', []):
                try:
                    px, py = int(pos[0]), int(pos[1])
                    cv2.circle(viz, (px, py), 5, (255, 0, 255), -1)
                except Exception:
                    continue
            # bboxes
            for i, bbox in enumerate(spikelets.get('bboxes', [])):
                try:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(viz, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(viz, str(i+1), (x1+2, y1+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                except Exception:
                    continue
        
        # 3. Dessiner le rachis et les angles d'insertion
        for result in spike_results:
            rachis = result.get('rachis')
            if rachis is not None:
                # Contour du rachis (semi-transparent rouge)
                contour = rachis.get('mask_contour_global')
                if contour is not None:
                    overlay = viz.copy()
                    cv2.fillPoly(overlay, [contour.reshape(-1, 1, 2)], (0, 0, 200))
                    cv2.addWeighted(overlay, 0.25, viz, 0.75, 0, viz)
                    cv2.drawContours(viz, [contour.reshape(-1, 1, 2)], -1, (0, 0, 200), 2)
                
                # Squelette du rachis (rouge)
                skeleton_pts = rachis.get('skeleton_pts_global')
                if skeleton_pts is not None and len(skeleton_pts) > 1:
                    step = max(1, len(skeleton_pts) // 200)
                    pts_draw = skeleton_pts[::step]
                    for j in range(len(pts_draw) - 1):
                        pt1 = tuple(pts_draw[j].astype(int))
                        pt2 = tuple(pts_draw[j + 1].astype(int))
                        cv2.line(viz, pt1, pt2, (0, 0, 255), 2)
            
            # Lignes d'insertion des épillets (bleu)
            angles = result.get('insertion_angles', [])
            details = result.get('spikelet_details', [])
            for sp, angle_info in zip(details, angles):
                attach = angle_info.get('attachment_point')
                if attach is None:
                    continue
                sp_center = (int(sp.get('center_x', 0)), int(sp.get('center_y', 0)))
                # Ligne bleue : attachment → centre épillet
                cv2.line(viz, tuple(attach), sp_center, (255, 150, 0), 2)
                # Point de rattachement (jaune)
                cv2.circle(viz, tuple(attach), 4, (0, 255, 255), -1)
                # Angle (petit texte)
                angle_deg = angle_info.get('insertion_angle_deg')
                if angle_deg is not None:
                    mid_x = (attach[0] + sp_center[0]) // 2
                    mid_y = (attach[1] + sp_center[1]) // 2 - 5
                    cv2.putText(viz, f"{angle_deg:.0f}", (mid_x, mid_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 2)
                    cv2.putText(viz, f"{angle_deg:.0f}", (mid_x, mid_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 150, 0), 1)
        
        # 4. Dessiner le sachet
        for bag in detections.get('bags', []):
            self._draw_obb(viz, bag, (255, 0, 0), 2)
        
        # 5. Ajouter les infos en superposition
        # Calibration
        if results.get('calibration', {}).get('ruler_detected'):
            cal_method = results['calibration'].get('calibration_method', 'ruler_obb')
            px_mm = results['calibration']['pixel_per_mm']
            cal_text = f"Calibration: {px_mm:.2f} px/mm ({cal_method})"
            cv2.putText(viz, cal_text,
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            cv2.putText(viz, cal_text,
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Dessiner les graduations utilisées si méthode graduations
            grad_info = results['calibration'].get('graduations_used')
            if grad_info and grad_info.get('grad_1') and grad_info.get('grad_2'):
                g1 = grad_info['grad_1']
                g2 = grad_info['grad_2']
                pt1 = (int(g1['center'][0]), int(g1['center'][1]))
                pt2 = (int(g2['center'][0]), int(g2['center'][1]))
                cv2.line(viz, pt1, pt2, (0, 255, 0), 2)
                cv2.circle(viz, pt1, 10, (0, 255, 0), -1)
                cv2.circle(viz, pt2, 10, (0, 255, 0), -1)
                # Labels
                cv2.putText(viz, f"{g1['value_mm']//10}cm", (pt1[0]-15, pt1[1]-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(viz, f"{g2['value_mm']//10}cm", (pt2[0]-15, pt2[1]-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
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
        cv2.rectangle(viz, (320, legend_y - 12), (340, legend_y + 2), (0, 0, 255), -1)
        cv2.putText(viz, "rachis", (345, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(viz, (410, legend_y - 12), (430, legend_y + 2), (255, 150, 0), -1)
        cv2.putText(viz, "insertion", (435, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Toujours sauvegarder result_annotated.png
        cv2.imwrite(str(session_dir / 'result_annotated.png'), viz)
        
        # En mode full-debug, sauvegarder aussi dans le dossier debug
        if self.debug_level >= 2:
            debug_dir = session_dir / 'debug'
            debug_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(debug_dir / '05_final_result.png'), viz)
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
            tta_obb_label = " + TTA" if self.use_tta_obb else ""
            logger.info(f"[1/10] Détection YOLO OBB{tta_obb_label}...")
            detections = self.detect_objects_obb(image, use_tta=self.use_tta_obb)
            
            if self.debug_level >= 2:
                self.save_debug_00_raw_detections(image, session_dir)
                self.save_debug_01_detections(image, detections, session_dir)
            
            # =====================================================================
            # ÉTAPE 2: CALIBRATION
            # =====================================================================
            logger.info("[2/10] Calibration...")
            ruler_det = None
            self.pixel_per_mm = None
            self.calibration_method = None
            self.calibration_details = {}
            graduations = []
            
            if detections['ruler']:
                # Prendre la règle avec la meilleure confiance
                ruler_det = max(detections['ruler'], key=lambda d: d.confidence)
                
                # Priorité 1: calibration par graduations (méthode principale)
                if self.graduation_detector is not None:
                    self.pixel_per_mm = self.calibrate_from_graduations(image, ruler_det)
                    if self.pixel_per_mm:
                        graduations = self.calibration_details.get('all_graduations', [])
                
                # Fallback: calibration par longueur de règle OBB
                if self.pixel_per_mm is None:
                    self.pixel_per_mm = self.calibrate_from_ruler(ruler_det)
            else:
                logger.warning("Aucune règle détectée - mesures en pixels uniquement")
            
            if self.debug_level >= 2:
                self.save_debug_02_calibration(image, ruler_det, self.pixel_per_mm, session_dir)
                if graduations or self.graduation_detector is not None:
                    self.save_debug_02b_graduations(image, ruler_det, graduations, self.calibration_details, session_dir)
            
            # =====================================================================
            # ÉTAPE 3: APPARIEMENT SPIKE ↔ WHOLE_SPIKE (Hungarian matching)
            # =====================================================================
            logger.info("[3/10] Appariement spike ↔ whole_spike...")
            
            # Appariement robuste via IoU OBB + algorithme hongrois
            pairs = match_spikes_hungarian(
                spikes=detections['spikes'],
                whole_spikes=detections['whole_spikes'],
                min_iou=self.matching_min_iou,
                min_containment=self.matching_min_containment,
            )
            
            # Construire la liste all_spike_dets à partir des paires
            all_spike_dets = []
            for ws_idx, sp_idx in pairs:
                ws_det = detections['whole_spikes'][ws_idx] if ws_idx is not None else None
                sp_det = detections['spikes'][sp_idx] if sp_idx is not None else None
                all_spike_dets.append((ws_det, sp_det))
            
            # =====================================================================
            # ÉTAPE 3.5: RAFFINEMENT DES SPIKES INCOMPLETS
            # =====================================================================
            # Si un bord de la bbox du spike est proche d'un bord du whole_spike,
            # le spike est probablement coupé. On relance la segmentation sur la
            # zone whole_spike et on compare (consensus) avec la détection OBB.
            refine_config = self.config.get('yolo', {}).get('spike_refinement', {})
            refine_enabled = refine_config.get('enabled', True)
            refine_border_px = refine_config.get('border_threshold_px', 40)
            refine_margin = refine_config.get('margin', 60)
            
            # Dict pour stocker les segmentations issues du raffinement
            # (clé = index dans all_spike_dets, valeur = seg_result dict)
            refinement_seg_results = {}
            
            if refine_enabled:
                n_refined = 0
                for i, (ws_det, sp_det) in enumerate(all_spike_dets):
                    if sp_det is None or ws_det is None:
                        continue
                    
                    result = self._refine_spike_with_whole_spike(
                        image, sp_det, ws_det,
                        border_threshold_px=refine_border_px,
                        margin=refine_margin,
                    )
                    
                    if result is not None:
                        refined_det, seg_result = result
                        # Remplacer la détection spike par la version raffinée
                        all_spike_dets[i] = (ws_det, refined_det)
                        # Stocker la segmentation pour réutilisation à l'étape 5
                        refinement_seg_results[i] = seg_result
                        # Mettre à jour aussi la liste globale des spikes
                        if sp_det in detections['spikes']:
                            idx_in_spikes = detections['spikes'].index(sp_det)
                            detections['spikes'][idx_in_spikes] = refined_det
                        n_refined += 1
                
                if n_refined > 0:
                    logger.info(f"  {n_refined} spike(s) raffiné(s) via segmentation whole_spike")
            
            # =====================================================================
            # ÉTAPE 4: MESURES MORPHOMÉTRIQUES DES ÉPIS
            # =====================================================================
            logger.info("[4/10] Mesures morphométriques...")
            
            spike_results = []
            for idx, (ws_det, sp_det) in enumerate(all_spike_dets):
                # Mesurer avec le système qui calcule les barbes
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
                    'segmentation': refinement_seg_results.get(idx),  # pré-rempli si raffiné
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
            # ÉTAPE 5: SEGMENTATION DES ÉPIS (YOLO-Seg)
            # =====================================================================
            if self.spike_segmenter is not None:
                tta_seg_label = " + TTA" if self.use_tta_spike_seg else ""
                logger.info(f"[5/10] Segmentation YOLO-Seg des épis{tta_seg_label}...")
                metrics = self._spike_metrics_helper
                
                n_segmented = 0
                for result in spike_results:
                    det = result['detection']
                    if det is None:
                        continue
                    
                    # Si la segmentation existe déjà (issue du raffinement step 3.5),
                    # on la réutilise directement au lieu de relancer le modèle
                    seg_result = result.get('segmentation')
                    if seg_result is not None:
                        logger.debug(f"  Épi #{result['id']}: segmentation réutilisée (raffinement)")
                    else:
                        seg_result = self._segment_spike_yolo(
                            image, det,
                            use_tta=self.use_tta_spike_seg,
                            tta_augmentations=self.tta_spike_seg_augs if self.use_tta_spike_seg else None,
                        )
                    
                    if seg_result is not None:
                        result['segmentation'] = seg_result
                        n_segmented += 1
                        
                        # Calculer les métriques du contour réel
                        contour_metrics = metrics.compute_contour_metrics(
                            seg_result, self.pixel_per_mm
                        )
                        result['measurements'].update(contour_metrics)
                        
                        # Longueur et largeur basées sur le squelette morphologique
                        skel_metrics = metrics.compute_skeleton_length_width(
                            seg_result, self.pixel_per_mm
                        )
                        skel_metrics.pop('_skeleton_img', None)
                        result['measurements'].update(skel_metrics)
                        
                        # Profil de largeur
                        width_profile = metrics.compute_width_profile(
                            seg_result, self.pixel_per_mm
                        )
                        result['measurements']['width_profile'] = width_profile
                        
                        # Couleur
                        color_stats = metrics.compute_color_stats(
                            image, seg_result
                        )
                        result['measurements']['color'] = color_stats
                        
                        area_val = contour_metrics.get('real_area_mm2')
                        area_str = f"{area_val:.1f}mm²" if area_val is not None else f"{contour_metrics.get('contour_area_px', 0):.0f}px²"
                        log_parts = [
                            f"  Épi #{result['id']}: ",
                            f"aire={area_str}",
                            f"circ={contour_metrics.get('circularity', 0):.3f}",
                            f"forme={width_profile.get('shape_class', '?')}",
                        ]
                        if skel_metrics.get('seg_length_mm'):
                            log_parts.append(f"L_seg={skel_metrics['seg_length_mm']:.1f}mm")
                        elif skel_metrics.get('seg_length_px'):
                            log_parts.append(f"L_seg={skel_metrics['seg_length_px']:.0f}px")
                        if skel_metrics.get('seg_width_mm'):
                            log_parts.append(f"W_seg={skel_metrics['seg_width_mm']:.1f}mm")
                        elif skel_metrics.get('seg_width_px'):
                            log_parts.append(f"W_seg={skel_metrics['seg_width_px']:.0f}px")
                        logger.debug(", ".join(log_parts))
                    else:
                        logger.debug(f"  Épi #{result['id']}: segmentation échouée")
                
                logger.info(f"  {n_segmented}/{len(spike_results)} épis segmentés")
                
                if self.debug_level >= 2:
                    self.save_debug_05b_segmentation(image, spike_results, session_dir)
            else:
                logger.info("[5/10] Segmentation des épis: désactivée")
            
            # =====================================================================
            # ÉTAPE 6: COMPTAGE DES ÉPILLETS (avec TTA optionnel)
            # =====================================================================
            tta_label = " + TTA" if self.use_tta else ""
            logger.info(f"[6/10] Comptage des épillets{tta_label}...")
            
            for result in spike_results:
                det = result['detection']
                if det is None:
                    result['spikelets'] = {'count': None, 'method': 'unavailable', 'positions': []}
                    continue
                
                spikelets = self.count_spikelets(
                    image, det,
                    use_tta=self.use_tta,
                    tta_augmentations=self.tta_augmentations if self.use_tta else None,
                    mask=result.get('segmentation', {}).get('mask') if result.get('segmentation') else None,
                )
                result['spikelets'] = spikelets
                
                if spikelets['count']:
                    # Calculer la densité d'épillets si calibré
                    spike_length_mm = result['measurements'].get('spike_length_mm') or result['measurements'].get('length_mm')
                    if spike_length_mm and spike_length_mm > 0:
                        density = spikelets['count'] / (spike_length_mm / 10)  # épillets/cm
                        result['measurements']['spikelet_density_per_cm'] = density
                    
                    logger.info(f"  Épi #{result['id']}: {spikelets['count']} épillets "
                               f"({spikelets['confidence']})"
                               + (f", densité={result['measurements'].get('spikelet_density_per_cm', 0):.1f}/cm" 
                                  if result['measurements'].get('spikelet_density_per_cm') else "")
                    )
            
            if self.debug_level >= 2:
                self.save_debug_03_spikes(image, spike_results, session_dir)
            
            # =====================================================================
            # ÉTAPE 7: SEGMENTATION DES ÉPILLETS (YOLO-Seg ou SAM2)
            # =====================================================================
            seg_enabled = self.config.get('spikelet_counting', {}).get('segmentation', {}).get('enabled', True)
            
            if seg_enabled:
                # Vérifier si les masques YOLO-Seg sont disponibles (du comptage étape 6)
                has_yolo_masks = any(
                    r.get('spikelets', {}).get('masks_roi') is not None
                    for r in spike_results
                )
                
                if has_yolo_masks:
                    logger.info("[7/10] Segmentation des épillets (YOLO-Seg)...")
                    for result in spike_results:
                        spikelets = result.get('spikelets', {})
                        masks_roi = spikelets.get('masks_roi')
                        bboxes = spikelets.get('bboxes', [])
                        roi_offset = spikelets.get('roi_offset', (0, 0))
                        spike_mask = result.get('segmentation', {}).get('mask') if result.get('segmentation') else None
                        
                        if masks_roi is None or not bboxes:
                            result['spikelet_details'] = []
                            continue
                        
                        spikelet_details = self._compute_spikelet_details_from_yolo_masks(
                            masks_roi=masks_roi,
                            bboxes_global=bboxes,
                            roi_offset=roi_offset,
                            spike_mask=spike_mask,
                            pixel_per_mm=self.pixel_per_mm,
                        )
                        result['spikelet_details'] = spikelet_details
                        
                        # Statistiques agrégées
                        spikelet_stats = SpikeSegmenter.compute_spikelet_stats(
                            spikelet_details, self.pixel_per_mm
                        )
                        result['measurements']['spikelet_stats'] = spikelet_stats
                        
                        n_seg = spikelet_stats.get('n_segmented', 0)
                        n_tot = spikelet_stats.get('n_total', 0)
                        if n_seg > 0:
                            mean_l = spikelet_stats.get('spikelet_length_mm_mean', spikelet_stats.get('spikelet_length_px_mean', 0))
                            mean_w = spikelet_stats.get('spikelet_width_mm_mean', spikelet_stats.get('spikelet_width_px_mean', 0))
                            unit = 'mm' if 'spikelet_length_mm_mean' in spikelet_stats else 'px'
                            logger.debug(
                                f"  Épi #{result['id']}: {n_seg}/{n_tot} épillets segmentés, "
                                f"L_moy={mean_l:.1f}{unit}, W_moy={mean_w:.1f}{unit}"
                            )
                        else:
                            logger.debug(f"  Épi #{result['id']}: aucun épillet segmenté")
                    
                    if self.debug_level >= 2:
                        self.save_debug_06_spikelets(image, spike_results, session_dir)
                
                else:
                    logger.info("[7/10] Segmentation épillets: pas de masques YOLO-Seg disponibles")
                    for result in spike_results:
                            result['spikelet_details'] = []
            else:
                logger.info("[7/10] Segmentation épillets: désactivée")
                for result in spike_results:
                    result['spikelet_details'] = []
            
            # --- Libérer les masques épillets/segmentation devenus inutiles ---
            for result in spike_results:
                spikelets = result.get('spikelets', {})
                if isinstance(spikelets, dict):
                    spikelets.pop('masks_roi', None)  # gros tableaux numpy
            gc.collect()
            
            # =====================================================================
            # ÉTAPE 8: DÉTECTION DU RACHIS (YOLO-Seg)
            # =====================================================================
            if self.rachis_detector is not None:
                rachis_config = self.config.get('rachis_detection', {})
                rachis_conf = rachis_config.get('confidence_threshold', 0.3)
                rachis_margin = rachis_config.get('crop_margin', 30)
                tta_rachis_label = " + TTA" if self.use_tta_rachis else ""
                logger.info(f"[8/10] Détection du rachis + angles d'insertion{tta_rachis_label}...")
                
                rachis_detected = 0
                for result in spike_results:
                    det = result.get('detection')
                    if det is None:
                        result['rachis'] = None
                        continue
                    
                    # --- Crop OBB redressé (même procédure que l'entraînement) ---
                    cx, cy = det.center
                    rw, rh = det.width, det.height  # width < height garanti
                    angle = det.angle  # déjà ajusté pour le swap w/h
                    
                    out_w = int(rw + 2 * rachis_margin)
                    out_h = int(rh + 2 * rachis_margin)
                    
                    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                    M[0, 2] += out_w / 2 - cx
                    M[1, 2] += out_h / 2 - cy
                    
                    crop = cv2.warpAffine(
                        image, M, (out_w, out_h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )
                    
                    # --- Inférence YOLO-Seg sur le crop (avec ou sans TTA) ---
                    if self.use_tta_rachis:
                        try:
                            from .tta import tta_rachis_segment
                        except ImportError:
                            from tta import tta_rachis_segment
                        tta_result = tta_rachis_segment(
                            model=self.rachis_detector,
                            crop=crop,
                            conf=rachis_conf,
                            augmentations=self.tta_rachis_augs,
                            consensus_threshold=self.tta_rachis_consensus,
                        )
                        if tta_result is not None:
                            consensus_mask, mean_conf = tta_result
                            rachis_info = self._extract_rachis_from_yolo(
                                None, crop, M, image.shape, self.pixel_per_mm,
                                precomputed_mask=consensus_mask,
                                precomputed_conf=mean_conf,
                            )
                        else:
                            rachis_info = None
                    else:
                        rachis_results = self.rachis_detector(
                            crop, conf=rachis_conf, verbose=False
                        )
                        rachis_info = self._extract_rachis_from_yolo(
                            rachis_results, crop, M, image.shape, self.pixel_per_mm
                        )
                    
                    result['rachis'] = rachis_info
                    
                    if rachis_info is not None:
                        rachis_detected += 1
                        # Stocker les mesures dans measurements
                        result['measurements']['rachis_length_px'] = rachis_info['length_px']
                        if rachis_info.get('length_mm'):
                            result['measurements']['rachis_length_mm'] = rachis_info['length_mm']
                        rachis_len = rachis_info.get('length_mm') or rachis_info['length_px']
                        rachis_unit = 'mm' if rachis_info.get('length_mm') else 'px'
                        logger.debug(
                            f"  Épi #{result['id']}: rachis détecté, "
                            f"L={rachis_len:.1f}{rachis_unit}"
                        )
                    else:
                        logger.debug(f"  Épi #{result['id']}: rachis non détecté")
                
                logger.info(f"  Rachis détecté sur {rachis_detected}/{len(spike_results)} épis")
                
                if self.debug_level >= 2:
                    self.save_debug_07_rachis(image, spike_results, session_dir)
                
                # --- Calcul des angles d'insertion des épillets ---
                angles_computed = 0
                for result in spike_results:
                    rachis = result.get('rachis')
                    details = result.get('spikelet_details', [])
                    
                    if rachis is None or not details:
                        result['insertion_angles'] = []
                        continue
                    
                    angle_data = self._compute_spikelet_insertion_angles(
                        image_shape=image.shape,
                        rachis_info=rachis,
                        spikelet_details=details,
                        pixel_per_mm=self.pixel_per_mm,
                    )
                    result['insertion_angles'] = angle_data
                    
                    # Stocker dans les mesures agrégées
                    valid_angles = [a['insertion_angle_deg'] for a in angle_data
                                    if a.get('insertion_angle_deg') is not None]
                    if valid_angles:
                        angles_computed += 1
                        result['measurements']['insertion_angle_mean'] = float(np.mean(valid_angles))
                        result['measurements']['insertion_angle_std'] = float(np.std(valid_angles))
                        result['measurements']['insertion_angle_min'] = float(min(valid_angles))
                        result['measurements']['insertion_angle_max'] = float(max(valid_angles))
                        n_left = sum(1 for a in angle_data if a.get('side') == 'left')
                        n_right = sum(1 for a in angle_data if a.get('side') == 'right')
                        result['measurements']['spikelets_left'] = n_left
                        result['measurements']['spikelets_right'] = n_right
                        logger.debug(
                            f"  Épi #{result['id']}: {len(valid_angles)} angles calculés, "
                            f"moy={np.mean(valid_angles):.1f}° ± {np.std(valid_angles):.1f}°, "
                            f"L={n_left}/R={n_right}"
                        )
                
                logger.info(f"  Angles d'insertion calculés sur {angles_computed}/{len(spike_results)} épis")
                
                if self.debug_level >= 2:
                    self.save_debug_08_insertion_angles(image, spike_results, session_dir)
            else:
                logger.info("[8/10] Détection du rachis: désactivée")
                for result in spike_results:
                    result['rachis'] = None
                    result['insertion_angles'] = []
            
            # --- Libérer les masques rachis / contours épillets lourds ---
            for result in spike_results:
                rachis = result.get('rachis')
                if rachis and isinstance(rachis, dict):
                    rachis.pop('mask_crop', None)
                    rachis.pop('mask_contour_global', None)
                for sp in result.get('spikelet_details', []):
                    if isinstance(sp, dict):
                        sp.pop('contour', None)
                        sp.pop('mask_roi', None)
            gc.collect()
            
            # =====================================================================
            # ÉTAPE 9: IDENTIFICATION DU SACHET
            # =====================================================================
            tta_bag_label = " + TTA" if self.use_tta_bag else ""
            logger.info(f"[9/10] Identification du sachet{tta_bag_label}...")
            
            bag_det = None
            bag_info = {'detected': False}
            
            if detections['bags']:
                bag_det = max(detections['bags'], key=lambda d: d.confidence)
                bag_info = self.identify_bag(
                    image, bag_det,
                    use_tta=self.use_tta_bag,
                    tta_augmentations=self.tta_bag_augs if self.use_tta_bag else None,
                )
                
                if bag_info.get('sample_id'):
                    logger.info(f"  Échantillon identifié: {bag_info['sample_id']}")
                else:
                    logger.info("  Chiffres non détectés sur le sachet")
            else:
                logger.info("  Aucun sachet détecté")
            
            if self.debug_level >= 2:
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
                    'calibration_method': self.calibration_method,
                    'ruler_length_px': ruler_det.height if ruler_det else None,
                    'graduations_used': {
                        'grad_1': self.calibration_details.get('grad_1'),
                        'grad_2': self.calibration_details.get('grad_2'),
                        'distance_px': self.calibration_details.get('distance_px'),
                        'distance_mm': self.calibration_details.get('distance_mm'),
                    } if self.calibration_method == 'graduations' else None,
                },
                'bag': bag_info,
                'spike_count': len(spike_results),
                'spikes': [
                    {
                        'id': r['id'],
                        'measurements': {
                            k: v for k, v in r['measurements'].items()
                            if k not in ('width_profile', 'spikelet_stats') or not isinstance(v, dict) or 'widths_px' not in v
                        },
                        # Flatten key segmentation metrics into measurements for JSON
                        'segmentation_metrics': {
                            k: r['measurements'][k]
                            for k in [
                                'real_area_px', 'real_area_mm2',
                                'real_perimeter_px', 'real_perimeter_mm',
                                'circularity', 'solidity',
                                'ellipse_eccentricity',
                                'seg_length_px', 'seg_length_mm',
                                'seg_width_px', 'seg_width_mm',
                                'seg_aspect_ratio',
                            ]
                            if k in r['measurements']
                        } if r.get('segmentation') else None,
                        'width_profile': {
                            k: v for k, v in r['measurements'].get('width_profile', {}).items()
                            if k != 'widths_px' and k != 'widths_mm'  # exclude raw arrays for brevity
                        } if r['measurements'].get('width_profile') else None,
                        'color': r['measurements'].get('color'),
                        'spikelet_count': r['spikelets'].get('count'),
                        'spikelet_method': r['spikelets'].get('method'),
                        'spikelet_confidence': r['spikelets'].get('confidence'),
                        'spikelet_density_per_cm': r['measurements'].get('spikelet_density_per_cm'),
                        'spikelets': r.get('spikelets', {}),
                        'has_segmentation': r.get('segmentation') is not None,
                        # Spikelet stats
                        'spikelet_stats': {
                            k: v for k, v in r['measurements'].get('spikelet_stats', {}).items()
                        } if r['measurements'].get('spikelet_stats') else None,
                        # Spikelet details (sans contours/masques pour la sérialisation)
                        'spikelet_details': [
                            {k: v for k, v in sp.items() if k not in ('contour', 'mask_roi')}
                            for sp in r.get('spikelet_details', [])
                        ] if r.get('spikelet_details') else None,
                        # Rachis
                        'rachis': {
                            'detected': r.get('rachis') is not None,
                            'confidence': r['rachis']['confidence'] if r.get('rachis') else None,
                            'length_px': r['rachis']['length_px'] if r.get('rachis') else None,
                            'length_mm': r['rachis'].get('length_mm') if r.get('rachis') else None,
                        },
                        # Angles d'insertion
                        'insertion_angles': [
                            {k: v for k, v in a.items()}
                            for a in r.get('insertion_angles', [])
                        ] if r.get('insertion_angles') else None,
                        'insertion_angle_stats': {
                            'mean': r['measurements'].get('insertion_angle_mean'),
                            'std': r['measurements'].get('insertion_angle_std'),
                            'min': r['measurements'].get('insertion_angle_min'),
                            'max': r['measurements'].get('insertion_angle_max'),
                            'spikelets_left': r['measurements'].get('spikelets_left'),
                            'spikelets_right': r['measurements'].get('spikelets_right'),
                        } if r['measurements'].get('insertion_angle_mean') is not None else None,
                    }
                    for r in spike_results
                ],
            }
            
            # Sauvegarder JSON
            with open(session_dir / 'results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Image finale (générée en mode low-debug ou full-debug)
            if self.debug_level >= 1:
                self.save_debug_05_final(image, results, spike_results, detections, session_dir)
            
            logger.info(f"✓ Résultats sauvegardés: {session_dir}")
            
            # --- Nettoyage mémoire fin d'image ---
            # Supprimer les gros objets intermédiaires avant de retourner
            # (l'image, les masques numpy, les crops, etc.)
            for sr in spike_results:
                sr.pop('segmentation', None)
                sr.pop('detection', None)
                sr.pop('whole_spike_det', None)
                sr.pop('spike_det', None)
                sr.pop('rachis', None)
                sr.pop('spikelet_details', None)
                spikelets = sr.get('spikelets', {})
                if isinstance(spikelets, dict):
                    spikelets.pop('masks_roi', None)
            del image, detections, spike_results, refinement_seg_results
            self._clear_detection_caches()
            self.clear_memory(log=False)
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur analyse: {e}")
            logger.exception("Détails:")
            # Nettoyage même en cas d'erreur
            self._clear_detection_caches()
            self.clear_memory(log=False)
            return None
    
    def analyze_batch(self, image_paths: List[str], include_existing: bool = True) -> List[Dict]:
        """
        Analyse un lot d'images et génère un CSV récapitulatif
        
        Args:
            image_paths: Liste des chemins d'images à traiter
            include_existing: Si True, inclure les résultats existants dans le CSV final
        """
        results = []
        
        # Intervalle de nettoyage mémoire (toutes les N images)
        memory_cleanup_interval = self.config.get('batch', {}).get(
            'memory_cleanup_interval', 5
        )
        
        for i, path in enumerate(image_paths):
            logger.info(f"\n[{i+1}/{len(image_paths)}] {Path(path).name}")
            try:
                result = self.analyze_image(path)
            except RuntimeError as e:
                # Attraper les erreurs GPU fatales (HIP/CUDA kernel failure)
                # pour ne pas stopper tout le batch
                logger.error(f"Erreur GPU fatale sur {Path(path).name}: {e}")
                logger.warning("Tentative de récupération du GPU...")
                self._recover_gpu()
                result = None
            if result:
                # Résultats déjà sauvegardés en JSON → on ne garde qu'un
                # dict léger (sans masques numpy) pour le CSV final.
                results.append(self._strip_heavy_arrays(result))
            
            # Nettoyage périodique RAM + VRAM GPU
            if (i + 1) % memory_cleanup_interval == 0:
                logger.info(f"🧹 Nettoyage mémoire (après {i+1} images)...")
                self.clear_memory()
        
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
                        'spikelet_density_per_cm': spike.get('spikelet_density_per_cm', ''),
                        # Segmentation (métriques réelles)
                        'has_segmentation': spike.get('has_segmentation', False),
                    })
                    
                    # Ajouter les métriques de segmentation si disponibles
                    seg_metrics = spike.get('segmentation_metrics') or {}
                    row.update({
                        'real_area_px': seg_metrics.get('real_area_px', ''),
                        'real_area_mm2': seg_metrics.get('real_area_mm2', ''),
                        'real_perimeter_px': seg_metrics.get('real_perimeter_px', ''),
                        'real_perimeter_mm': seg_metrics.get('real_perimeter_mm', ''),
                        'circularity': seg_metrics.get('circularity', ''),
                        'solidity': seg_metrics.get('solidity', ''),
                        'ellipse_eccentricity': seg_metrics.get('ellipse_eccentricity', ''),
                        # Longueur/largeur par segmentation (squelette)
                        'seg_length_px': seg_metrics.get('seg_length_px', ''),
                        'seg_length_mm': seg_metrics.get('seg_length_mm', ''),
                        'seg_width_px': seg_metrics.get('seg_width_px', ''),
                        'seg_width_mm': seg_metrics.get('seg_width_mm', ''),
                        'seg_aspect_ratio': seg_metrics.get('seg_aspect_ratio', ''),
                    })
                    
                    # Profil de largeur
                    wp = spike.get('width_profile') or {}
                    row.update({
                        'shape_class': wp.get('shape_class', ''),
                        'apical_width_mm': wp.get('apical_width_mm', ''),
                        'medial_width_mm': wp.get('medial_width_mm', ''),
                        'basal_width_mm': wp.get('basal_width_mm', ''),
                        'max_width_mm': wp.get('max_width_mm', ''),
                        'max_width_position': wp.get('max_width_position', ''),
                    })
                    
                    # Couleur
                    col = spike.get('color') or {}
                    row.update({
                        'hue_mean': col.get('hue_mean', ''),
                        'saturation_mean': col.get('saturation_mean', ''),
                        'value_mean': col.get('value_mean', ''),
                        'greenness_index': col.get('greenness_index', ''),
                        'yellowing_index': col.get('yellowing_index', ''),
                    })
                    
                    # Coordonnées
                    row.update({
                        'center_x': m.get('center_x', ''),
                        'center_y': m.get('center_y', ''),
                        'confidence': m.get('confidence', ''),
                    })
                    
                    # Statistiques épillets segmentés
                    sp_stats = spike.get('spikelet_stats') or {}
                    row.update({
                        'spikelet_seg_count': sp_stats.get('n_segmented', ''),
                        'spikelet_length_mm_mean': sp_stats.get('spikelet_length_mm_mean', ''),
                        'spikelet_length_mm_std': sp_stats.get('spikelet_length_mm_std', ''),
                        'spikelet_width_mm_mean': sp_stats.get('spikelet_width_mm_mean', ''),
                        'spikelet_width_mm_std': sp_stats.get('spikelet_width_mm_std', ''),
                        'spikelet_area_mm2_mean': sp_stats.get('spikelet_area_mm2_mean', ''),
                        'spikelet_area_mm2_std': sp_stats.get('spikelet_area_mm2_std', ''),
                        'spikelet_aspect_ratio_mean': sp_stats.get('spikelet_aspect_ratio_mean', ''),
                        'spikelet_circularity_mean': sp_stats.get('spikelet_circularity_mean', ''),
                        'spikelet_length_cv': sp_stats.get('spikelet_length_cv', ''),
                        'spikelet_area_cv': sp_stats.get('spikelet_area_cv', ''),
                    })
                    
                    # Rachis
                    rachis_data = spike.get('rachis') or {}
                    row.update({
                        'rachis_detected': rachis_data.get('detected', False),
                        'rachis_confidence': rachis_data.get('confidence', ''),
                        'rachis_length_px': rachis_data.get('length_px', ''),
                        'rachis_length_mm': rachis_data.get('length_mm', ''),
                    })
                    
                    # Angles d'insertion
                    angle_stats = spike.get('insertion_angle_stats') or {}
                    row.update({
                        'insertion_angle_mean': angle_stats.get('mean', ''),
                        'insertion_angle_std': angle_stats.get('std', ''),
                        'insertion_angle_min': angle_stats.get('min', ''),
                        'insertion_angle_max': angle_stats.get('max', ''),
                        'spikelets_left': angle_stats.get('spikelets_left', ''),
                        'spikelets_right': angle_stats.get('spikelets_right', ''),
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
            'spikelet_count', 'spikelet_method', 'spikelet_confidence', 'spikelet_density_per_cm',
            # Segmentation metrics
            'has_segmentation', 'real_area_px', 'real_area_mm2', 'real_perimeter_px', 'real_perimeter_mm',
            'circularity', 'solidity', 'ellipse_eccentricity',
            # Segmentation-based length/width (skeleton)
            'seg_length_px', 'seg_length_mm', 'seg_width_px', 'seg_width_mm', 'seg_aspect_ratio',
            # Width profile
            'shape_class', 'apical_width_mm', 'medial_width_mm', 'basal_width_mm',
            'max_width_mm', 'max_width_position',
            # Color
            'hue_mean', 'saturation_mean', 'value_mean', 'greenness_index', 'yellowing_index',
            # Position
            'center_x', 'center_y', 'confidence',
            # Spikelet segmentation stats
            'spikelet_seg_count', 'spikelet_length_mm_mean', 'spikelet_length_mm_std',
            'spikelet_width_mm_mean', 'spikelet_width_mm_std',
            'spikelet_area_mm2_mean', 'spikelet_area_mm2_std',
            'spikelet_aspect_ratio_mean', 'spikelet_circularity_mean',
            'spikelet_length_cv', 'spikelet_area_cv',
            # Rachis
            'rachis_detected', 'rachis_confidence', 'rachis_length_px', 'rachis_length_mm',
            # Angles d'insertion
            'insertion_angle_mean', 'insertion_angle_std', 'insertion_angle_min', 'insertion_angle_max',
            'spikelets_left', 'spikelets_right',
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
