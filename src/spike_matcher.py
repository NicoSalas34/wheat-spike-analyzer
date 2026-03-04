#!/usr/bin/env python3
"""
Module d'appariement robuste spike ↔ whole_spike.

Utilise le calcul d'IoU sur des rectangles orientés (OBB) et l'algorithme
hongrois (Hungarian algorithm) pour un appariement optimal 1-à-1.

Remplace la méthode naïve « le centre de spike est-il dans la bbox de
whole_spike ? » par une assignation globalement optimale basée sur l'IoU.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compute_obb_iou(det1, det2) -> float:
    """
    Calcule l'Intersection over Union (IoU) entre deux détections OBB.
    
    Utilise cv2.rotatedRectangleIntersection pour calculer l'aire
    d'intersection de deux rectangles orientés.
    
    Args:
        det1: OBBDetection (doit avoir .center, .width, .height, .angle)
        det2: OBBDetection
        
    Returns:
        IoU entre 0.0 et 1.0
    """
    # Construire les RotatedRect OpenCV: ((cx, cy), (w, h), angle)
    rect1 = (
        (float(det1.center[0]), float(det1.center[1])),
        (float(det1.width), float(det1.height)),
        float(det1.angle),
    )
    rect2 = (
        (float(det2.center[0]), float(det2.center[1])),
        (float(det2.width), float(det2.height)),
        float(det2.angle),
    )
    
    # Calculer l'intersection
    ret, intersection_pts = cv2.rotatedRectangleIntersection(rect1, rect2)
    
    if ret == cv2.INTERSECT_NONE or intersection_pts is None:
        return 0.0
    
    # Aire d'intersection (polygone convexe ordonné)
    intersection_pts = intersection_pts.reshape(-1, 2)
    
    if len(intersection_pts) < 3:
        return 0.0
    
    # Ordonner les points et calculer l'aire du polygone
    hull = cv2.convexHull(intersection_pts.astype(np.float32))
    intersection_area = cv2.contourArea(hull)
    
    # Aires des deux rectangles
    area1 = det1.width * det1.height
    area2 = det2.width * det2.height
    
    # IoU
    union_area = area1 + area2 - intersection_area
    
    if union_area <= 0:
        return 0.0
    
    return float(intersection_area / union_area)


def compute_containment_score(inner_det, outer_det) -> float:
    """
    Calcule à quel point inner_det est contenu dans outer_det.
    
    Score = intersection_area / area(inner_det)
    
    Utile car un spike doit être *contenu* dans un whole_spike,
    pas juste avoir du recouvrement.
    
    Args:
        inner_det: Détection supposée contenue (spike)
        outer_det: Détection englobante (whole_spike)
        
    Returns:
        Score de containment entre 0.0 et 1.0
    """
    rect1 = (
        (float(inner_det.center[0]), float(inner_det.center[1])),
        (float(inner_det.width), float(inner_det.height)),
        float(inner_det.angle),
    )
    rect2 = (
        (float(outer_det.center[0]), float(outer_det.center[1])),
        (float(outer_det.width), float(outer_det.height)),
        float(outer_det.angle),
    )
    
    ret, intersection_pts = cv2.rotatedRectangleIntersection(rect1, rect2)
    
    if ret == cv2.INTERSECT_NONE or intersection_pts is None:
        return 0.0
    
    intersection_pts = intersection_pts.reshape(-1, 2)
    
    if len(intersection_pts) < 3:
        return 0.0
    
    hull = cv2.convexHull(intersection_pts.astype(np.float32))
    intersection_area = cv2.contourArea(hull)
    
    area_inner = inner_det.width * inner_det.height
    
    if area_inner <= 0:
        return 0.0
    
    return float(min(intersection_area / area_inner, 1.0))


def match_spikes_hungarian(
    spikes: List,
    whole_spikes: List,
    min_iou: float = 0.15,
    min_containment: float = 0.5,
) -> List[Tuple[Optional[int], Optional[int]]]:
    """
    Appariement optimal 1-à-1 entre spikes et whole_spikes.
    
    Utilise l'algorithme hongrois pour trouver l'assignation qui
    maximise la somme des IoU tout en respectant les contraintes.
    
    Args:
        spikes: Liste de OBBDetection pour les spikes
        whole_spikes: Liste de OBBDetection pour les whole_spikes
        min_iou: IoU minimum pour considérer un appariement valide
        min_containment: Score de containment minimum (spike dans whole_spike)
        
    Returns:
        Liste de tuples (whole_spike_idx, spike_idx).
        - (ws_idx, sp_idx): appariement trouvé
        - (ws_idx, None): whole_spike sans spike associé
        - (None, sp_idx): spike sans whole_spike associé
    """
    n_spikes = len(spikes)
    n_whole = len(whole_spikes)
    
    if n_spikes == 0 and n_whole == 0:
        return []
    
    if n_spikes == 0:
        # Tous les whole_spikes sont seuls
        return [(i, None) for i in range(n_whole)]
    
    if n_whole == 0:
        # Tous les spikes sont seuls
        return [(None, i) for i in range(n_spikes)]
    
    # Construire la matrice de coût (on veut maximiser l'IoU → minimiser -IoU)
    # Utiliser un score combiné: IoU pondéré par containment
    cost_matrix = np.zeros((n_whole, n_spikes), dtype=np.float64)
    iou_matrix = np.zeros((n_whole, n_spikes), dtype=np.float64)
    containment_matrix = np.zeros((n_whole, n_spikes), dtype=np.float64)
    
    for i, ws in enumerate(whole_spikes):
        for j, sp in enumerate(spikes):
            iou = compute_obb_iou(ws, sp)
            containment = compute_containment_score(sp, ws)
            
            iou_matrix[i, j] = iou
            containment_matrix[i, j] = containment
            
            # Score combiné : IoU * containment
            # On prend le négatif car linear_sum_assignment minimise
            cost_matrix[i, j] = -(iou * containment)
    
    logger.debug(f"Matrice IoU ({n_whole} whole × {n_spikes} spikes):")
    logger.debug(f"  IoU max par whole_spike: {iou_matrix.max(axis=1).tolist()}")
    logger.debug(f"  IoU max par spike: {iou_matrix.max(axis=0).tolist()}")
    
    # Algorithme hongrois
    try:
        from scipy.optimize import linear_sum_assignment
        ws_indices, sp_indices = linear_sum_assignment(cost_matrix)
    except ImportError:
        logger.warning("scipy non disponible, fallback vers appariement glouton")
        return _greedy_matching(
            spikes, whole_spikes, iou_matrix, containment_matrix,
            min_iou, min_containment,
        )
    
    # Construire les paires valides
    matched_ws = set()
    matched_sp = set()
    pairs = []
    
    for ws_idx, sp_idx in zip(ws_indices, sp_indices):
        iou = iou_matrix[ws_idx, sp_idx]
        containment = containment_matrix[ws_idx, sp_idx]
        
        if iou >= min_iou and containment >= min_containment:
            pairs.append((ws_idx, sp_idx))
            matched_ws.add(ws_idx)
            matched_sp.add(sp_idx)
            logger.debug(
                f"  Appariement: whole_spike[{ws_idx}] ↔ spike[{sp_idx}] "
                f"(IoU={iou:.3f}, containment={containment:.3f})"
            )
        else:
            logger.debug(
                f"  Rejeté: whole_spike[{ws_idx}] ↔ spike[{sp_idx}] "
                f"(IoU={iou:.3f}, containment={containment:.3f})"
            )
    
    # Ajouter les non-appariés
    for i in range(n_whole):
        if i not in matched_ws:
            pairs.append((i, None))
    
    for j in range(n_spikes):
        if j not in matched_sp:
            pairs.append((None, j))
    
    logger.info(
        f"Appariement: {len(matched_ws)} paires, "
        f"{n_whole - len(matched_ws)} whole_spikes seuls, "
        f"{n_spikes - len(matched_sp)} spikes seuls"
    )
    
    return pairs


def _greedy_matching(
    spikes: List,
    whole_spikes: List,
    iou_matrix: np.ndarray,
    containment_matrix: np.ndarray,
    min_iou: float,
    min_containment: float,
) -> List[Tuple[Optional[int], Optional[int]]]:
    """
    Fallback glouton si scipy n'est pas disponible.
    
    Trie les paires par IoU*containment décroissant et assigne en priorité.
    """
    n_whole, n_spikes = iou_matrix.shape
    
    # Construire la liste de toutes les paires triées par score
    candidates = []
    for i in range(n_whole):
        for j in range(n_spikes):
            iou = iou_matrix[i, j]
            cont = containment_matrix[i, j]
            if iou >= min_iou and cont >= min_containment:
                candidates.append((iou * cont, i, j))
    
    candidates.sort(reverse=True)
    
    matched_ws = set()
    matched_sp = set()
    pairs = []
    
    for score, ws_idx, sp_idx in candidates:
        if ws_idx not in matched_ws and sp_idx not in matched_sp:
            pairs.append((ws_idx, sp_idx))
            matched_ws.add(ws_idx)
            matched_sp.add(sp_idx)
    
    # Non-appariés
    for i in range(n_whole):
        if i not in matched_ws:
            pairs.append((i, None))
    
    for j in range(n_spikes):
        if j not in matched_sp:
            pairs.append((None, j))
    
    return pairs
