#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) utilities for robust YOLO inference.

Provides geometric augmentation/inverse transforms for:
- Masks (pixel-level consensus via majority voting)
- Bounding boxes (NMS-based aggregation)
- Digit detections (majority voting on values)

Used across all pipeline steps for more robust predictions.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Callable

logger = logging.getLogger(__name__)

# =============================================================================
# AUGMENTATION PRIMITIVES
# =============================================================================

# Geometric augmentations that preserve image dimensions
GEOMETRIC_AUGMENTATIONS = ['original', 'flip_h', 'flip_v', 'rot180']

# Photometric augmentations (no geometry change)
PHOTOMETRIC_AUGMENTATIONS = [
    'brightness_up',
    'brightness_down',
    'contrast_up',
    'contrast_down',
    'hue_shift',
    'saturation_up',
    'saturation_down',
]

# All augmentations suitable for mask-based TTA
MASK_TTA_AUGMENTATIONS = GEOMETRIC_AUGMENTATIONS + PHOTOMETRIC_AUGMENTATIONS

# All augmentations suitable for detection-based TTA
DETECT_TTA_AUGMENTATIONS = GEOMETRIC_AUGMENTATIONS


def apply_augmentation(image: np.ndarray, aug_name: str):
    """
    Apply an augmentation to an image.
    
    Returns:
        Tuple of (augmented_image, inverse_mask_fn, inverse_bbox_fn)
        - inverse_mask_fn(mask) → mask in original space
        - inverse_bbox_fn(x1,y1,x2,y2) → (x1,y1,x2,y2) in original space
        Returns (None, None, None) if augmentation name is unknown.
    """
    h, w = image.shape[:2]
    
    if aug_name == 'original':
        return (
            image,
            lambda m: m,
            lambda x1, y1, x2, y2: (x1, y1, x2, y2),
        )
    
    elif aug_name == 'flip_h':
        aug = cv2.flip(image, 1)
        return (
            aug,
            lambda m: cv2.flip(m, 1),
            lambda x1, y1, x2, y2: (w - 1 - x2, y1, w - 1 - x1, y2),
        )
    
    elif aug_name == 'flip_v':
        aug = cv2.flip(image, 0)
        return (
            aug,
            lambda m: cv2.flip(m, 0),
            lambda x1, y1, x2, y2: (x1, h - 1 - y2, x2, h - 1 - y1),
        )
    
    elif aug_name == 'rot180':
        aug = cv2.flip(image, -1)
        return (
            aug,
            lambda m: cv2.flip(m, -1),
            lambda x1, y1, x2, y2: (w - 1 - x2, h - 1 - y2, w - 1 - x1, h - 1 - y1),
        )
    
    elif aug_name == 'brightness_up':
        aug = cv2.convertScaleAbs(image, alpha=1.0, beta=30)
        return (
            aug,
            lambda m: m,  # No geometric change
            lambda x1, y1, x2, y2: (x1, y1, x2, y2),
        )
    
    elif aug_name == 'brightness_down':
        aug = cv2.convertScaleAbs(image, alpha=1.0, beta=-30)
        return (
            aug,
            lambda m: m,
            lambda x1, y1, x2, y2: (x1, y1, x2, y2),
        )
    
    elif aug_name == 'contrast_up':
        aug = cv2.convertScaleAbs(image, alpha=1.3, beta=0)
        return (
            aug,
            lambda m: m,
            lambda x1, y1, x2, y2: (x1, y1, x2, y2),
        )

    elif aug_name == 'contrast_down':
        aug = cv2.convertScaleAbs(image, alpha=0.75, beta=0)
        return (
            aug,
            lambda m: m,
            lambda x1, y1, x2, y2: (x1, y1, x2, y2),
        )

    elif aug_name == 'hue_shift':
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].astype(np.int16)
        h = (h + 10) % 180
        hsv[:, :, 0] = h.astype(np.uint8)
        aug = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return (
            aug,
            lambda m: m,
            lambda x1, y1, x2, y2: (x1, y1, x2, y2),
        )

    elif aug_name == 'saturation_up':
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1].astype(np.float32)
        s = np.clip(s * 1.25, 0, 255)
        hsv[:, :, 1] = s.astype(np.uint8)
        aug = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return (
            aug,
            lambda m: m,
            lambda x1, y1, x2, y2: (x1, y1, x2, y2),
        )

    elif aug_name == 'saturation_down':
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1].astype(np.float32)
        s = np.clip(s * 0.75, 0, 255)
        hsv[:, :, 1] = s.astype(np.uint8)
        aug = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return (
            aug,
            lambda m: m,
            lambda x1, y1, x2, y2: (x1, y1, x2, y2),
        )
    
    else:
        logger.warning(f"Augmentation TTA inconnue: {aug_name}")
        return None, None, None


# =============================================================================
# MASK-LEVEL TTA (for segmentation steps)
# =============================================================================

def merge_masks_consensus(
    masks: List[np.ndarray],
    threshold: float = 0.5,
) -> Optional[np.ndarray]:
    """
    Merge multiple binary masks by pixel-level majority voting.
    
    Args:
        masks: List of binary masks (uint8, 0/255), all same shape
        threshold: Fraction of masks that must agree (default 0.5 = majority)
    
    Returns:
        Consensus mask (uint8, 0/255) or None if no masks
    """
    if not masks:
        return None
    
    if len(masks) == 1:
        return masks[0]
    
    # Stack and vote
    stack = np.stack([(m > 127).astype(np.float32) for m in masks], axis=0)
    vote = stack.mean(axis=0)
    consensus = (vote >= threshold).astype(np.uint8) * 255
    
    return consensus


def tta_yolo_segment(
    model,
    crop: np.ndarray,
    conf: float,
    augmentations: List[str] = None,
    min_mask_area: int = 0,
    consensus_threshold: float = 0.5,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Run YOLO-Seg with TTA and return consensus mask.
    
    For each augmentation:
    1. Transform the crop
    2. Run model inference
    3. Get best mask
    4. Inverse-transform mask back to original space
    5. Merge all masks by pixel voting
    
    Args:
        model: YOLO-Seg model
        crop: Input crop (BGR)
        conf: Confidence threshold
        augmentations: List of augmentation names
        min_mask_area: Minimum mask area in pixels
        consensus_threshold: Voting threshold (0.5 = majority)
    
    Returns:
        (consensus_mask_uint8, mean_confidence) or None
    """
    if augmentations is None:
        augmentations = GEOMETRIC_AUGMENTATIONS
    
    h, w = crop.shape[:2]
    collected_masks = []
    confidences = []
    
    for aug_name in augmentations:
        result = apply_augmentation(crop, aug_name)
        if result[0] is None:
            continue
        
        aug_image, inv_mask_fn, _ = result
        
        try:
            results = model(aug_image, conf=conf, verbose=False)
        except Exception as e:
            logger.warning(f"TTA inference failed for {aug_name}: {e}")
            continue
        
        if not results or results[0].masks is None:
            continue
        
        res = results[0]
        if res.masks.data is None or len(res.masks.data) == 0:
            continue
        
        # Take best confidence mask
        best_idx = int(res.boxes.conf.argmax())
        mask_tensor = res.masks.data[best_idx]
        conf_val = float(res.boxes.conf[best_idx])
        
        # Resize to crop dimensions
        mask_np = mask_tensor.cpu().numpy()
        mask_resized = cv2.resize(
            mask_np, (aug_image.shape[1], aug_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        mask_bin = (mask_resized > 0.5).astype(np.uint8) * 255
        
        # Inverse transform back to original space
        mask_original = inv_mask_fn(mask_bin)
        
        # Ensure correct size (should be, but safety check)
        if mask_original.shape[:2] != (h, w):
            mask_original = cv2.resize(
                mask_original, (w, h), interpolation=cv2.INTER_NEAREST
            )
        
        collected_masks.append(mask_original)
        confidences.append(conf_val)
    
    if not collected_masks:
        return None
    
    # Merge by consensus
    consensus = merge_masks_consensus(collected_masks, threshold=consensus_threshold)
    
    if consensus is None or cv2.countNonZero(consensus) < min_mask_area:
        return None
    
    mean_conf = float(np.mean(confidences))
    logger.debug(
        f"TTA segment: {len(collected_masks)}/{len(augmentations)} augs succeeded, "
        f"conf_mean={mean_conf:.3f}"
    )
    
    return consensus, mean_conf


# =============================================================================
# DETECTION-LEVEL TTA (for bounding box detections)
# =============================================================================

def _compute_iou(box1, box2):
    """IoU between two axis-aligned boxes (x1,y1,x2,y2)."""
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


def _nms_indices(boxes, scores, iou_threshold=0.5):
    """Simple NMS, returns indices to keep."""
    if not boxes:
        return []
    
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    
    while idxs:
        best = idxs.pop(0)
        keep.append(best)
        remaining = []
        for i in idxs:
            if _compute_iou(boxes[best], boxes[i]) < iou_threshold:
                remaining.append(i)
        idxs = remaining
    
    return keep


def tta_yolo_detect(
    model,
    image: np.ndarray,
    conf: float,
    augmentations: List[str] = None,
    nms_iou: float = 0.5,
    cluster_iou: float = 0.3,
) -> List[Dict]:
    """
    Run YOLO detection with TTA and return consensus detections.
    
    For each augmentation:
    1. Transform image
    2. Run detection
    3. Inverse-transform bboxes
    4. Aggregate all detections via weighted NMS
    
    Args:
        model: YOLO detection model
        image: Input image (BGR)
        conf: Confidence threshold
        augmentations: List of augmentation names
        nms_iou: IoU threshold for NMS
        cluster_iou: IoU threshold to define a cluster for averaging
    
    Returns:
        List of dicts: [{'bbox': (x1,y1,x2,y2), 'class': int, 'value': int,
                         'confidence': float, 'center_x': float, 'center_y': float}]
    """
    if augmentations is None:
        augmentations = DETECT_TTA_AUGMENTATIONS
    
    h, w = image.shape[:2]
    all_detections = []  # (x1, y1, x2, y2, score, cls, value_or_none)
    
    for aug_name in augmentations:
        result = apply_augmentation(image, aug_name)
        if result[0] is None:
            continue
        
        aug_image, _, inv_bbox_fn = result
        
        try:
            results = model(aug_image, conf=conf, verbose=False)
        except Exception as e:
            logger.warning(f"TTA detect failed for {aug_name}: {e}")
            continue
        
        for res in results:
            if res.boxes is None:
                continue
            for box in res.boxes:
                cls = int(box.cls[0])
                score = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                
                # Inverse transform
                ox1, oy1, ox2, oy2 = inv_bbox_fn(x1, y1, x2, y2)
                
                # Clamp
                ox1 = max(0, min(w - 1, ox1))
                oy1 = max(0, min(h - 1, oy1))
                ox2 = max(0, min(w - 1, ox2))
                oy2 = max(0, min(h - 1, oy2))
                
                if ox2 > ox1 and oy2 > oy1:
                    all_detections.append({
                        'bbox': (ox1, oy1, ox2, oy2),
                        'score': score / len(augmentations),
                        'cls': cls,
                    })
    
    if not all_detections:
        return []
    
    # Group by class
    classes = set(d['cls'] for d in all_detections)
    final = []
    
    for cls in classes:
        cls_dets = [d for d in all_detections if d['cls'] == cls]
        boxes = [d['bbox'] for d in cls_dets]
        scores = [d['score'] for d in cls_dets]
        
        keep_idxs = _nms_indices(boxes, scores, nms_iou)
        
        for idx in keep_idxs:
            anchor = cls_dets[idx]
            ax1, ay1, ax2, ay2 = anchor['bbox']
            
            # Average cluster
            cluster = []
            cluster_scores = []
            for d in cls_dets:
                iou = _compute_iou(anchor['bbox'], d['bbox'])
                if iou > cluster_iou:
                    cluster.append(d['bbox'])
                    cluster_scores.append(d['score'])
            
            if cluster:
                total = sum(cluster_scores)
                avg_x1 = sum(b[0] * s for b, s in zip(cluster, cluster_scores)) / total
                avg_y1 = sum(b[1] * s for b, s in zip(cluster, cluster_scores)) / total
                avg_x2 = sum(b[2] * s for b, s in zip(cluster, cluster_scores)) / total
                avg_y2 = sum(b[3] * s for b, s in zip(cluster, cluster_scores)) / total
                bbox = (avg_x1, avg_y1, avg_x2, avg_y2)
            else:
                bbox = anchor['bbox']
            
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            
            final.append({
                'bbox': tuple(int(v) for v in bbox),
                'class': cls,
                'confidence': sum(cluster_scores) if cluster else anchor['score'],
                'center_x': cx,
                'center_y': cy,
                'n_votes': len(cluster) if cluster else 1,
            })
    
    return final


def _inverse_transform_points(points: np.ndarray, aug_name: str, h: int, w: int) -> np.ndarray:
    """Inverse geometric transform for Nx2 points back to original image space."""
    pts = points.astype(np.float32).copy()

    if aug_name == 'original':
        pts[:, 0] = np.clip(pts[:, 0], 0.0, float(w - 1))
        pts[:, 1] = np.clip(pts[:, 1], 0.0, float(h - 1))
        return pts
    if aug_name == 'flip_h':
        pts[:, 0] = (w - 1) - pts[:, 0]
        pts[:, 0] = np.clip(pts[:, 0], 0.0, float(w - 1))
        pts[:, 1] = np.clip(pts[:, 1], 0.0, float(h - 1))
        return pts
    if aug_name == 'flip_v':
        pts[:, 1] = (h - 1) - pts[:, 1]
        pts[:, 0] = np.clip(pts[:, 0], 0.0, float(w - 1))
        pts[:, 1] = np.clip(pts[:, 1], 0.0, float(h - 1))
        return pts
    if aug_name == 'rot180':
        pts[:, 0] = (w - 1) - pts[:, 0]
        pts[:, 1] = (h - 1) - pts[:, 1]
        pts[:, 0] = np.clip(pts[:, 0], 0.0, float(w - 1))
        pts[:, 1] = np.clip(pts[:, 1], 0.0, float(h - 1))
        return pts
    # Photometric augs (or unknown) have no geometric change.
    pts[:, 0] = np.clip(pts[:, 0], 0.0, float(w - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0.0, float(h - 1))
    return pts


def _canonical_obb_from_points(points: np.ndarray) -> Dict:
    """Build a canonical OBB representation from 4+ points via minAreaRect."""
    pts = points.astype(np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect

    length = max(w, h)
    width = min(w, h)
    adjusted_angle = float(angle)
    if w > h:
        adjusted_angle += 90.0

    box_pts = cv2.boxPoints(rect).astype(np.float32)
    return {
        'obb_points': box_pts,
        'center': (float(cx), float(cy)),
        'width': float(width),
        'height': float(length),
        'angle': adjusted_angle,
    }


def _obb_iou_dict(det_a: Dict, det_b: Dict) -> float:
    """IoU between two oriented boxes represented as dicts."""
    rect_a = (det_a['center'], (det_a['width'], det_a['height']), det_a['angle'])
    rect_b = (det_b['center'], (det_b['width'], det_b['height']), det_b['angle'])

    try:
        ret, pts = cv2.rotatedRectangleIntersection(rect_a, rect_b)
        if ret == cv2.INTERSECT_NONE or pts is None:
            return 0.0
        inter = cv2.contourArea(pts)
        area_a = det_a['width'] * det_a['height']
        area_b = det_b['width'] * det_b['height']
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0
    except Exception:
        return 0.0


def _nms_obb_dict(detections: List[Dict], iou_threshold: float) -> List[Dict]:
    """Simple OBB NMS on dict detections."""
    if len(detections) <= 1:
        return detections

    dets = sorted(detections, key=lambda d: d.get('confidence', 0.0), reverse=True)
    kept = []
    while dets:
        best = dets.pop(0)
        kept.append(best)
        remaining = []
        for det in dets:
            same_class = det.get('class_id') == best.get('class_id')
            if (not same_class) or _obb_iou_dict(best, det) < iou_threshold:
                remaining.append(det)
        dets = remaining
    return kept


def _weighted_consensus_obb(cluster: List[Dict]) -> Dict:
    """Compute confidence-weighted OBB consensus from a cluster."""
    if len(cluster) == 1:
        det = cluster[0].copy()
        det['n_votes'] = 1
        det['support_ratio'] = 1.0
        return det

    weights = np.array([
        max(1e-6, float(d.get('vote_weight', d.get('confidence', 0.0))))
        for d in cluster
    ], dtype=np.float64)
    total_w = float(weights.sum())

    cx = float(np.average([d['center'][0] for d in cluster], weights=weights))
    cy = float(np.average([d['center'][1] for d in cluster], weights=weights))
    width = float(np.average([d['width'] for d in cluster], weights=weights))
    height = float(np.average([d['height'] for d in cluster], weights=weights))

    # Orientation robuste via PCA pondérée sur les coins OBB.
    # Cela évite l'ambiguïté ±90° des angles de rectangles orientés.
    corner_points = []
    corner_weights = []
    for det, w in zip(cluster, weights):
        pts = np.array(det.get('obb_points', []), dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] >= 4:
            corner_points.append(pts)
            corner_weights.extend([float(w)] * pts.shape[0])

    if corner_points:
        pts_all = np.vstack(corner_points)
        w_all = np.array(corner_weights, dtype=np.float64)
        w_sum = float(w_all.sum())
        if w_sum > 0:
            mu = np.average(pts_all, axis=0, weights=w_all)
            centered = pts_all - mu
            cov = (centered * w_all[:, None]).T @ centered / w_sum
            eigvals, eigvecs = np.linalg.eigh(cov)
            major_vec = eigvecs[:, int(np.argmax(eigvals))]
            mean_angle = float(np.rad2deg(np.arctan2(major_vec[1], major_vec[0])))
        else:
            mean_angle = float(cluster[0].get('angle', 0.0))
    else:
        mean_angle = float(cluster[0].get('angle', 0.0))

    # Build box from weighted parameters, then canonicalize with minAreaRect.
    major = np.array([np.cos(np.deg2rad(mean_angle)), np.sin(np.deg2rad(mean_angle))], dtype=np.float32)
    minor = np.array([-major[1], major[0]], dtype=np.float32)
    c = np.array([cx, cy], dtype=np.float32)
    half_h = 0.5 * height
    half_w = 0.5 * width
    box_pts = np.array([
        c - major * half_h - minor * half_w,
        c - major * half_h + minor * half_w,
        c + major * half_h + minor * half_w,
        c + major * half_h - minor * half_w,
    ], dtype=np.float32)

    canonical = _canonical_obb_from_points(box_pts)
    confidence_mean = float(np.mean([d.get('confidence', 0.0) for d in cluster]))
    confidence_boost = float(min(1.0, total_w / max(1e-6, len(cluster))))
    confidence_final = float(max(0.0, min(1.0, confidence_mean * (0.5 + 0.5 * confidence_boost))))

    return {
        **canonical,
        'class_id': int(cluster[0]['class_id']),
        'class_name': cluster[0].get('class_name', f"class_{int(cluster[0]['class_id'])}"),
        'confidence': confidence_final,
        'n_votes': int(len(cluster)),
        'source': 'tta_consensus',
    }


def tta_yolo_obb(
    model,
    image: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int = 1024,
    augmentations: Optional[List[str]] = None,
    augmentation_weights: Optional[Dict[str, float]] = None,
    cluster_iou: float = 0.25,
    nms_iou: float = 0.45,
    min_votes: int = 1,
) -> List[Dict]:
    """Custom TTA for YOLO-OBB with confidence-weighted geometric consensus.

    Workflow:
    1) Run inference for each augmentation image.
    2) Back-transform each OBB to original image coordinates.
    3) Cluster overlapping OBBs per class.
    4) Build a confidence-weighted consensus OBB for each cluster.
    5) Apply final OBB NMS.
    """
    if augmentations is None:
        augmentations = GEOMETRIC_AUGMENTATIONS

    h, w = image.shape[:2]
    all_dets: List[Dict] = []

    for aug_name in augmentations:
        aug_image, _, _ = apply_augmentation(image, aug_name)
        if aug_image is None:
            continue

        try:
            results = model.predict(
                source=aug_image,
                conf=conf,
                iou=iou,
                verbose=False,
                half=False,
                augment=False,
                imgsz=imgsz,
            )
        except Exception as e:
            logger.warning(f"TTA OBB failed for {aug_name}: {e}")
            continue

        aug_weight = 1.0
        if augmentation_weights:
            aug_weight = float(augmentation_weights.get(aug_name, 1.0))

        for res in results:
            if hasattr(res, 'obb') and res.obb is not None and res.obb.data is not None:
                pts_batch = None
                conf_batch = None
                cls_batch = None

                if hasattr(res.obb, 'xyxyxyxy') and res.obb.xyxyxyxy is not None:
                    try:
                        pts_batch = res.obb.xyxyxyxy.cpu().numpy()
                    except Exception:
                        pts_batch = None

                if hasattr(res.obb, 'conf') and res.obb.conf is not None:
                    try:
                        conf_batch = res.obb.conf.cpu().numpy()
                    except Exception:
                        conf_batch = None

                if hasattr(res.obb, 'cls') and res.obb.cls is not None:
                    try:
                        cls_batch = res.obb.cls.cpu().numpy()
                    except Exception:
                        cls_batch = None

                if pts_batch is not None and len(pts_batch) > 0:
                    n = len(pts_batch)
                    for i_row in range(n):
                        pts_aug = np.array(pts_batch[i_row], dtype=np.float32).reshape(-1, 2)
                        if pts_aug.shape[0] < 4:
                            continue

                        if conf_batch is not None and i_row < len(conf_batch):
                            conf_val = float(conf_batch[i_row])
                        else:
                            conf_val = 0.5

                        if cls_batch is not None and i_row < len(cls_batch):
                            class_id = int(cls_batch[i_row])
                        else:
                            class_id = -1

                        pts_orig = _inverse_transform_points(pts_aug, aug_name, h, w)
                        pts_orig[:, 0] = np.clip(pts_orig[:, 0], 0.0, float(w - 1))
                        pts_orig[:, 1] = np.clip(pts_orig[:, 1], 0.0, float(h - 1))
                        canonical = _canonical_obb_from_points(pts_orig)

                        all_dets.append({
                            **canonical,
                            'class_id': class_id,
                            'class_name': f"class_{class_id}",
                            'confidence': conf_val,
                            'vote_weight': max(1e-6, conf_val * aug_weight),
                            'aug_name': aug_name,
                            'source': 'tta_raw',
                        })
                else:
                    # Fallback: reconstruire les coins depuis cx,cy,w,h,angle.
                    obb_data = res.obb.data
                    for i_row in range(len(obb_data)):
                        row = obb_data[i_row].cpu().numpy()
                        cx, cy, bw, bh, angle_rad, confidence, class_id = row
                        class_id = int(class_id)
                        angle_deg = float(angle_rad) * 180.0 / np.pi
                        rect = ((float(cx), float(cy)), (float(bw), float(bh)), angle_deg)
                        pts_aug = cv2.boxPoints(rect).astype(np.float32)
                        pts_orig = _inverse_transform_points(pts_aug, aug_name, h, w)
                        pts_orig[:, 0] = np.clip(pts_orig[:, 0], 0.0, float(w - 1))
                        pts_orig[:, 1] = np.clip(pts_orig[:, 1], 0.0, float(h - 1))
                        canonical = _canonical_obb_from_points(pts_orig)
                        conf_val = float(confidence)

                        all_dets.append({
                            **canonical,
                            'class_id': class_id,
                            'class_name': f"class_{class_id}",
                            'confidence': conf_val,
                            'vote_weight': max(1e-6, conf_val * aug_weight),
                            'aug_name': aug_name,
                            'source': 'tta_raw',
                        })
            elif hasattr(res, 'boxes') and res.boxes is not None:
                for box in res.boxes:
                    class_id = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    pts_aug = np.array(
                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        dtype=np.float32,
                    )
                    pts_orig = _inverse_transform_points(pts_aug, aug_name, h, w)
                    canonical = _canonical_obb_from_points(pts_orig)
                    all_dets.append({
                        **canonical,
                        'class_id': class_id,
                        'class_name': f"class_{class_id}",
                        'confidence': conf_val,
                        'vote_weight': max(1e-6, conf_val * aug_weight),
                        'aug_name': aug_name,
                        'source': 'tta_raw',
                    })

    if not all_dets:
        return []

    final_consensus: List[Dict] = []
    classes = sorted(set(int(d['class_id']) for d in all_dets))

    for class_id in classes:
        class_dets = [d for d in all_dets if int(d['class_id']) == class_id]
        class_dets = sorted(class_dets, key=lambda d: d.get('vote_weight', d.get('confidence', 0.0)), reverse=True)
        consumed = set()

        for idx, anchor in enumerate(class_dets):
            if idx in consumed:
                continue

            cluster_indices = [idx]
            for j in range(idx + 1, len(class_dets)):
                if j in consumed:
                    continue
                if _obb_iou_dict(anchor, class_dets[j]) >= cluster_iou:
                    cluster_indices.append(j)

            cluster = [class_dets[k] for k in cluster_indices]
            for k in cluster_indices:
                consumed.add(k)

            if len(cluster) < min_votes:
                continue

            consensus = _weighted_consensus_obb(cluster)
            consensus['support_ratio'] = float(len(cluster)) / float(max(1, len(augmentations)))
            final_consensus.append(consensus)

    final_consensus = _nms_obb_dict(final_consensus, iou_threshold=nms_iou)
    logger.debug(
        "TTA OBB custom: %d raw dets from %d augs -> %d consensus dets",
        len(all_dets), len(augmentations), len(final_consensus),
    )
    return final_consensus


# =============================================================================
# DIGIT DETECTION TTA (for bag OCR)
# =============================================================================

def tta_detect_digits(
    model,
    bag_roi: np.ndarray,
    conf: float,
    class_to_value: Dict[int, int],
    augmentations: List[str] = None,
    nms_iou: float = 0.5,
    device: str = '0',
) -> List[Dict]:
    """
    Run digit detection with TTA and return consensus digits.
    
    Strategy: run detection on multiple augmented versions, project back
    to original space, group by position, vote on digit values.
    
    Args:
        model: YOLO digit detection model
        bag_roi: Bag ROI image (BGR)
        conf: Confidence threshold
        class_to_value: Mapping class_id → digit value
        augmentations: List of augmentation names
        nms_iou: IoU threshold for grouping
        device: Device for inference
    
    Returns:
        List of dicts with 'class', 'value', 'center_x', 'center_y',
        'bbox', 'confidence', 'n_votes'
    """
    if augmentations is None:
        augmentations = GEOMETRIC_AUGMENTATIONS + ['brightness_up', 'contrast_up']
    
    h, w = bag_roi.shape[:2]
    all_dets = []
    
    for aug_name in augmentations:
        result = apply_augmentation(bag_roi, aug_name)
        if result[0] is None:
            continue
        
        aug_image, _, inv_bbox_fn = result
        
        try:
            results = model(aug_image, verbose=False, conf=conf, device=device)
        except Exception as e:
            logger.warning(f"TTA digit detect failed for {aug_name}: {e}")
            continue
        
        for res in results:
            if res.boxes is None:
                continue
            for box in res.boxes:
                cls = int(box.cls[0])
                score = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                ox1, oy1, ox2, oy2 = inv_bbox_fn(x1, y1, x2, y2)
                ox1 = max(0, min(w - 1, int(ox1)))
                oy1 = max(0, min(h - 1, int(oy1)))
                ox2 = max(0, min(w - 1, int(ox2)))
                oy2 = max(0, min(h - 1, int(oy2)))
                
                if ox2 > ox1 and oy2 > oy1:
                    all_dets.append({
                        'bbox': (ox1, oy1, ox2, oy2),
                        'score': score,
                        'cls': cls,
                        'value': class_to_value.get(cls, cls + 1),
                    })
    
    if not all_dets:
        return []
    
    # Cluster detections by position using NMS-like grouping
    boxes = [d['bbox'] for d in all_dets]
    scores = [d['score'] for d in all_dets]
    keep = _nms_indices(boxes, scores, nms_iou)
    
    final_dets = []
    used = set()
    
    for idx in keep:
        anchor = all_dets[idx]
        cluster = []
        
        for i, d in enumerate(all_dets):
            if i in used:
                continue
            iou = _compute_iou(anchor['bbox'], d['bbox'])
            if iou > 0.3:
                cluster.append(d)
                used.add(i)
        
        if not cluster:
            cluster = [anchor]
        
        # Majority vote on digit value, weighted by confidence
        value_scores = {}
        for d in cluster:
            v = d['value']
            value_scores[v] = value_scores.get(v, 0) + d['score']
        
        best_value = max(value_scores, key=value_scores.get)
        best_cls = next(d['cls'] for d in cluster if d['value'] == best_value)
        
        # Average bbox
        total_score = sum(d['score'] for d in cluster)
        avg_x1 = sum(d['bbox'][0] * d['score'] for d in cluster) / total_score
        avg_y1 = sum(d['bbox'][1] * d['score'] for d in cluster) / total_score
        avg_x2 = sum(d['bbox'][2] * d['score'] for d in cluster) / total_score
        avg_y2 = sum(d['bbox'][3] * d['score'] for d in cluster) / total_score
        
        cx = (avg_x1 + avg_x2) / 2
        cy = (avg_y1 + avg_y2) / 2
        
        final_dets.append({
            'class': best_cls,
            'value': best_value,
            'center_x': cx,
            'center_y': cy,
            'bbox': (int(avg_x1), int(avg_y1), int(avg_x2), int(avg_y2)),
            'confidence': total_score / len(cluster),
            'n_votes': len(cluster),
        })
    
    # Sort by center_y (top to bottom)
    final_dets.sort(key=lambda d: d['center_y'])
    
    return final_dets


# =============================================================================
# RACHIS SKELETON TTA
# =============================================================================

def tta_rachis_segment(
    model,
    crop: np.ndarray,
    conf: float,
    augmentations: List[str] = None,
    consensus_threshold: float = 0.4,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Run rachis segmentation with TTA.
    
    Like tta_yolo_segment but with a lower consensus threshold (0.4)
    since rachis is thin and sensitive to slight positional shifts.
    
    Returns:
        (consensus_mask_uint8, mean_confidence) or None
    """
    if augmentations is None:
        # For rachis, also include photometric to help with contrast
        augmentations = GEOMETRIC_AUGMENTATIONS + ['brightness_up', 'contrast_up']
    
    return tta_yolo_segment(
        model=model,
        crop=crop,
        conf=conf,
        augmentations=augmentations,
        min_mask_area=50,  # Rachis is thin
        consensus_threshold=consensus_threshold,
    )
