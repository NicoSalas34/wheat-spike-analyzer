#!/usr/bin/env python3
"""Collect all results.json from output/ into a single CSV (one row per spike)."""

import csv
import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
CSV_PATH = OUTPUT_DIR / "results_summary.csv"

# ── helpers ────────────────────────────────────────────────────────────────
def safe(d, *keys, default=None):
    """Nested dict accessor."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def build_row(image_name, data, spike):
    """Build one CSV row from a spike entry."""
    m = spike.get("measurements", {})
    ss = spike.get("spikelet_stats") or {}
    rachis = spike.get("rachis") or {}
    ia = spike.get("insertion_angle_stats") or {}
    wp = spike.get("width_profile") or {}
    bag = data.get("bag") or {}
    cal = data.get("calibration") or {}

    return {
        # identification
        "image": image_name,
        "spike_id": spike.get("id"),
        "sample_id": bag.get("sample_id"),
        "bac": bag.get("bac"),
        "ligne": bag.get("ligne"),
        "colonne": bag.get("colonne"),
        "bag_confidence": bag.get("confidence"),
        # calibration
        "pixel_per_mm": cal.get("pixel_per_mm"),
        "calibration_method": cal.get("calibration_method"),
        # spike OBB
        "spike_length_mm": m.get("spike_length_mm"),
        "spike_width_mm": m.get("spike_width_mm"),
        "whole_spike_length_mm": m.get("whole_spike_length_mm"),
        "whole_spike_width_mm": m.get("whole_spike_width_mm"),
        "length_mm": m.get("length_mm"),
        "width_mm": m.get("width_mm"),
        "area_mm2": m.get("area_mm2"),
        "perimeter_mm": m.get("perimeter_mm"),
        "aspect_ratio": m.get("aspect_ratio"),
        "angle_degrees": m.get("angle_degrees"),
        "has_awns": m.get("has_awns"),
        "awns_length_mm": m.get("awns_length_mm"),
        # segmentation-derived
        "seg_length_mm": m.get("seg_length_mm"),
        "seg_width_mm": m.get("seg_width_mm"),
        "seg_width_mean_mm": m.get("seg_width_mean_mm"),
        "seg_width_max_mm": m.get("seg_width_max_mm"),
        "seg_width_min_mm": m.get("seg_width_min_mm"),
        "seg_aspect_ratio": m.get("seg_aspect_ratio"),
        "real_area_mm2": m.get("real_area_mm2"),
        "real_perimeter_mm": m.get("real_perimeter_mm"),
        "circularity": m.get("circularity"),
        "solidity": m.get("solidity"),
        "ellipse_major_mm": m.get("ellipse_major_axis_mm"),
        "ellipse_minor_mm": m.get("ellipse_minor_axis_mm"),
        "ellipse_eccentricity": m.get("ellipse_eccentricity"),
        # width profile / shape
        "shape_class": wp.get("shape_class") if wp else None,
        # color
        "color_L": safe(spike, "color", "L_mean"),
        "color_a": safe(spike, "color", "a_mean"),
        "color_b": safe(spike, "color", "b_mean"),
        "color_H": safe(spike, "color", "H_mean"),
        "color_S": safe(spike, "color", "S_mean"),
        "color_V": safe(spike, "color", "V_mean"),
        # spikelets
        "spikelet_count": spike.get("spikelet_count"),
        "spikelet_method": spike.get("spikelet_method"),
        "spikelet_confidence": spike.get("spikelet_confidence"),
        "spikelet_density_per_cm": spike.get("spikelet_density_per_cm"),
        # spikelet morpho stats (mm)
        "spikelet_length_mm_mean": ss.get("spikelet_length_mm_mean"),
        "spikelet_length_mm_std": ss.get("spikelet_length_mm_std"),
        "spikelet_width_mm_mean": ss.get("spikelet_width_mm_mean"),
        "spikelet_width_mm_std": ss.get("spikelet_width_mm_std"),
        "spikelet_area_mm2_mean": ss.get("spikelet_area_mm2_mean"),
        "spikelet_area_mm2_std": ss.get("spikelet_area_mm2_std"),
        "spikelet_aspect_ratio_mean": ss.get("spikelet_aspect_ratio_mean"),
        "spikelet_circularity_mean": ss.get("spikelet_circularity_mean"),
        "spikelet_solidity_mean": ss.get("spikelet_solidity_mean"),
        "spikelet_length_cv": ss.get("spikelet_length_cv"),
        "spikelet_n_segmented": ss.get("n_segmented"),
        # rachis
        "rachis_detected": rachis.get("detected"),
        "rachis_length_mm": rachis.get("length_mm"),
        "rachis_confidence": rachis.get("confidence"),
        # insertion angles
        "insertion_angle_mean": m.get("insertion_angle_mean"),
        "insertion_angle_std": m.get("insertion_angle_std"),
        "insertion_angle_min": m.get("insertion_angle_min"),
        "insertion_angle_max": m.get("insertion_angle_max"),
        "spikelets_left": m.get("spikelets_left"),
        "spikelets_right": m.get("spikelets_right"),
        # detection meta
        "detection_confidence": m.get("confidence"),
        "detection_type": m.get("detection_type"),
        "has_segmentation": spike.get("has_segmentation"),
    }


# ── main ───────────────────────────────────────────────────────────────────
def main():
    rows = []
    json_files = sorted(OUTPUT_DIR.glob("*/results.json"))
    print(f"Found {len(json_files)} results.json files in {OUTPUT_DIR}")

    for jf in json_files:
        image_name = jf.parent.name
        try:
            data = json.loads(jf.read_text())
        except Exception as e:
            print(f"  SKIP {image_name}: {e}")
            continue

        spikes = data.get("spikes", [])
        if not spikes:
            # Still emit a row so the image appears in the CSV
            rows.append({
                "image": image_name,
                "spike_id": None,
                "sample_id": safe(data, "bag", "sample_id"),
                "bac": safe(data, "bag", "bac"),
                "ligne": safe(data, "bag", "ligne"),
                "colonne": safe(data, "bag", "colonne"),
                "pixel_per_mm": safe(data, "calibration", "pixel_per_mm"),
                "spikelet_count": 0,
            })
            continue

        for spike in spikes:
            rows.append(build_row(image_name, data, spike))

    if not rows:
        print("No results found.")
        return

    # Determine all columns from all rows (union of keys)
    all_keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ CSV written: {CSV_PATH}")
    print(f"  {len(rows)} rows, {len(all_keys)} columns")

    # Quick stats
    n_images = len(set(r["image"] for r in rows))
    n_spikes = sum(1 for r in rows if r.get("spikelet_count") is not None)
    n_with_id = sum(1 for r in rows if r.get("sample_id"))
    n_with_rachis = sum(1 for r in rows if r.get("rachis_detected"))
    print(f"  {n_images} images, {n_spikes} spikes, {n_with_id} with sample_id, {n_with_rachis} with rachis")


if __name__ == "__main__":
    main()
