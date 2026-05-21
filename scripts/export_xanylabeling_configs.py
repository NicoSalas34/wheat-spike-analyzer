"""Export project .pt models to ONNX + write xanylabeling custom-model YAMLs."""
from pathlib import Path
import torch, yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

# (filename, xanylabeling type, extra-fields-fn)
# type is decided from the head class + train_args.task + train_args.model arch family
SPEC = {
    "wheat_spike_yolo.pt":   "yolo26_obb",
    "graduations_yolo.pt":   "yolo11_obb",
    "spikelets_yolo.pt":     "yolo26_seg",
    "spike_seg_yolo.pt":     "yolo26_seg",
    "rachis_yolo.pt":        "yolo26_seg",
    "rachis_yolo_pose.pt":   "yolo11_pose",
    "bag_digits_yolo.pt":    "yolo11",
    "bag_opening_yolo.pt":   "yolo26",
}

def read_meta(pt_path: Path):
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    m = ckpt.get("model", ckpt)
    names = m.names if hasattr(m, "names") else {}
    kpt_shape = getattr(m.model[-1], "kpt_shape", None)
    return names, kpt_shape

def build_yaml(pt: Path, onnx: Path, mtype: str, names: dict, kpt_shape):
    base = {
        "type": mtype,
        "name": f"{pt.stem}-r20260521",
        "provider": "Ultralytics",
        "display_name": pt.stem.replace("_", " ").title(),
        "model_path": str(onnx),
        "conf_threshold": 0.25,
    }
    base["iou_threshold"] = 0.45
    if mtype.startswith("yolo26"):
        base["max_det"] = 300

    class_list = [str(names[i]) for i in sorted(names)]

    if mtype.endswith("_pose"):
        k = kpt_shape[0] if kpt_shape else 0
        base["kpt_threshold"] = 0.25
        base["has_visible"] = bool(kpt_shape and len(kpt_shape) > 1 and kpt_shape[1] == 3)
        base["classes"] = {cn: [f"kpt{i}" for i in range(k)] for cn in class_list}
    else:
        base["classes"] = class_list
    return base

def main():
    for fname, mtype in SPEC.items():
        pt = MODELS / fname
        onnx = pt.with_suffix(".onnx")
        ycfg = pt.with_suffix(".yaml")
        names, kpt_shape = read_meta(pt)
        if not onnx.exists():
            print(f"[export] {fname} -> {onnx.name}")
            YOLO(str(pt)).export(format="onnx", opset=12, dynamic=False, simplify=False)
        else:
            print(f"[skip export] {onnx.name} already exists")
        cfg = build_yaml(pt, onnx, mtype, names, kpt_shape)
        with open(ycfg, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f"[yaml]   wrote {ycfg.name} (type={mtype}, {len(names)} class(es))")

if __name__ == "__main__":
    main()
