"""
Fine-tune a YOLOv11 'student' price-tag detector on the bootstrapped dataset.
Defaults are tuned for a consumer 8GB GPU (e.g. RTX 2060 Super).

Run: python scripts/03_train.py
Env tuning (optional):
  YOLO_MODEL=yolo11s.pt   YOLO_EPOCHS=60   YOLO_IMGSZ=960   YOLO_BATCH=12  (→8/4 if OOM)

Output ONNX: runs/detect/tags/weights/best.onnx
"""
import os

# ---- pin every cache inside the project so the repo is fully relocatable -----
ROOT = os.path.abspath(".")
CACHE = os.path.join(ROOT, ".cache")
os.environ.setdefault("HF_HOME", os.path.join(CACHE, "hf"))
os.environ.setdefault("TORCH_HOME", os.path.join(CACHE, "torch"))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(CACHE, "ultralytics"))
for d in (os.environ["HF_HOME"], os.environ["TORCH_HOME"], os.environ["YOLO_CONFIG_DIR"]):
    os.makedirs(d, exist_ok=True)

MODEL  = os.environ.get("YOLO_MODEL", "yolo11s.pt")
EPOCHS = int(os.environ.get("YOLO_EPOCHS", "60"))
IMGSZ  = int(os.environ.get("YOLO_IMGSZ", "960"))
BATCH  = int(os.environ.get("YOLO_BATCH", "12"))
DATA   = os.path.join(ROOT, "datasets", "tags", "data.yaml")

def main():
    import torch
    from ultralytics import YOLO, settings
    if not torch.cuda.is_available():
        print("⚠️  No CUDA GPU detected — training on CPU is impractical. Aborting."); return
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}  · caches: {CACHE}")

    settings.update({"datasets_dir": ROOT, "weights_dir": os.path.join(CACHE, "weights"),
                     "runs_dir": os.path.join(ROOT, "runs")})
    # ensure data.yaml uses an ABSOLUTE path (no ambiguity across cwd)
    txt = open(DATA, encoding="utf-8").read()
    if "path:" in txt:
        import re
        txt = re.sub(r"path:.*", "path: " + os.path.join(ROOT, "datasets", "tags").replace("\\", "/"), txt, count=1)
        open(DATA, "w", encoding="utf-8").write(txt)

    model = YOLO(MODEL)
    model.train(data=DATA, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                patience=15, project=os.path.join(ROOT, "runs", "detect"), name="tags",
                exist_ok=True, mosaic=1.0, close_mosaic=10)
    m = model.val()
    print(f"\nval mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}")
    onnx = model.export(format="onnx", imgsz=IMGSZ, opset=12, simplify=True, dynamic=False)
    print(f"\n✅ DONE. ONNX → {onnx}")

if __name__ == "__main__":
    main()
