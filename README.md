# price-tag-yolo-bootstrap

Train a domain-specific **price-tag detector** for retail shelf photos **without hand-labeling a single box**, using teacher–student bootstrapping over public models and public data.

Extracted from a production retail-execution SaaS (AI shelf audit for FMCG field sales). Client images and data are not included — the pipeline runs end-to-end on public data plus any shelf photos you provide.

## Why

Generic price-tag detectors exist (e.g. the OpenFoodFacts model), but they under-detect domain-specific tag styles — in our case, yellow rail-strip tags common in Thai convenience stores. Hand-labeling thousands of boxes is the usual answer; this pipeline avoids it:

```
public teacher ONNX ──auto-label──> your shelf photos ─┐
                                                       ├─> YOLOv11 student ──export──> best.onnx
public dataset (HF) ──convert/merge──> extra volume  ──┘         │
                                                        validated ONLY on your own domain
```

1. **Bootstrap labels** — run the [OpenFoodFacts price-tag ONNX](https://huggingface.co/openfoodfacts/price-tag-detection) as a *teacher* over your raw shelf photos; write YOLO-format labels (letterbox → inference → NMS → un-letterbox, all in numpy).
2. **Merge public data** — convert the [D-Shatnev/price-tag-detection](https://huggingface.co/datasets/D-Shatnev/price-tag-detection) parquet dataset to YOLO format and merge into the train split. Foreign tags add volume and variety (a tag is a tag) — but they stay **out of the val split**, so metrics measure *your* domain only.
3. **Train the student** — fine-tune YOLOv11s with Ultralytics. The student generalises past the teacher: it learns tag styles the teacher partially misses, because the merged data and augmentation regularise away the teacher's blind spots.
4. **Eyeball the result** — run any image through the exported ONNX and get a numbered crop montage for quick qualitative review.

The trained detector runs in production via `onnxruntime-node` inside a Node.js web server — no Python at inference time.

## Run it

```bash
pip install -r requirements.txt

# 1. put your shelf photos in datasets/tags/images_raw/  (any .jpg)
python scripts/01_bootstrap_labels.py

# 2. merge the public dataset (downloads from Hugging Face)
python scripts/02_merge_dshatnev.py

# 3. train (needs a CUDA GPU; tuned defaults for an 8GB card)
python scripts/03_train.py
#    env overrides: YOLO_MODEL=yolo11s.pt YOLO_EPOCHS=60 YOLO_IMGSZ=960 YOLO_BATCH=12

# 4. sanity-check detections on any image
python scripts/04_probe_onnx.py path/to/shelf.jpg --onnx runs/detect/tags/weights/best.onnx
```

`03_train.py` prints `val mAP50 / mAP50-95` at the end — measured on **your** val images only.

## Design notes

- **Val split is domain-pure.** The public D-Shatnev images are merged into *train only*. If foreign tags leaked into val, the metric would flatter the model on data you don't care about.
- **Teacher confidence is a knob, not a truth.** `TAG_CONF` (default 0.20) trades label recall vs noise. A slightly noisy teacher is fine — the student averages over it; a silent teacher (threshold too high) starves training.
- **Empty labels are kept on purpose.** Images where the teacher finds nothing still enter training as explicit negatives.
- **Caches are pinned into the project** (`HF_HOME`, `TORCH_HOME`, Ultralytics settings) so datasets, weights and runs stay on the fast disk you chose, and the repo is fully relocatable.

## License

MIT
