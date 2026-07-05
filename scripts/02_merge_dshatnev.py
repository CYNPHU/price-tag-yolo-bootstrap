"""
Convert the D-Shatnev/price-tag-detection HF dataset (parquet, foreign price tags)
to YOLO format and MERGE into datasets/tags/images|labels/train. Foreign data adds
volume + variety (a tag is a tag); YOUR val set stays the validation target so
metrics measure your own domain. bbox is already YOLO (xc,yc,w,h normalized).

Run: python scripts/02_merge_dshatnev.py
"""
import os, io
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
from PIL import Image

OUT = "datasets/tags"
os.makedirs(f"{OUT}/images/train", exist_ok=True); os.makedirs(f"{OUT}/labels/train", exist_ok=True)

def main():
    api = HfApi()
    files = [f for f in api.list_repo_files("D-Shatnev/price-tag-detection", repo_type="dataset")
             if f.endswith(".parquet")]
    print(f"{len(files)} parquet shards")
    n_img = n_box = 0
    for fp in files:
        local = hf_hub_download("D-Shatnev/price-tag-detection", fp, repo_type="dataset")
        rows = pq.read_table(local).to_pylist()
        for row in rows:
            try:
                img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            except Exception:
                continue
            name = "dshatnev_" + str(row["image_id"])
            img.save(f"{OUT}/images/train/{name}.jpg")
            with open(f"{OUT}/labels/train/{name}.txt", "w") as f:
                for b in (row["objects"]["bbox"] or []):
                    xc, yc, w, h = b
                    if w <= 0 or h <= 0: continue
                    f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"); n_box += 1
            n_img += 1
        print(f"  {fp.split('/')[-1]} → {n_img} imgs, {n_box} boxes")
    print(f"\n== merged {n_img} D-Shatnev images ({n_box} tags) into {OUT}/images/train ==")

if __name__ == "__main__":
    main()
