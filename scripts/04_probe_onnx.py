"""
Qualitative check: run a price-tag ONNX on any shelf photo(s), draw a numbered
crop montage to eyeball detection quality. Works with the public teacher model
(default) or your trained student (--onnx runs/detect/tags/weights/best.onnx).

Run: python scripts/04_probe_onnx.py shelf1.jpg shelf2.jpg [--onnx path/to/best.onnx]
  TAG_CONF confidence threshold (default 0.25)
"""
import os, sys, math, argparse
import numpy as np
from PIL import Image, ImageDraw
import onnxruntime as ort
from huggingface_hub import hf_hub_download

CONF = float(os.environ.get("TAG_CONF", "0.25")); IOU = 0.5

def nms(b, s, thr):
    idx = s.argsort()[::-1]; keep = []
    while len(idx):
        i = idx[0]; keep.append(i)
        if len(idx) == 1: break
        xx1=np.maximum(b[i,0],b[idx[1:],0]); yy1=np.maximum(b[i,1],b[idx[1:],1])
        xx2=np.minimum(b[i,2],b[idx[1:],2]); yy2=np.minimum(b[i,3],b[idx[1:],3])
        w=np.maximum(0,xx2-xx1); h=np.maximum(0,yy2-yy1); inter=w*h
        a1=(b[i,2]-b[i,0])*(b[i,3]-b[i,1]); a2=(b[idx[1:],2]-b[idx[1:],0])*(b[idx[1:],3]-b[idx[1:],1])
        iou=inter/(a1+a2-inter+1e-6); idx=idx[1:][iou<thr]
    return keep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="shelf photo(s)")
    ap.add_argument("--onnx", default=None, help="detector ONNX (default: download OpenFoodFacts teacher)")
    args = ap.parse_args()

    p = args.onnx or hf_hub_download('openfoodfacts/price-tag-detection','weights/model_ir_10_opset_19.onnx')
    s = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
    iname = s.get_inputs()[0].name; ishape = s.get_inputs()[0].shape
    S = int(ishape[2]) if isinstance(ishape[2], int) else 640
    print(f"input {iname} {ishape} → size {S} · output {s.get_outputs()[0].shape}")

    for path in args.images:
        if not os.path.exists(path): print(f"  {path}: no image"); continue
        img = Image.open(path).convert("RGB"); W, H = img.size
        r = min(S/W, S/H); nw, nh = round(W*r), round(H*r); px, py = (S-nw)//2, (S-nh)//2
        cv = Image.new('RGB', (S, S), (114,114,114)); cv.paste(img.resize((nw,nh)), (px,py))
        x = (np.asarray(cv).astype('float32')/255).transpose(2,0,1)[None]
        out = s.run(None, {iname: x})[0]
        o = out[0]                                  # (4+nc, N) or (N, 4+nc)
        if o.shape[0] < o.shape[1]: o = o.T         # → (N, 4+nc)
        boxes_cxcywh = o[:, :4]; scores = o[:, 4:].max(1)
        m = scores > CONF; boxes_cxcywh = boxes_cxcywh[m]; scores = scores[m]
        if not len(scores): print(f"  {path}: 0 tags (max score {o[:,4:].max():.2f})"); continue
        cx,cy,w,h = boxes_cxcywh.T
        bx = np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], 1)
        keep = nms(bx, scores, IOU); bx = bx[keep]
        bx[:,[0,2]] = (bx[:,[0,2]]-px)/r; bx[:,[1,3]] = (bx[:,[1,3]]-py)/r
        bx[:,[0,2]] = bx[:,[0,2]].clip(0,W); bx[:,[1,3]] = bx[:,[1,3]].clip(0,H)
        crops = [img.crop((int(a),int(b2),int(c),int(d))) for a,b2,c,d in bx if c>a+2 and d>b2+2]
        print(f"  {path}: {len(crops)} tags (max score {scores.max():.2f})")
        cols=3; rows=math.ceil(len(crops)/cols); cw=260; ch=80
        M=Image.new("RGB",(cols*cw, rows*ch+rows*16),(255,255,255)); dr=ImageDraw.Draw(M)
        for k,c in enumerate(crops):
            rr,cc=divmod(k,cols); xx=cc*cw; yy=rr*(ch+16)
            th=c.copy(); th.thumbnail((cw-8,ch-8)); M.paste(th,(xx+4,yy+16)); dr.text((xx+4,yy+2),f"#{k}",fill=(200,0,0))
        op = os.path.splitext(path)[0] + ".tags.jpg"
        M.save(op); print(f"    montage → {op}")

if __name__ == "__main__":
    main()
