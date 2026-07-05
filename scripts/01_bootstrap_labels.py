"""
Active-learning bootstrap: auto-label your shelf images with the OpenFoodFacts
price-tag ONNX (the 'teacher'), write YOLO-format labels, split train/val + data.yaml.
The 'student' YOLO trained on these labels generalises to tag styles the teacher
partly misses (e.g. coloured rail strips).

Run: python scripts/01_bootstrap_labels.py
  RAW_DIR  input photos dir (default datasets/tags/images_raw)
  TAG_CONF teacher confidence threshold (default 0.20 — noisy-but-recalls beats silent)
"""
import os, glob, random, shutil
import numpy as np
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download

RAW = os.environ.get("RAW_DIR", "datasets/tags/images_raw")
OUT = "datasets/tags"
CONF = float(os.environ.get("TAG_CONF", "0.20")); IOU = 0.5; S = 960
random.seed(42)

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

def detect(sess, iname, img):
    # letterbox to SxS → infer → NMS → un-letterbox back to normalized YOLO (xc,yc,w,h)
    W,H = img.size; r=min(S/W,S/H); nw,nh=round(W*r),round(H*r); px,py=(S-nw)//2,(S-nh)//2
    cv=Image.new('RGB',(S,S),(114,114,114)); cv.paste(img.resize((nw,nh)),(px,py))
    x=(np.asarray(cv).astype('float32')/255).transpose(2,0,1)[None]
    o=sess.run(None,{iname:x})[0][0]
    if o.shape[0]<o.shape[1]: o=o.T
    sc=o[:,4:].max(1); m=sc>CONF; o=o[m]; sc=sc[m]
    if not len(sc): return []
    cx,cy,w,h=o[:,0],o[:,1],o[:,2],o[:,3]
    bx=np.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],1)
    keep=nms(bx,sc,IOU); bx=bx[keep]
    out=[]
    for (x1,y1,x2,y2) in bx:
        ox1=(x1-px)/r; oy1=(y1-py)/r; ox2=(x2-px)/r; oy2=(y2-py)/r
        ox1=max(0,min(W,ox1)); ox2=max(0,min(W,ox2)); oy1=max(0,min(H,oy1)); oy2=max(0,min(H,oy2))
        if ox2<=ox1+2 or oy2<=oy1+2: continue
        out.append(((ox1+ox2)/2/W,(oy1+oy2)/2/H,(ox2-ox1)/W,(oy2-oy1)/H))
    return out

def main():
    p=hf_hub_download('openfoodfacts/price-tag-detection','weights/model_ir_10_opset_19.onnx')
    sess=ort.InferenceSession(p,providers=['CPUExecutionProvider']); iname=sess.get_inputs()[0].name
    for sub in ['images/train','images/val','labels/train','labels/val']:
        os.makedirs(os.path.join(OUT,sub),exist_ok=True)
    files=sorted(glob.glob(os.path.join(RAW,'*.jpg'))); random.shuffle(files)
    if not files:
        print(f"no .jpg files in {RAW} — put your shelf photos there first"); return
    nval=max(1,len(files)//10)
    tot_tags=0; with_tags=0
    for i,fp in enumerate(files):
        split='val' if i<nval else 'train'
        try: img=Image.open(fp).convert('RGB')
        except Exception: continue
        boxes=detect(sess,iname,img)
        if boxes: with_tags+=1; tot_tags+=len(boxes)
        name=os.path.splitext(os.path.basename(fp))[0]
        shutil.copy(fp, os.path.join(OUT,f'images/{split}/{name}.jpg'))
        # empty label files are written on purpose: teacher-blank images are explicit negatives
        with open(os.path.join(OUT,f'labels/{split}/{name}.txt'),'w') as f:
            for (xc,yc,w,h) in boxes: f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        if (i+1)%100==0: print(f"  {i+1}/{len(files)} · {tot_tags} tags so far")
    with open(os.path.join(OUT,'data.yaml'),'w') as f:
        f.write("path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: price_tag\n")
    print(f"\n== bootstrap done ==")
    print(f"  images: {len(files)} ({with_tags} with ≥1 tag, {len(files)-with_tags} empty)")
    print(f"  total auto-labelled tags: {tot_tags} (~{tot_tags/max(1,with_tags):.1f}/labelled image)")
    print(f"  data.yaml → {OUT}/data.yaml")

if __name__ == "__main__":
    main()
