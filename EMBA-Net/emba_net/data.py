"""Small paired image/YOLO-label dataset; no automatic split or augmentation."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .inference import letterbox


class DetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir, size=640, nc=1):
        self.images = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
        self.label_dir, self.size, self.nc = Path(label_dir), size, nc
        if not self.images:
            raise ValueError(f"No images found in {image_dir}.")
        missing = [p.name for p in self.images if not (self.label_dir/(p.stem+".txt")).is_file()]
        if missing:
            raise ValueError(f"Missing labels, first examples: {missing[:5]}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        p = self.images[index]
        with Image.open(p) as src:
            image = src.convert("RGB")
        image_tensor, ratios, pad = letterbox(image, self.size)
        rows=[]
        for line in (self.label_dir/(p.stem+".txt")).read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            values = [float(v) for v in line.split()]
            if len(values)!=5 or not np.isfinite(values).all():
                raise ValueError(f"Invalid label row for {p.name}.")
            c,x,y,w,h=values
            if not c.is_integer() or not 0<=c<self.nc or not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1):
                raise ValueError(f"Invalid class or box for {p.name}.")
            rows.append([c,(x*image.width*ratios[0]+pad[0])/self.size,
                         (y*image.height*ratios[1]+pad[1])/self.size,
                         w*image.width*ratios[0]/self.size,h*image.height*ratios[1]/self.size])
        return image_tensor, torch.tensor(rows,dtype=torch.float32).reshape(-1,5)


def collate_batch(items):
    images, labels = zip(*items)
    classes, boxes, indices = [], [], []
    for i, label in enumerate(labels):
        classes.append(label[:,:1]); boxes.append(label[:,1:])
        indices.append(torch.full((len(label),),i,dtype=torch.float32))
    return {"img":torch.stack(images), "cls":torch.cat(classes), "bboxes":torch.cat(boxes), "batch_idx":torch.cat(indices)}
