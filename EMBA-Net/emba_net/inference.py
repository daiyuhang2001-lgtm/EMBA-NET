"""Image preprocessing and class-aware NMS for the focused model."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from .geometry import xywh2xyxy, box_iou


def letterbox(image, size=640):
    if size < 32 or size % 32:
        raise ValueError("Image size must be a positive multiple of 32.")
    w, h = image.size
    ratio = min(size/w, size/h)
    nw, nh = round(w*ratio), round(h*ratio)
    left, top = (size-nw)//2, (size-nh)//2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(image.convert("RGB").resize((nw, nh), Image.Resampling.BILINEAR), (left, top))
    arr = np.array(canvas, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0
    return torch.from_numpy(arr), (nw/w, nh/h), (left, top)


def non_max_suppression(prediction, conf=0.25, iou=0.7, max_det=300, max_nms=30000):
    """Input B,(4+nc),N; output per-image tensors of xyxy,score,class."""
    if not 0 <= conf <= 1 or not 0 <= iou <= 1:
        raise ValueError("Confidence and IoU thresholds must lie in [0,1].")
    output = []
    for pred in prediction.transpose(1, 2):
        score, cls = pred[:, 4:].max(1)
        valid = score > conf
        boxes, score, cls = xywh2xyxy(pred[valid, :4]), score[valid], cls[valid]
        order = score.argsort(descending=True)[:max_nms]
        keep = []
        while order.numel() and len(keep) < max_det:
            idx = order[0]
            keep.append(idx)
            rest = order[1:]
            overlap = box_iou(boxes[idx].view(1,4), boxes[rest]).view(-1)
            order = rest[~((overlap > iou) & (cls[rest] == cls[idx]))]
        if keep:
            k = torch.stack(keep)
            output.append(torch.cat((boxes[k], score[k,None], cls[k,None].to(boxes.dtype)), 1))
        else:
            output.append(pred.new_zeros((0,6)))
    return output


@torch.inference_mode()
def predict_image(model, source, size=640, conf=0.25, iou=0.7):
    with Image.open(source) as src:
        image = src.convert("RGB")
    tensor, ratios, pad = letterbox(image, size)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.eval()
    pred = model(tensor.unsqueeze(0).to(device=device, dtype=dtype))[0]
    det = non_max_suppression(pred, conf, iou)[0]
    if len(det):
        det[:, [0,2]] = (det[:, [0,2]]-pad[0])/ratios[0]
        det[:, [1,3]] = (det[:, [1,3]]-pad[1])/ratios[1]
        det[:, [0,2]] = det[:, [0,2]].clamp(0, image.width)
        det[:, [1,3]] = det[:, [1,3]].clamp(0, image.height)
    return image, det.cpu()


def save_annotated(image, detections, path, names=None):
    draw = ImageDraw.Draw(image)
    names = names or {}
    for row in detections.tolist():
        x1,y1,x2,y2,score,cls = row
        draw.rectangle((x1,y1,x2,y2), outline="red", width=2)
        draw.text((x1,max(0,y1-12)), f"{names.get(int(cls),int(cls))} {score:.2f}", fill="red")
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    image.save(path)
