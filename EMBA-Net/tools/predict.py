"""Predict with trained tensor-only weights; random-weight predictions are disallowed."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from emba_net import EMBANet
from emba_net.inference import predict_image,save_annotated

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default=str(Path(__file__).resolve().parents[1]/"configs/EMBA-Net.yaml"))
    p.add_argument("--weights",required=True)
    p.add_argument("--source",required=True)
    p.add_argument("--output",default="prediction.jpg")
    p.add_argument("--nc",type=int,default=1)
    p.add_argument("--scale",default="n",choices=list("nslmx"))
    p.add_argument("--imgsz",type=int,default=640)
    p.add_argument("--conf",type=float,default=0.25)
    p.add_argument("--iou",type=float,default=0.7)
    p.add_argument("--device",default="cpu")
    a=p.parse_args()
    m=EMBANet(a.config,nc=a.nc,scale=a.scale).load_weights(a.weights).to(a.device)
    image,det=predict_image(m,a.source,a.imgsz,a.conf,a.iou)
    save_annotated(image,det,a.output,m.names)
    path=Path(a.output).with_suffix(".json")
    path.write_text(json.dumps({"source":a.source,"columns":["x1","y1","x2","y2","confidence","class"],"detections":det.tolist()},indent=2),encoding="utf-8")
    print(f"Saved {a.output} and {path}")

if __name__=="__main__":main()
