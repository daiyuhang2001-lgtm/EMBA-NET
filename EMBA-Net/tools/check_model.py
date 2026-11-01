"""Build the selected graph and inspect forward outputs without downloading weights."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from emba_net import EMBANet
from emba_net.complexity import profile_model

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default=str(Path(__file__).resolve().parents[1]/"configs/EMBA-Net.yaml"))
    p.add_argument("--nc",type=int,default=1)
    p.add_argument("--scale",default="n",choices=list("nslmx"))
    p.add_argument("--imgsz",type=int,default=640)
    p.add_argument("--device",default="cpu")
    a=p.parse_args()
    torch.set_num_threads(2)
    m=EMBANet(a.config,nc=a.nc,scale=a.scale).to(a.device).eval()
    with torch.inference_mode():
        y,raw=m(torch.zeros(1,3,a.imgsz,a.imgsz,device=a.device))
    result=m.summary()
    result['complexity']=profile_model(m,a.imgsz)
    result.update({"prediction_shape":list(y.shape),"raw_shapes":[list(t.shape) for t in raw],"trained_weights_loaded":False})
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
