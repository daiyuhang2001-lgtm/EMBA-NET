"""Count model parameters and deploy-time convolution/GEMM FLOPs."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from emba_net import EMBANet
from emba_net.complexity import profile_model

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'configs/EMBA-Net.yaml'))
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--output')
    a=p.parse_args();torch.set_num_threads(2)
    result=profile_model(EMBANet(a.config,nc=1,scale='n').eval(),a.imgsz)
    report=json.dumps(result,indent=2)
    if a.output:Path(a.output).write_text(report,encoding='utf8')
    print(report)

if __name__=='__main__':main()
