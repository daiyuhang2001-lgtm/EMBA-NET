"""Convert a compatible training checkpoint to the fused deployment graph."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from emba_net import EMBANet

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'configs/EMBA-Net.yaml'))
    p.add_argument('--weights',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    if Path(a.weights).resolve()==Path(a.output).resolve():
        p.error('Choose a separate output path to preserve the training checkpoint.')
    model=EMBANet(a.config).load_weights(a.weights).eval().switch_to_deploy()
    model.save_weights(a.output)
    print(f'Saved fused checkpoint: {a.output}')

if __name__=='__main__':main()
