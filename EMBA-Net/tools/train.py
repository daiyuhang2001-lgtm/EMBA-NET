"""Minimal new training example; not the historical paper training pipeline."""
import argparse,csv,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from emba_net import EMBANet
from emba_net.data import DetectionDataset,collate_batch
from emba_net.loss import DetectionLoss

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default=str(Path(__file__).resolve().parents[1]/"configs/EMBA-Net.yaml"))
    for name in ("train-images","train-labels","val-images","val-labels"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--output",default="runs/experiment")
    p.add_argument("--weights")
    p.add_argument("--nc",type=int,default=1)
    p.add_argument("--scale",default="n",choices=list("nslmx"))
    p.add_argument("--epochs",type=int,default=300)
    p.add_argument("--batch",type=int,default=4)
    p.add_argument("--imgsz",type=int,default=640)
    p.add_argument("--optimizer",choices=["SGD","AdamW"],default="SGD")
    p.add_argument("--lr",type=float,default=0.01)
    p.add_argument("--weight-decay",type=float,default=0.0005)
    p.add_argument("--momentum",type=float,default=0.937)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--workers",type=int,default=0)
    p.add_argument("--device",default="cpu")
    a=p.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    (out/"arguments.json").write_text(json.dumps(vars(a),indent=2),encoding="utf-8")
    train=DetectionDataset(a.train_images,a.train_labels,a.imgsz,a.nc)
    val=DetectionDataset(a.val_images,a.val_labels,a.imgsz,a.nc)
    if {x.resolve() for x in train.images} & {x.resolve() for x in val.images}:
        raise ValueError("Training and validation contain the same image paths.")
    # Path separation is only a basic guard. Source grouping and duplicate-content
    # checks must be performed on the user's experimental split before training.
    loaders=[DataLoader(ds,batch_size=a.batch,shuffle=(i==0),num_workers=a.workers,collate_fn=collate_batch)
             for i,ds in enumerate((train,val))]
    model=EMBANet(a.config,nc=a.nc,scale=a.scale)
    if a.weights:model.load_weights(a.weights)
    model.to(a.device)
    criterion=DetectionLoss(model)
    if a.optimizer=="SGD":
        optimizer=torch.optim.SGD(model.parameters(),lr=a.lr,momentum=a.momentum,weight_decay=a.weight_decay)
    else:
        optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    best=float("inf")
    with (out/"loss.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f);writer.writerow(["epoch","train_loss_per_image","val_loss_per_image"])
        for epoch in range(a.epochs):
            totals=[]
            for phase,loader in enumerate(loaders):
                model.train(phase==0)
                total,count=0.0,0
                with torch.set_grad_enabled(phase==0):
                    for step,batch in enumerate(loader,1):
                        batch={k:v.to(a.device) for k,v in batch.items()}
                        if phase==0:optimizer.zero_grad(set_to_none=True)
                        loss,_=criterion(model(batch["img"]),batch)
                        if not torch.isfinite(loss):raise RuntimeError("Non-finite loss; experiment stopped.")
                        if phase==0:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(),10)
                            optimizer.step()
                        total+=float(loss.detach());count+=len(batch["img"])
                        if step==1 or step%25==0 or step==len(loader):
                            print(f'epoch {epoch+1}/{a.epochs} {"train" if phase==0 else "val"} '
                                  f'batch {step}/{len(loader)} loss_per_image={float(loss.detach())/len(batch["img"]):.6f}',flush=True)
                totals.append(total/count)
            writer.writerow([epoch+1,*totals]);f.flush()
            model.save_weights(out/"last.pt",epoch=epoch+1,validation_loss=totals[1])
            if totals[1]<best:
                best=totals[1];model.save_weights(out/"best.pt",epoch=epoch+1,validation_loss=best)
            print(f"epoch {epoch+1}/{a.epochs}: train={totals[0]:.6f}, val={totals[1]:.6f}",flush=True)
    print("Best checkpoint selected by validation loss; this script does not report mAP.")

if __name__=="__main__":main()
