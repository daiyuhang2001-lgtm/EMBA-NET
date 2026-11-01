"""Ready-to-run entry points for the bundled EMBA-Net code and supplied dataset."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse,json,subprocess,sys

ROOT=Path(__file__).resolve().parent


def absolute(path):
    p=Path(path)
    return p if p.is_absolute() else ROOT/p


def settings(data_root=None):
    import yaml
    config=yaml.safe_load((ROOT/'configs/train.yaml').read_text(encoding='utf-8-sig'))
    ds=yaml.safe_load(absolute(config['data']).read_text(encoding='utf-8-sig'))
    base=Path(data_root).resolve() if data_root else absolute(ds.get('path','data'))
    paths={}
    for split in ['train','val','test']:
        paths[split]=(base/ds[split],base/ds.get(split+'_labels','labels/'+split))
        if not all(p.is_dir() for p in paths[split]):
            raise FileNotFoundError(f'Dataset directory missing: {paths[split]}')
    return config,ds,paths


def choose_device(request):
    import torch
    if request!='auto':return request
    if torch.cuda.is_available():
        try:
            torch.zeros(1,device='cuda:0').add_(1)
            return 'cuda:0'
        except RuntimeError as error:
            print(f'CUDA is unavailable in this environment; selecting CPU. Detail: {error}')
    return 'cpu'


def output_dir(kind,requested=None):
    if requested:return absolute(requested)
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    path=ROOT/'runs'/f'{kind}_{stamp}'
    n=1
    while path.exists():path=ROOT/'runs'/f'{kind}_{stamp}_{n}';n+=1
    return path


def check(config,paths,device):
    import torch
    from emba_net import EMBANet
    from emba_net.data import DetectionDataset
    report={'python':sys.executable,'torch':torch.__version__,'device':device,'model_config':'configs/EMBA-Net.yaml','dataset':{}}
    if device.startswith('cuda'):report['gpu']=torch.cuda.get_device_name(0)
    for split,(images,labels) in paths.items():
        ds=DetectionDataset(images,labels,config['imgsz'],config['nc'])
        tensor,rows=ds[0]
        report['dataset'][split]={'images':len(ds),'sample_image_shape':list(tensor.shape),'sample_boxes':len(rows)}
    model=EMBANet(absolute(config['model']),nc=config['nc'],scale=config['scale']).to(device).eval()
    with torch.inference_mode():prediction,_=model(torch.zeros(1,3,config['imgsz'],config['imgsz'],device=device))
    report.update({'backbone':'PVFeaturePerceptionNet','parameters':sum(p.numel() for p in model.parameters()),
                   'prediction_shape':list(prediction.shape),'check_passed':True})
    from emba_net.complexity import profile_model
    report['complexity']=profile_model(model,config['imgsz'])
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return report


def smoke(config,paths,device,args):
    import torch
    from torch.utils.data import DataLoader
    from emba_net import EMBANet
    from emba_net.data import DetectionDataset,collate_batch
    from emba_net.loss import DetectionLoss
    torch.manual_seed(config['seed'])
    size=args.imgsz or config['imgsz'];batch=args.batch or 2
    loaders={split:DataLoader(DetectionDataset(*paths[split],size,config['nc']),batch_size=batch,shuffle=False,
                              num_workers=0,collate_fn=collate_batch) for split in ['train','val']}
    model=EMBANet(absolute(config['model']),nc=config['nc'],scale=config['scale']).to(device)
    criterion=DetectionLoss(model)
    if config['optimizer']=='AdamW':
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    else:
        optimizer=torch.optim.SGD(model.parameters(),lr=config['lr'],momentum=config['momentum'],weight_decay=config['weight_decay'])
    out=output_dir('smoke',args.output);out.mkdir(parents=True,exist_ok=False)
    results={}
    for phase in ['train','val']:
        model.train(phase=='train')
        item=next(iter(loaders[phase]));item={key:value.to(device) for key,value in item.items()}
        with torch.set_grad_enabled(phase=='train'):
            loss,parts=criterion(model(item['img']),item)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite smoke-test loss.')
            if phase=='train':
                optimizer.zero_grad(set_to_none=True);loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise RuntimeError('Non-finite gradients.')
                torch.nn.utils.clip_grad_norm_(model.parameters(),10)
                optimizer.step()
        results[phase]={'batch':len(item['img']),'loss':float(loss.detach()),'components':parts.tolist()}
    model.save_weights(out/'smoke_only.pt',smoke_only=True)
    result={'smoke_passed':True,'device':device,'imgsz':size,'model':'EMBA-Net.yaml','results':results,
            'output':str(out),'note':'One training batch and one validation batch; not a trained research model.'}
    (out/'smoke_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


def train(config,paths,device,args):
    out=output_dir('train',args.output)
    command=[sys.executable,str(ROOT/'tools/train.py'),'--config',str(absolute(config['model'])),
             '--train-images',str(paths['train'][0]),'--train-labels',str(paths['train'][1]),
             '--val-images',str(paths['val'][0]),'--val-labels',str(paths['val'][1]),
             '--output',str(out),'--device',device]
    values={'nc':config['nc'],'scale':config['scale'],'epochs':args.epochs or config['epochs'],
            'batch':args.batch or config['batch'],'imgsz':args.imgsz or config['imgsz'],
            'optimizer':config['optimizer'],'lr':config['lr'],'weight-decay':config['weight_decay'],
            'momentum':config['momentum'],'seed':config['seed'],'workers':config['workers']}
    for key,value in values.items():command.extend(['--'+key,str(value)])
    if args.weights:command.extend(['--weights',str(absolute(args.weights))])
    print(f'Training output: {out}',flush=True)
    print(f'epochs={values["epochs"]}, batch={values["batch"]}, imgsz={values["imgsz"]}, device={device}',flush=True)
    return subprocess.call(command,cwd=ROOT)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',nargs='?',choices=['check','smoke','train','predict'],
                   help='Optional: without an action, check the project and show a menu.')
    p.add_argument('--device',default=None,help='auto, cpu or cuda:0')
    p.add_argument('--epochs',type=int)
    p.add_argument('--batch',type=int)
    p.add_argument('--imgsz',type=int)
    p.add_argument('--output')
    p.add_argument('--weights')
    p.add_argument('--source')
    p.add_argument('--data-root',help='Dataset root containing images/ and labels/; does not alter the split.')
    args=p.parse_args()
    try:
        import torch,yaml,numpy,PIL,einops
    except ModuleNotFoundError as error:
        p.exit(1,f'Missing dependency: {error.name}. Run this interpreter with: -m pip install -r requirements.txt\n')
    torch.set_num_threads(2)
    config,_,paths=settings(args.data_root)
    device=choose_device(args.device or config['device'])
    if args.action is None or args.action=='check':
        check(config,paths,device)
        if args.action=='check':return
        print('\n检查通过。请选择下一步：',flush=True)
        print('  1 = 试跑一个训练批次',flush=True)
        print(f'  2 = 开始训练（{args.epochs or config["epochs"]}轮，device={device}）',flush=True)
        print('  0 = 退出（直接回车也退出）',flush=True)
        try:
            choice=input('请输入 0 / 1 / 2：').strip()
        except (EOFError,KeyboardInterrupt):
            print('\n检查完成。');return
        if choice in ('','0'):return
        if choice not in ('1','2'):
            print('未选择有效操作，已退出。');return
        args.action='smoke' if choice=='1' else 'train'
    if args.action=='smoke':smoke(config,paths,device,args)
    elif args.action=='train':raise SystemExit(train(config,paths,device,args))
    else:
        if not args.weights or not args.source:p.error('predict requires --weights and --source')
        command=[sys.executable,str(ROOT/'tools/predict.py'),'--config',str(absolute(config['model'])),
                 '--weights',str(absolute(args.weights)),'--source',str(absolute(args.source)),
                 '--output',str(absolute(args.output or 'prediction.jpg')),'--nc',str(config['nc']),
                 '--scale',config['scale'],'--device',device,'--imgsz',str(args.imgsz or config['imgsz'])]
        raise SystemExit(subprocess.call(command,cwd=ROOT))

if __name__=='__main__':main()
