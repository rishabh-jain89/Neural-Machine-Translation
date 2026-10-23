import argparse, json
from pathlib import Path
import torch

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--project-root',default=None)
    p.add_argument('--top-k',type=int,default=5)
    p.add_argument('--metric',choices=['dev_chrf','dev_bleu','val_cross_entropy'],default='dev_chrf')
    p.add_argument('--epochs',type=int,nargs='*',default=None)
    p.add_argument('--output-name',default=None)
    p.add_argument("--run-name", default="refined_transformer")
    return p.parse_args()

def main():
    a=parse_args(); root=Path(a.project_root).resolve() if a.project_root else Path(__file__).resolve().parent.parent
    ck=root/f"output/checkpoints/{a.run_name}_seed19"
    epdir=ck/'epoch_models'
    hist=ck/'history.json'

    if a.epochs:
        selected=sorted(set(a.epochs))
    else:
        history=json.load(open(hist,'r',encoding='utf-8'))
        reverse=(a.metric!='val_cross_entropy')
        ranked=sorted(history,key=lambda x:x[a.metric],reverse=reverse)
        selected=sorted(int(x['epoch']) for x in ranked[:a.top_k])

    paths=[epdir/f'epoch_{e:02d}.pt' for e in selected]
    missing=[p for p in paths if not p.exists()]

    if missing: 
        raise FileNotFoundError('\n'.join(map(str,missing)))
    
    print('Averaging epochs:',selected,flush=True)
    payloads=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
    states=[x['model_state_dict'] for x in payloads]; avg={}

    for k in states[0]:
        ts=[s[k] for s in states]; first=ts[0]
        if torch.is_floating_point(first):
            acc=torch.zeros_like(first,dtype=torch.float64)
            for t in ts: acc+=t.double()
            avg[k]=(acc/len(ts)).to(first.dtype)
        else:
            avg[k]=first.clone()

    outname=a.output_name or f'averaged_{len(selected)}_{a.metric}.pt'
    out=ck/outname
    torch.save({'model_state_dict':avg,'config':payloads[0]['config'],'averaged_epochs':selected,'selection_metric':a.metric,
                'source_metrics':[x.get('metrics',{}) for x in payloads]},out)
    print('Saved:',out,flush=True)

if __name__=='__main__': 
    main()
