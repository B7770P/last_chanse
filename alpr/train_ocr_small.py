"""Fine-tune existing OCR on cached real crops and contour-aligned variants."""
import argparse,json,random
from pathlib import Path
import cv2,numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from .common import read_meta,read_image,numbers,rectify,ALPHABET
from .alignment import alignment_quads
from .data import ocr_collate
from .models import load_model,save_model
from .train import ctc_text

class Crops(Dataset):
    def __init__(self,rows,root,augment):
        self.samples=[];self.augment=augment
        for r in rows:
            im=read_image(Path(root)/r['image']);q=numbers(r['quad'],8)
            variants=[rectify(im,q,r['plate_type'])]
            if augment:
                x,y,w,h=numbers(r['bbox'],4);box=[x,y,x+w,y+h]
                quads=[np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])]+alignment_quads(im,box,max_candidates=2)
                for quad in quads:
                    try:variants.append(rectify(im,quad,r['plate_type']))
                    except ValueError:pass
            self.samples.append((variants,r['plate_num']))
    def __len__(self):return len(self.samples) * (1 if self.augment else 2)
    def __getitem__(self,i):
        degraded = not self.augment and i >= len(self.samples)
        variants,text=self.samples[i % len(self.samples)]
        crop=random.choice(variants).copy() if self.augment else variants[0]
        if self.augment:
            crop=np.clip(crop.astype(np.float32)*random.uniform(.8,1.2)+random.uniform(-10,10),0,255).astype(np.uint8)
            if random.random()<.25:crop=cv2.GaussianBlur(crop,(3,3),.5)
        if (self.augment and random.random()<.65) or (not self.augment and degraded):
            width=random.choice([64,80,96,128,160]) if self.augment else 96
            height=max(12,round(width*.15))
            crop=cv2.resize(crop,(width,height),interpolation=cv2.INTER_AREA)
            crop=cv2.resize(crop,(320,48),interpolation=cv2.INTER_CUBIC)
        return torch.from_numpy(crop.copy()).float()[None]/127.5-1,torch.tensor([ALPHABET.index(c)+1 for c in text]),text

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data','split','resume','out']:p.add_argument('--'+key,required=True)
    p.add_argument('--epochs',type=int,default=50);p.add_argument('--lr',type=float,default=.0003)
    a=p.parse_args();out=Path(a.out);assert not out.exists();out.mkdir(parents=True)
    random.seed(20261024);np.random.seed(20261024);torch.manual_seed(20261024);torch.set_num_threads(4)
    split=json.loads(Path(a.split).read_text(encoding='utf-8'));assert not set(split['train'])&set(split['val']+split['test'])
    rows=[r for r in read_meta(a.data) if r['plate_type']!='other' and '#' not in r['plate_num']]
    sets={part:Crops([r for r in rows if r['image'] in split[part]],a.data,part=='train') for part in ['train','val']}
    print(json.dumps({'cached_crops':{k:len(v) for k,v in sets.items()}}),flush=True)
    loaders={k:DataLoader(v,batch_size=32,shuffle=k=='train',collate_fn=ocr_collate) for k,v in sets.items()}
    model,_=load_model(a.resume,'ocr','cuda');opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.0001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs);ctc=torch.nn.CTCLoss(blank=0)
    best=(float('inf'),float('inf'));history=[]
    for epoch in range(a.epochs+1):
        metrics={}
        for part in (['val'] if epoch==0 else ['train','val']):
            model.train(part=='train');total=seen=exact=0
            with torch.set_grad_enabled(part=='train'):
                for images,targets,lengths,texts in loaders[part]:
                    logits=model(images.cuda());loss=ctc(logits.log_softmax(-1).transpose(0,1),targets.cuda(),torch.full((len(images),),logits.shape[1],dtype=torch.long),lengths)
                    assert torch.isfinite(loss)
                    if part=='train':
                        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
                    seen+=len(images);total+=float(loss.detach())*len(images);exact+=sum(x==y for x,y in zip(ctc_text(logits.detach()),texts))
            metrics[part+'_loss']=total/seen;metrics[part+'_exact']=exact/seen
        if epoch:schedule.step()
        record={'epoch':epoch,**metrics};history.append(record);print(json.dumps(record),flush=True)
        info={'epoch':epoch,'metrics':metrics,'config':vars(a),'checkpoint_selection':'val_exact then val_loss','seed':20261024,'crop_version':'small-resolution-v1'}
        selection=(-metrics['val_exact'],metrics['val_loss'])
        if selection<best:best=selection;save_model(out/'best.pt',model,'ocr',**info)
        save_model(out/'last.pt',model,'ocr',**info)
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')

if __name__=='__main__':main()
