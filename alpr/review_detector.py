"""Detector-only validation, annotated images and reproducible bbox metrics (no OCR)."""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image, ImageDraw
from .common import read_meta, read_image, numbers, iou, TYPES
from .models import load_model
from .infer import detect_plates


def score_frame(truth, predicted, cutoff=.5):
    """Greedy confidence-ordered one-to-one matching, independent of type."""
    used=set()
    matches=[]
    for prediction_index in sorted(range(len(predicted)),key=lambda i:-predicted[i]['score']):
        p=predicted[prediction_index]
        candidates=[(iou(np.asarray(p['box']),np.asarray(g['box'])),i)
                    for i,g in enumerate(truth) if i not in used]
        overlap,index=max(candidates,default=(0,-1))
        if overlap>=cutoff:
            used.add(index)
            matches.append((index,prediction_index))
    return matches


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True)
    parser.add_argument('--split',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--part',choices=['train','val','test'],default='val')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--size',type=int)
    parser.add_argument('--threshold',type=float,default=.35)
    parser.add_argument('--refine',action='store_true')
    args=parser.parse_args()
    out=Path(args.out)
    if out.exists():
        parser.error('Use a new output directory; reports are not overwritten')
    data=Path(args.data)
    names=json.loads(Path(args.split).read_text(encoding='utf-8-sig'))[args.part]
    truth=defaultdict(list)
    for row in read_meta(data):
        if row['image'] in names:
            x,y,w,h=numbers(row['bbox'],4)
            truth[row['image']].append({'box':[float(x),float(y),float(x+w),float(y+h)],
                                        'plate_type':row['plate_type']})
    torch.set_num_threads(4)
    model,checkpoint=load_model(args.model,'detector',args.device)
    if checkpoint.get('generic_plate_pretraining'):
        parser.error('Fine-tune generic geometry weights on plate types first')
    size=args.size or checkpoint.get('size',512)
    out.mkdir(parents=True)
    (out/'images').mkdir()
    counts={k:{'gt':0,'pred':0,'tp':0} for k in TYPES}
    records=[];details={};times=[];confusion=defaultdict(int)
    detect_plates(model,np.full((size,size,3),114,np.uint8),args.device,size,args.threshold,
                  args.refine,min(.15,args.threshold))
    for index,name in enumerate(names):
        if args.device.startswith('cuda'):torch.cuda.synchronize()
        start=time.perf_counter()
        im=read_image(data/name)
        predictions=detect_plates(model,im,args.device,size,args.threshold,args.refine,min(.15,args.threshold))
        if args.device.startswith('cuda'):torch.cuda.synchronize()
        times.append((time.perf_counter()-start)*1000)
        matches=score_frame(truth[name],predictions)
        correct=0
        for gt in truth[name]:counts[gt['plate_type']]['gt']+=1
        for p in predictions:counts[p['plate_type']]['pred']+=1
        for gi,pi in matches:
            gt,p=truth[name][gi],predictions[pi]
            confusion[gt['plate_type']+' -> '+p['plate_type']]+=1
            if gt['plate_type']==p['plate_type']:
                counts[gt['plate_type']]['tp']+=1
                correct+=1
        records.append({'image':name,'gt':len(truth[name]),'pred':len(predictions),
                        'localized':len(matches),'correct_type':correct,'file':f'images/{index:03d}.jpg'})
        details[name]=[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in p.items()} for p in predictions]
        canvas=Image.fromarray(im);scale=min(1,1100/max(canvas.size))
        canvas=canvas.resize((round(canvas.width*scale),round(canvas.height*scale)))
        draw=ImageDraw.Draw(canvas)
        for gt in truth[name]:
            b=[v*scale for v in gt['box']];draw.rectangle(b,outline='#00ff70',width=3)
            draw.text((b[0],max(0,b[1]-12)),'GT '+gt['plate_type'],fill='#00ff70',stroke_width=1,stroke_fill='black')
        for p in predictions:
            b=[float(v)*scale for v in p['box']];draw.rectangle(b,outline='#ff4040',width=2)
            draw.text((b[0],min(canvas.height-12,b[3]+2)),p['plate_type']+' %.2f'%p['score'],fill='#ff4040',stroke_width=1,stroke_fill='black')
        canvas.save(out/records[-1]['file'],quality=92)
        if (index+1)%30==0:print(f'Checked {index+1}/{len(names)}',flush=True)
    total=sum(c['gt'] for c in counts.values());num_pred=sum(c['pred'] for c in counts.values())
    tp=sum(c['tp'] for c in counts.values());localized=sum(r['localized'] for r in records)
    for c in counts.values():
        c.update(fp=c['pred']-c['tp'],fn=c['gt']-c['tp'],
                 precision=c['tp']/c['pred'] if c['pred'] else None,
                 recall=c['tp']/c['gt'] if c['gt'] else None)
    report={'config':vars(args),'checkpoint_epoch':checkpoint.get('epoch'),'size':size,
            'metric':'detector-only, IoU >= 0.5; confidence-ordered one-to-one matching; no vehicle filtering',
            'images':len(names),'gt':total,'predictions':num_pred,'localized':localized,'correct_type':tp,
            'precision_with_type':tp/max(num_pred,1),'recall_with_type':tp/max(total,1),
            'per_type':counts,'confusion':dict(confusion),'frames':records,
            'timing':{'mean_ms':float(np.mean(times)),'p95_ms':float(np.percentile(times,95)),
                      'hardware':torch.cuda.get_device_name() if args.device.startswith('cuda') else 'CPU',
                      'includes':'image decoding and detector/refinement; excludes model startup, warmup, OCR and review rendering'},
            'limitations':'Validation tuning is not independent test performance. OCR and unlabelled objects not evaluated.'}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'predictions.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf-8')
    page=['<meta charset="utf-8"><title>Detector review</title><style>body{font:16px sans-serif;background:#222;color:white}img{max-width:100%}</style>',
          '<h1>Проверка рамок детектора</h1><p>Зелёный — разметка; красный — предсказание. IoU ≥ 0,5; OCR не проверялся.</p>']
    for r in records:
        page.append(f'<h3>{html.escape(r["image"])}</h3><p>Размечено {r["gt"]}; найдено {r["localized"]}; верен тип {r["correct_type"]}</p><img loading="lazy" src="{r["file"]}">')
    (out/'review.html').write_text('\n'.join(page),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['images','gt','predictions','localized','correct_type','timing']}),flush=True)


if __name__=='__main__':
    main()
