"""Review a locally trained four-class YOLO detector without OCR or network access."""
import argparse
from collections import defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image, ImageDraw
from .common import read_meta, read_image, numbers, TYPES
from .review_detector import score_frame


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--split',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--part',choices=['train','val','test'],default='val')
    p.add_argument('--device',default='0')
    p.add_argument('--size',type=int,default=640)
    p.add_argument('--threshold',type=float,default=.35)
    a=p.parse_args()
    out=Path(a.out).resolve()
    if out.exists():p.error('Choose a new output folder')
    modelpath=Path(a.model).resolve()
    if not modelpath.is_file():p.error('Local weights file is required; no automatic download')
    runtime=out.parent/'yolo_runtime';(runtime/'Ultralytics').mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(runtime)
    os.environ['YOLO_OFFLINE']='true'
    os.environ['YOLO_AUTOINSTALL']='false'
    os.environ['WANDB_DISABLED']='true'
    from ultralytics import YOLO,settings
    for key in ['sync','clearml','comet','dvc','mlflow','raytune','tensorboard','wandb']:
        if key in settings:settings.update({key:False})
    model=YOLO(str(modelpath))
    if [model.names[i] for i in range(len(model.names))]!=list(TYPES):
        p.error('Weights must have type1/type1a/type1b/other classes in that order')
    torch.set_num_threads(4)
    data=Path(a.data)
    names=json.loads(Path(a.split).read_text(encoding='utf-8-sig'))[a.part]
    truth=defaultdict(list)
    for row in read_meta(data):
        if row['image'] in names:
            x,y,w,h=numbers(row['bbox'],4)
            truth[row['image']].append({'box':[float(x),float(y),float(x+w),float(y+h)],'plate_type':row['plate_type']})
    def predict(im):
        result=model.predict(Image.fromarray(im),imgsz=a.size,conf=a.threshold,iou=.4,
                             agnostic_nms=True,device=a.device,verbose=False,max_det=60)[0]
        return [{'box':box.tolist(),'plate_type':TYPES[int(kind)],'score':float(conf)}
                for box,kind,conf in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.cls.cpu().numpy(),result.boxes.conf.cpu().numpy())]
    predict(np.full((a.size,a.size,3),114,np.uint8))
    out.mkdir(parents=True);(out/'images').mkdir()
    counts={k:{'gt':0,'pred':0,'tp':0} for k in TYPES}
    frames=[];details={};elapsed=[];confusion=defaultdict(int)
    for index,name in enumerate(names):
        if a.device!='cpu':torch.cuda.synchronize()
        start=time.perf_counter();im=read_image(data/name);pred=predict(im)
        if a.device!='cpu':torch.cuda.synchronize()
        elapsed.append((time.perf_counter()-start)*1000)
        matches=score_frame(truth[name],pred)
        correct=0
        for gt in truth[name]:counts[gt['plate_type']]['gt']+=1
        for item in pred:counts[item['plate_type']]['pred']+=1
        for gi,pi in matches:
            gt,item=truth[name][gi],pred[pi]
            confusion[gt['plate_type']+' -> '+item['plate_type']]+=1
            if gt['plate_type']==item['plate_type']:
                counts[gt['plate_type']]['tp']+=1;correct+=1
        frame={'image':name,'gt':len(truth[name]),'pred':len(pred),'localized':len(matches),
               'correct_type':correct,'file':f'images/{index:03d}.jpg'}
        frames.append(frame);details[name]=pred
        canvas=Image.fromarray(im);canvas.thumbnail((1100,1100))
        scales=[canvas.width/im.shape[1],canvas.height/im.shape[0]]*2
        draw=ImageDraw.Draw(canvas)
        for gt in truth[name]:
            b=[v*s for v,s in zip(gt['box'],scales)];draw.rectangle(b,outline='#00ff70',width=3)
            draw.text((b[0],max(0,b[1]-12)),'GT '+gt['plate_type'],fill='#00ff70',stroke_width=1,stroke_fill='black')
        for item in pred:
            b=[v*s for v,s in zip(item['box'],scales)];draw.rectangle(b,outline='#ff4040',width=2)
            draw.text((b[0],min(canvas.height-12,b[3]+2)),item['plate_type']+' %.2f'%item['score'],fill='#ff4040',stroke_width=1,stroke_fill='black')
        canvas.save(out/frame['file'],quality=92)
        if (index+1)%30==0:print(f'Checked {index+1}/{len(names)}',flush=True)
    total=sum(c['gt'] for c in counts.values());num_pred=sum(c['pred'] for c in counts.values());tp=sum(c['tp'] for c in counts.values())
    for c in counts.values():
        c.update(fp=c['pred']-c['tp'],fn=c['gt']-c['tp'],precision=c['tp']/c['pred'] if c['pred'] else None,
                 recall=c['tp']/c['gt'] if c['gt'] else None)
    report={'config':vars(a),'model_sha256':hashlib.sha256(modelpath.read_bytes()).hexdigest(),
            'metric':'detector-only, IoU >= 0.5; confidence-ordered one-to-one matching; class-agnostic NMS IoU 0.4; no vehicle filtering',
            'images':len(names),'gt':total,'predictions':num_pred,'localized':sum(r['localized'] for r in frames),
            'correct_type':tp,'precision_with_type':tp/max(num_pred,1),'recall_with_type':tp/max(total,1),
            'per_type':counts,'confusion':dict(confusion),'frames':frames,
            'timing':{'mean_ms':float(np.mean(elapsed)),'p95_ms':float(np.percentile(elapsed,95)),
                      'hardware':torch.cuda.get_device_name() if a.device!='cpu' else 'CPU',
                      'includes':'image decode, preprocessing, YOLO and postprocessing; excludes warmup, OCR and report rendering'},
            'limitations':'Validation only; no independent type1 evaluation. Axis-aligned boxes, no learned corner/vehicle heads. No OCR.'}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'predictions.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf-8')
    page=['<meta charset="utf-8"><title>YOLO review</title><style>body{font:16px sans-serif;background:#222;color:#eee}img{max-width:100%}</style><h1>YOLO11n: проверка детектора</h1><p>Зелёные рамки — разметка; красные — предсказания. OCR не проверялся.</p>']
    for r in frames:page.append(f'<h3>{html.escape(r["image"])}</h3><p>Размечено {r["gt"]}; найдено {r["localized"]}; верен тип {r["correct_type"]}</p><img loading="lazy" src="{r["file"]}">')
    (out/'review.html').write_text('\n'.join(page),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['images','gt','predictions','localized','correct_type','timing']}),flush=True)


if __name__=='__main__':main()
