"""Evaluate OCR on ground-truth crops and complete YOLO+OCR on a fixed split."""
import argparse,json,html,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from .common import read_meta,read_image,numbers,rectify,TYPES
from .infer import Pipeline,decode_text
from .train import ctc_text
from .review_detector import score_frame

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','split','detector','ocr','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--part',choices=['val','test'],default='val')
    p.add_argument('--align',action='store_true')
    p.add_argument('--white-detector')
    p.add_argument('--type1-ocr')
    p.add_argument('--multiscale-type1',action='store_true')
    p.add_argument('--type1-char-threshold',type=float)
    p.add_argument('--device',default='cuda');a=p.parse_args()
    out=Path(a.out);assert not out.exists();out.mkdir(parents=True)
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)
    pipe=Pipeline(a.detector,a.ocr,a.device,keep_nonvehicle=True,detector_backend='yolo',align=a.align,white_detector=a.white_detector,type1_ocr=a.type1_ocr,type1_char_threshold=a.type1_char_threshold,multiscale_type1=a.multiscale_type1)
    names=json.loads(Path(a.split).read_text(encoding='utf-8'))[a.part]
    rows=defaultdict(list)
    for r in read_meta(a.data):
        if r['image'] in names:rows[r['image']].append(r)
    stats={k:dict(readable=0,oracle_exact=0,pipeline_exact=0) for k in TYPES[:3]}
    records=[];predictions={};times=[]
    pipe(np.full((640,640,3),114,np.uint8))
    for name in names:
        start=time.perf_counter()
        im=read_image(Path(a.data)/name)
        pred=pipe(im)
        if a.device.startswith('cuda'):torch.cuda.synchronize()
        times.append((time.perf_counter()-start)*1000)
        gt=[]
        for r in rows[name]:
            x,y,w,h=numbers(r['bbox'],4)
            gt.append({**r,'box':[float(x),float(y),float(x+w),float(y+h)]})
        matches=dict(score_frame(gt,pred))
        predictions[name]=[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in r.items()} for r in pred]
        for i,r in enumerate(gt):
            kind=r['plate_type'];text=r['plate_num']
            if kind=='other' or '#' in text:continue
            crop=rectify(im,numbers(r['quad'],8),kind)
            logits=pipe.recognize_crops(torch.from_numpy(crop.copy()).float()[None,None].to(a.device)/127.5-1,[kind])
            raw=ctc_text(logits)[0]
            recognized=pred[matches[i]] if i in matches else None
            exact=bool(recognized and recognized['plate_type']==kind and recognized['plate_num']==text)
            stats[kind]['readable']+=1;stats[kind]['oracle_exact']+=int(raw==text);stats[kind]['pipeline_exact']+=int(exact)
            records.append({'image':name,'type':kind,'truth':text,'ocr_on_manual_crop':raw,'pipeline':recognized['plate_num'] if recognized else '', 'exact':exact})
    report={'part':a.part,'images':len(names),'counts':stats,'readable_total':sum(s['readable'] for s in stats.values()),'oracle_exact':sum(s['oracle_exact'] for s in stats.values()),'pipeline_exact':sum(s['pipeline_exact'] for s in stats.values()),'threshold':.65,'multiscale_type1':a.multiscale_type1,'type1_threshold':a.type1_char_threshold,'det_threshold':.35,'matching_iou':.5,'vehicle_filter':False,'records':records}
    report.update(alignment=a.align,predictions=sum(len(v) for v in predictions.values()),mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95)),timing_scope='Image read + YOLO + OCR, warmup excluded; no CSV writing',hardware=torch.cuda.get_device_name() if a.device.startswith('cuda') else 'CPU')
    for filename,data in [('report.json',report),('predictions.json',predictions)]:
        (out/filename).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in ['image','type','truth','ocr_on_manual_crop','pipeline','exact'])+'</tr>' for r in records)
    (out/'review.html').write_text('<meta charset="utf-8"><h1>OCR validation</h1><p>Manual crop vs YOLO + OCR. Exact full-string match.</p><table border="1"><tr><th>Image</th><th>Type</th><th>Truth</th><th>Manual crop</th><th>YOLO + OCR</th><th>Exact</th></tr>'+table+'</table>',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)

if __name__=='__main__':main()


