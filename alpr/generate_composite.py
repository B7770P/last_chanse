"""Reproducible plate replacement on training photographs; no held-out backgrounds."""
import argparse
from collections import Counter, defaultdict
import hashlib
from importlib.metadata import version as package_version, PackageNotFoundError
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .common import read_meta, read_image, write_image, write_meta, write_labels, numbers, valid_number
from .generate import random_number, FONT


def opencv_package():
    for name in ('opencv-python-headless', 'opencv-python'):
        try:
            value=package_version(name)
            if value.startswith(cv2.__version__+'.') or value==cv2.__version__:
                return name,value
        except PackageNotFoundError:
            pass
    raise ValueError('Cannot identify the installed OpenCV package for reproducibility')


def render(text, kind, rng):
    """Approximate plate typography, with Cyrillic glyph shapes and Latin labels."""
    if not valid_number(text, kind) or kind == 'other':
        raise ValueError('Expected a target plate with a valid transcription')
    square = kind == 'type1a'
    w, h = (290, 170) if square else (520, 112)
    s = 2
    color = (239, 193, 48) if kind == 'type1b' else (231, 232, 225)
    im = Image.new('RGB', (w*s, h*s), color)
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(str(FONT), 180)

    def glyph(text, x, y, gw, gh):
        # Y labels represent Cyrillic У, not a Latin Y painted on the plate.
        text = text.replace('Y', 'У')
        box = font.getbbox(text)
        mask = Image.new('L', (box[2]-box[0], box[3]-box[1]), 0)
        ImageDraw.Draw(mask).text((-box[0], -box[1]), text, font=font, fill=255)
        mask = mask.resize((int(gw*s), int(gh*s)), Image.Resampling.LANCZOS)
        im.paste((24, 25, 23), (int(x*s), int(y*s)), mask)

    def line(coords, width=1):
        d.line(tuple(int(v*s) for v in coords), fill=(34,35,32), width=width*s)

    def flag(x,y):
        for j,c in enumerate([(245,245,240),(35,65,154),(172,35,38)]):
            d.rectangle((x*s,(y+4*j)*s,(x+33)*s,(y+4*j+3)*s),fill=c)

    d.rounded_rectangle((3*s,3*s,(w-4)*s,(h-4)*s),radius=6*s,outline=(40,41,38),width=2*s)
    if square:
        for i,c in enumerate(text[:4]):
            glyph(c, 18+i*65, 18 if i == 0 else 12, 45, 59 if i == 0 else 65)
        glyph(text[4],18,96,42,53)
        glyph(text[5],78,96,42,53)
        line((137,87,137,166))
        region=text[6:]
        step=40 if len(region)==3 else 54
        for i,c in enumerate(region):
            glyph(c,150+i*step,94,30 if len(region)==3 else 36,49)
        glyph('RUS',151,151,45,10)
        flag(232,151)
    else:
        prefix=5 if kind=='type1b' else 6
        for i,c in enumerate(text[:prefix]):
            letter=(i in ((0,1) if kind=='type1b' else (0,4,5)))
            glyph(c,18+i*(72 if prefix==5 else 60),26 if letter else 16,
                  48 if prefix==5 else 42,57 if letter else 67)
        line((385,4,385,107),2)
        region=text[prefix:]
        step=36 if len(region)==3 else 53
        for i,c in enumerate(region):
            glyph(c,397+i*step,17,29 if len(region)==3 else 39,54)
        glyph('RUS',397,86,46,12)
        if kind!='type1b':
            flag(465,85)
    for x in (9,w-10):
        d.ellipse(((x-2)*s,(h/2-2)*s,(x+2)*s,(h/2+2)*s),fill=(100,101,95))
    plate=np.asarray(im.resize((w,h),Image.Resampling.LANCZOS)).astype(np.float32)
    # Gentle surface variation, not invented unreadable characters or heavy occlusion.
    shade=cv2.resize(rng.uniform(.94,1.03,(3,5)).astype(np.float32),(w,h),interpolation=cv2.INTER_CUBIC)
    plate=plate*shade[:,:,None]+rng.normal(0,.6,plate.shape)
    return np.clip(plate,0,255).astype(np.uint8)


def replace(im, q, plate):
    """Replace only the annotated plate, retaining low-frequency scene lighting."""
    ph,pw=plate.shape[:2]
    h,w=im.shape[:2]
    dest=np.float32([[0,0],[pw-1,0],[pw-1,ph-1],[0,ph-1]])
    old=cv2.warpPerspective(im,cv2.getPerspectiveTransform(q,dest),(pw,ph))
    lum=cv2.cvtColor(old,cv2.COLOR_RGB2GRAY)
    # Constant exposure from the plate's bright surface; do not imprint old characters.
    exposure=float(np.clip(np.percentile(lum,80)/220,.48,1.08))
    plate=np.clip(plate.astype(np.float32)*exposure,0,255).astype(np.uint8)
    # Supersample just the plate ROI, then area-downsample to avoid jagged thin glyphs.
    lo=np.maximum(np.floor(q.min(0)).astype(int)-2,0)
    hi=np.minimum(np.ceil(q.max(0)).astype(int)+3,[w,h])
    rw,rh=hi-lo
    factor=3
    local=(q-lo).astype(np.float32)*factor
    matrix=cv2.getPerspectiveTransform(dest,local)
    warped=cv2.warpPerspective(plate,matrix,(rw*factor,rh*factor),borderMode=cv2.BORDER_REPLICATE)
    alpha=cv2.warpPerspective(np.ones((ph,pw),np.float32),matrix,(rw*factor,rh*factor))
    # Premultiply before downsampling, so the border is composited once.
    color=cv2.resize(warped.astype(np.float32)*alpha[:,:,None],(rw,rh),interpolation=cv2.INTER_AREA)
    alpha=cv2.resize(alpha,(rw,rh),interpolation=cv2.INTER_AREA)[:,:,None]
    result=im.copy()
    roi=im[lo[1]:hi[1],lo[0]:hi[0]]
    result[lo[1]:hi[1],lo[0]:hi[0]]=np.clip(color+roi*(1-alpha),0,255).astype(np.uint8)
    return result


def eligible(row, scale):
    q=numbers(row['quad'],8).reshape(4,2)*scale
    widths=[np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3])]
    heights=[np.linalg.norm(q[3]-q[0]),np.linalg.norm(q[2]-q[1])]
    return (row['is_vehicle']=='1' and row['plate_type']!='other'
            and min(widths)>=65 and min(heights)>=(38 if row['plate_type']=='type1a' else 16))


def generate(data, split_file, out, count=5000, seed=20261023):
    data, out = Path(data).resolve(), Path(out).resolve()
    cv_package,cv_version=opencv_package()
    if out.exists():
        raise ValueError('Output must be a new directory')
    cv2.setNumThreads(1)
    rng=np.random.default_rng(seed)
    split=json.loads(Path(split_file).read_text(encoding='utf-8-sig'))
    train=set(split['train'])
    holdout=set(split['val']+split['test'])
    if train & holdout or set(split['val']) & set(split['test']):
        raise ValueError('Split partitions overlap')
    real=[r for r in read_meta(data) if r['is_synthetic']=='0']
    grouped=defaultdict(list)
    for r in real:
        if r['image'] in train:
            grouped[r['image']].append(r)
    forbidden={r['plate_num'] for r in real if r['plate_type']!='other'}
    pools=defaultdict(list)
    dimensions={}
    # Pool membership is deterministic, and requires reasonably resolved originals.
    for name,rows in sorted(grouped.items()):
        with Image.open(data/name) as im:
            w,h=im.size
        scale=min(1,960/max(w,h))
        dimensions[name]=(w,h)
        kinds={r['plate_type'] for r in rows if eligible(r,scale)}
        if all(r['plate_type']=='other' for r in rows):
            kinds.add('other')
        for kind in sorted(kinds):
            pools[kind].append(name)
    kinds=['type1','type1a','type1b','other']
    if any(not pools[k] for k in kinds):
        raise ValueError(f'Insufficient backgrounds: {dict((k,len(pools[k])) for k in kinds)}')
    # 18/38/38/6 percent; other examples are augmented, not fabricated transcriptions.
    quotas=[int(count*p) for p in [.18,.38,.38]]
    quotas.append(count-sum(quotas))
    schedule=[k for k,n in zip(kinds,quotas) for _ in range(n)]
    rng.shuffle(schedule)
    out.mkdir(parents=True)
    rows_out=[]
    provenance=[]
    used=set()
    pool_counts=Counter()
    for i,kind in enumerate(schedule):
        # Cycle shuffled backgrounds for coverage rather than repeatedly drawing a few.
        cycle=pool_counts[kind]//len(pools[kind])
        position=pool_counts[kind]%len(pools[kind])
        if position==0:
            rng.shuffle(pools[kind])
        name=pools[kind][position]
        pool_counts[kind]+=1
        original=read_image(data/name)
        oh,ow=original.shape[:2]
        resize=min(1,960/max(ow,oh))
        w,h=round(ow*resize),round(oh*resize)
        im=cv2.resize(original,(w,h),interpolation=cv2.INTER_AREA) if resize<1 else original.copy()
        scale=np.array([w/ow,h/oh],np.float32)
        target=f'images/synthetic/composite_{i:06d}.jpg'
        frame=[]
        replaced=[]
        for index,source in enumerate(grouped[name]):
            row=dict(source)
            q=numbers(row['quad'],8).reshape(4,2)*scale
            if eligible(source,scale):
                text=random_number(rng,row['plate_type'])
                while text in forbidden:
                    text=random_number(rng,row['plate_type'])
                forbidden.add(text)
                im=replace(im,q,render(text,row['plate_type'],rng))
                row['plate_num']=text
                replaced.append(index)
            lo,hi=q.min(0),q.max(0)
            row.update(image=target,is_synthetic='1',
                       bbox=','.join(f'{x:.3f}' for x in [*lo,*(hi-lo)]),
                       quad=','.join(f'{x:.3f}' for x in q.ravel()),
                       source=f"{source['source']} | composite-v1; parent={name}; seed={seed}")
            # A derivative photograph retains its background license, not blanket CC-BY.
            frame.append(row)
        exposure=float(rng.uniform(.86,1.08))
        im=np.clip(im.astype(np.float32)*exposure+rng.uniform(-3,3),0,255)
        im=np.clip(im+rng.normal(0,float(rng.uniform(.3,1.3)),im.shape),0,255).astype(np.uint8)
        blurred=bool(rng.random()<.15)
        if blurred:
            im=cv2.GaussianBlur(im,(3,3),.42)
        for row in frame:
            if blurred:
                row['conditions']=','.join(dict.fromkeys(row['conditions'].split(',')+['blur']))
        write_image(out/target,im,quality=int(rng.integers(88,97)))
        rows_out.extend(frame)
        used.add(name)
        provenance.append({'image':target,'parent':name,'primary_type':kind,
                           'replaced_row_indices':replaced,'exposure':exposure,
                           'augmentation_only':not bool(replaced)})
        if (i+1)%100==0:
            print(f'Generated {i+1}/{count}',flush=True)
    write_meta(out,rows_out)
    write_labels(out,rows_out)
    parents={name:{'sha256':hashlib.sha256((data/name).read_bytes()).hexdigest(),
                   'annotations':grouped[name]} for name in sorted(used)}
    report={'version':'composite-v1','seed':seed,'count':count,
            'primary_type_images':dict(Counter(schedule)),
            'eligible_backgrounds':{k:len(pools[k]) for k in kinds},
            'used_backgrounds':len(used),'parents':parents,'images':provenance}
    (out/'provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    gd=out/'generator'
    (gd/'alpr').mkdir(parents=True)
    (gd/'assets').mkdir()
    for filename in ['__init__.py','common.py','generate.py','generate_composite.py']:
        shutil.copy2(Path(__file__).parent/filename,gd/'alpr'/filename)
    for filename in [FONT.name,'FONT-LICENSE.txt']:
        shutil.copy2(FONT.parent/filename,gd/'assets'/filename)
    shutil.copy2(split_file,gd/'real_split.json')
    recipe={'version':'composite-v1','count':count,'seed':seed,
            'numpy':np.__version__,'opencv':cv2.__version__,
            'opencv_package':cv_package+'=='+cv_version,
            'pillow':__import__('PIL').__version__,
            'font_sha256':hashlib.sha256(FONT.read_bytes()).hexdigest(),
            'input_meta_sha256':hashlib.sha256((data/'meta.csv').read_bytes()).hexdigest()}
    (gd/'recipe.json').write_text(json.dumps(recipe,indent=2),encoding='utf-8')
    (gd/'requirements.txt').write_text(f'numpy=={np.__version__}\nPillow=={__import__("PIL").__version__}\n{cv_package}=={cv_version}\n',encoding='utf-8')
    (gd/'generate.py').write_text('from alpr.generate_composite import main\nif __name__ == "__main__":\n    main()\n',encoding='utf-8')
    (out/'LICENSE').write_text('Derivative photographs: retain the source licenses and attribution in meta.csv and provenance.json. No blanket license override. Generator/plate artwork: CC BY 4.0, attribution to project participant (supply publication name). Font: generator/assets/FONT-LICENSE.txt.\n',encoding='utf-8')
    (out/'README.md').write_text('''# Composite synthetic data

New plate text composited onto real TRAIN photographs, with inherited perspective,
exposure, mild noise/blur and JPEG compression. Validation/test photographs are never used.
type1/type1a: A123BC77 or A123BC777; yellow type1b: AA12377 (user-confirmed).
The assignment PDF describes type1b differently; this discrepancy is documented.
Typeface and geometry remain approximations, not certified GOST artwork.
No claim of added rain, snow, dirt or night scenes; existing conditions are inherited.
All original annotations, including distant secondary plates, are retained and scaled.
Only adequately resolved target plates on vehicles are replaced. Other-only examples
receive mild photographic augmentation and retain their original text and class.
5000 variants are not 5000 independent backgrounds; parent counts are in provenance.json.
Small unmodified plates retain the source transcription, including # where already present.
Visual sampling is not an exhaustive readability or face/privacy audit.
Licenses and attribution follow source photographs; these are derivative images.

Reproduce with the same source dataset (or mixed dataset retaining images/real and real
metadata), this bundled generator and the recorded dependency versions:
python generator/generate.py --data SOURCE_DATASET --split generator/real_split.json --out NEW_DIR --count 5000 --seed 20261023
Parent hashes and source annotations are in provenance.json. Do not move any generated
variant into validation/test independently of its parent. Improvement in recognition
must be measured on real held-out images; it has not been established by generation.
''',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['count','primary_type_images','eligible_backgrounds','used_backgrounds']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--split',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--count',type=int,default=5000)
    p.add_argument('--seed',type=int,default=20261023)
    a=p.parse_args()
    if a.count<20:
        p.error('count must be >=20 to cover all classes')
    generate(a.data,a.split,a.out,a.count,a.seed)


if __name__=='__main__':
    main()
