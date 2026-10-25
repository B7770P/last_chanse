"""YOLO export: yolo_real606 (train/val/test как есть) + сцены из roboflow_geometry в train, класс type1.

    python build_yolo_rf.py --out yolo_real606_rf

Из roboflow_geometry берутся только сцены (номер не шире 60 % кадра). Отбрасываются:
- почти-дубликаты (dHash <= 4) реальных фото датасета (все 606), fin_test, test и test22;
- фото с квадратным номером (рамка уже 2.5:1) или с жёлтым номером: класс type1 для них был бы неверным.
Исходные папки не меняются, всё копируется в новую. Фото уменьшаются до 1280 px, как в alpr/prepare_yolo.py.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

YOLO_SRC = Path(r'C:\Users\glamurchik\.codex\.chatgpt-projects\g-p-6ac900fb802c8191802f460936af3320\yolo_real606')
RF = Path('roboflow_geometry')
REFS = ['dataset_composite5000/images/real', 'fin_test', 'test', 'test22']
EXT = {'.jpg', '.jpeg', '.png', '.webp'}
TYPE1 = 0  # alpr.common.TYPES = ('type1', 'type1a', 'type1b', 'other')


def dhash(im):
    px = np.asarray(im.convert('L').resize((9, 8), Image.LANCZOS), np.int16)
    return int(sum(1 << k for k, v in enumerate((px[:, :8] > px[:, 1:]).ravel()) if v))


def yellow(im, box):
    x, y, w, h = box
    crop = np.asarray(im.convert('HSV').crop((int(x), int(y), int(x + w), int(y + h))), np.float32)
    if crop.size == 0:
        return False
    hue, sat = crop[..., 0], crop[..., 1]
    return float(((hue > 20) & (hue < 50) & (sat > 90)).mean()) > .35  # PIL hue 0-255: yellow ~ 28-45


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='yolo_real606_rf')
    a = p.parse_args()
    out = Path(a.out)
    assert not out.exists(), f'{out} already exists'
    for part in ('train', 'val', 'test'):
        shutil.copytree(YOLO_SRC / 'images' / part, out / 'images' / part)
        shutil.copytree(YOLO_SRC / 'labels' / part, out / 'labels' / part)
    refs = [dhash(Image.open(f)) for d in REFS for f in Path(d).rglob('*') if f.suffix.lower() in EXT]
    g = json.load(open(RF / 'geometry.json', encoding='utf-8'))
    stats = dict(rf_images=len(g['images']), no_plates=0, crops=0, near_duplicate=0, square=0, yellow=0, added=0, plates=0)
    added = []
    for rec in g['images']:
        objs = rec['objects']
        if not objs:
            stats['no_plates'] += 1
            continue
        if max(o['bbox'][2] for o in objs) > .6 * rec['width']:
            stats['crops'] += 1
            continue
        im = Image.open(RF / rec['image']).convert('RGB')
        h = dhash(im)
        if any(bin(h ^ r).count('1') <= 4 for r in refs):
            stats['near_duplicate'] += 1
            continue
        if any(o['bbox'][2] < 2.5 * o['bbox'][3] for o in objs):
            stats['square'] += 1
            continue
        if any(yellow(im, o['bbox']) for o in objs):
            stats['yellow'] += 1
            continue
        w, hgt = im.size
        small = im.copy()
        small.thumbnail((1280, 1280))
        name = f'rf_{Path(rec["image"]).stem}.jpg'
        small.save(out / 'images' / 'train' / name, quality=95)
        lines = []
        for o in objs:
            x, y, bw, bh = o['bbox']
            lines.append(f'{TYPE1} {(x + bw / 2) / w:.9f} {(y + bh / 2) / hgt:.9f} {bw / w:.9f} {bh / hgt:.9f}')
        (out / 'labels' / 'train' / f'rf_{Path(rec["image"]).stem}.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        stats['added'] += 1
        stats['plates'] += len(objs)
        added.append(dict(image=name, source=rec['image'], original_filename=rec['original_filename'], license=rec['license'],
                          source_url=rec['source']))
    (out / 'data.yaml').write_text(f'path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n'
                                   'names:\n  0: type1\n  1: type1a\n  2: type1b\n  3: other\n', encoding='utf-8')
    counts = {part: len(list((out / 'images' / part).iterdir())) for part in ('train', 'val', 'test')}
    (out / 'manifest_rf.json').write_text(json.dumps(dict(stats=stats, counts=counts, base=str(YOLO_SRC), added=added),
                                                     ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(dict(stats=stats, counts=counts), ensure_ascii=False))


if __name__ == '__main__':
    main()
