"""Разбор ошибок итогового режима (v7) по каждому размеченному номеру. fin_test здесь не используется.

    python analyze_val_errors.py                                   # val -> val_errors.json
    python analyze_val_errors.py --split-part train --limit 200 --out val_errors_train.json

Для каждого номера type1/type1a/type1b из разметки: missed (ни одна рамка решения не пересекается
с рамкой разметки: IoU >= 0.3 или центр одной внутри другой; делится на missed_not_detected — основной
детектор рамку не нашёл, и missed_no_text — рамка есть, но текст отброшен или тип other), wrong_type, read_with_hash,
wrong_chars_1 / _2 / _3+ (расстояние Левенштейна), correct. Номера other: not_detected,
detected_not_output (найден, в вывод не идёт — так и задумано), output_as_type1x.
"""
import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import compare_fast_v5 as cf
from alpr.common import read_image, read_meta
from alpr.infer import Pipeline

CYR = str.maketrans('АВЕКМНОРСТУХ', 'ABEKMHOPCTYX')
TARGETS = ('type1', 'type1a', 'type1b')


def norm(text):
    return text.strip().upper().replace(' ', '').translate(CYR)


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def iou(a, b):
    w = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


def inside(box, point):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def overlaps(label, pred):
    lc = ((label[0] + label[2]) / 2, (label[1] + label[3]) / 2)
    pc = ((pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2)
    return iou(label, pred) >= .3 or inside(label, pc) or inside(pred, lc)


def classify(row, box, preds, candidates):
    near = [p for p in preds if overlaps(box, p['box'])]
    found = [c for c in candidates if overlaps(box, c['box'])]
    kind = row['plate_type']
    if kind == 'other':
        bad = [p for p in near if p['plate_type'] in TARGETS]
        if bad:
            return 'output_as_type1x', bad[0]
        return ('detected_not_output', None) if found else ('not_detected', None)
    if not near:
        if found:
            best = max(found, key=lambda c: iou(box, c['box']))
            return 'missed_no_text', dict(plate_num='', plate_type=best['plate_type'], confidence=best['score'])
        return 'missed_not_detected', None
    truth = norm(row['plate_num'])
    same = [p for p in near if p['plate_type'] == kind]
    if any(p['plate_num'] == truth for p in same):
        return 'correct', next(p for p in same if p['plate_num'] == truth)
    if not same:
        return 'wrong_type', max(near, key=lambda p: iou(box, p['box']))
    best = min(same, key=lambda p: (levenshtein(p['plate_num'], truth), -iou(box, p['box'])))
    if '#' in best['plate_num']:
        return 'read_with_hash', best
    d = levenshtein(best['plate_num'], truth)
    return f'wrong_chars_{d if d < 3 else "3+"}', best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default='dataset_composite5000')
    p.add_argument('--split', default='splits/real606.json')
    p.add_argument('--split-part', default='val', choices=['train', 'val', 'test'])
    p.add_argument('--limit', type=int, default=0, help='не больше N фото (0 = все)')
    p.add_argument('--ocr', default='models/ocr_synth2_20261025/best.pt')
    p.add_argument('--det', type=float, default=.35)
    p.add_argument('--char', type=float, default=.4)
    p.add_argument('--extra-size', type=int, default=0, help='добавочный проход основного детектора (V4: 960)')
    p.add_argument('--extra-gate', default='large_or_empty')
    p.add_argument('--length-repair', action='store_true')
    p.add_argument('--mask-decode', action='store_true')
    p.add_argument('--decode', type=int, default=1600, help='--decode-max-side (v8: 2048)')
    p.add_argument('--out', default='val_errors.json')
    a = p.parse_args()
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)

    split = json.loads(Path(a.split).read_text(encoding='utf-8'))
    meta = defaultdict(list)
    all_rows = read_meta(a.data)
    for r in all_rows:
        meta[r['image']].append(r)

    print('Размеченные номера в splits/real606.json:')
    for part in ('train', 'val', 'test'):
        c = Counter(r['plate_type'] for n in split[part] for r in meta[n])
        print(f'  {part:5s}: {len(split[part])} фото, ' + ', '.join(f'{k} {c[k]}' for k in (*TARGETS, 'other')))
    synth = [r for r in all_rows if str(r.get('is_synthetic')) == '1']
    synth_type1 = {r['image'] for r in synth if r['plate_type'] == 'type1'}
    print(f'  синтетика в {a.data}: {len({r["image"] for r in synth})} фото, из них с type1: {len(synth_type1)} '
          f'(номеров type1: {sum(r["plate_type"] == "type1" for r in synth)})')

    names = split[a.split_part][:a.limit or None]
    pipe = Pipeline(cf.MODELS['detector'], a.ocr, 'cuda', det_threshold=a.det, char_threshold=a.char,
                    keep_nonvehicle=True, detector_backend='yolo', align=True,
                    white_detector=cf.MODELS['white'], type1_ocr=cf.MODELS['type1_ocr'],
                    type1_char_threshold=.5, multiscale_type1=True, direct_yolo=True,
                    gate_second_pass=True, cuda_graphs=True, length_repair=a.length_repair, mask_decode=a.mask_decode,
                    **(dict(base_extra_size=a.extra_size, base_extra_gate=a.extra_gate) if a.extra_size else {}))
    pipe(np.full((640, 640, 3), 114, np.uint8))

    records = []
    for name in names:
        path = Path(a.data) / name
        with Image.open(path) as probe:
            width, height = probe.size
        im = read_image(path, a.decode)
        scale = width / im.shape[1]
        preds = [dict(plate_num=o['plate_num'], plate_type=o['plate_type'], confidence=float(o['confidence']),
                      box=[float(v) * scale for v in o['box'][:4]]) for o in pipe(im)]
        candidates = [dict(plate_type=c['plate_type'], score=float(c['score']),
                           box=[float(v) * scale for v in c['box'][:4]]) for c in pipe._base_candidates]
        for row in meta[name]:
            x, y, w, h = (float(v) for v in row['bbox'].split(','))
            box = [x, y, x + w, y + h]
            truth = norm(row['plate_num'])
            if row['plate_type'] in TARGETS and (not truth or '#' in truth):
                category, match = 'unreadable_label', None
            else:
                category, match = classify(row, box, preds, candidates)
            records.append(dict(image=name, plate_type=row['plate_type'], truth=truth, category=category,
                                pred=f'{match["plate_num"] or "<нет текста>"} {match["plate_type"]}' if match else None,
                                pred_conf=round(match['confidence'], 3) if match else None,
                                width=round(w, 1), height=round(h, 1), image_w=width, image_h=height,
                                all_preds=[f'{q["plate_num"]} {q["plate_type"]}' for q in preds]))
    del pipe
    torch.cuda.empty_cache()

    print(f'\nЧасть {a.split_part}: {len(names)} фото, OCR {a.ocr}, det {a.det}, char {a.char}, '
          f'extra {a.extra_size or "нет"}, length_repair {a.length_repair}, mask_decode {a.mask_decode}')
    cats = ['correct', 'read_with_hash', 'wrong_chars_1', 'wrong_chars_2', 'wrong_chars_3+', 'wrong_type',
            'missed_no_text', 'missed_not_detected', 'unreadable_label']
    print(f'{"тип":7s} {"всего":>5s} ' + ' '.join(f'{c:>14s}' for c in cats)
          + '  медиана ширины: missed_no_text / missed_not_detected / correct')
    summary = {}
    for kind in TARGETS:
        rows = [r for r in records if r['plate_type'] == kind]
        if not rows:
            print(f'{kind:7s} {0:5d}')
            continue
        c = Counter(r['category'] for r in rows)
        med = {k: (statistics.median(r['width'] for r in rows if r['category'] == k) if c[k] else None)
               for k in ('missed_no_text', 'missed_not_detected', 'correct')}
        summary[kind] = dict(counts=dict(c), median_width=med)
        fmt = lambda v: '-' if v is None else f'{v:.0f}'
        print(f'{kind:7s} {len(rows):5d} ' + ' '.join(f'{c[k]:14d}' for k in cats)
              + f'  {fmt(med["missed_no_text"])} / {fmt(med["missed_not_detected"])} / {fmt(med["correct"])} px')
    others = Counter(r['category'] for r in records if r['plate_type'] == 'other')
    summary['other'] = dict(others)
    print(f'other: {sum(others.values())} номеров: ' + ', '.join(f'{k} {v}' for k, v in sorted(others.items())))

    order = {'missed_not_detected': 0, 'missed_no_text': 1, 'wrong_type': 2, 'wrong_chars_3+': 3, 'read_with_hash': 4,
             'wrong_chars_2': 5, 'wrong_chars_1': 6}
    worst = sorted((r for r in records if r['category'] in order and r['plate_type'] in TARGETS),
                   key=lambda r: (order[r['category']], -r['width']))[:15]
    print('\n15 худших примеров (сначала пропуски, крупные первыми):')
    for r in worst:
        print(f'  {r["category"]:19s} {r["plate_type"]:6s} {r["truth"]:10s} -> {str(r["pred"]):22s} '
              f'w={r["width"]:.0f} h={r["height"]:.0f} img={r["image_w"]}x{r["image_h"]}  {Path(r["image"]).name[:60]}')

    Path(a.out).write_text(json.dumps(dict(config=vars(a), summary=summary, records=records), ensure_ascii=False,
                                      indent=2), encoding='utf-8')
    print(f'\nподробности: {a.out}')


if __name__ == '__main__':
    main()
