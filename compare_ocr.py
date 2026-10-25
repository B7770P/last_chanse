"""Итоговый режим (v5) с прежним и с новым общим OCR: сравнение на val.

    python compare_ocr.py --candidate models/ocr_synth_20261025/best.pt

Новый OCR принимаем, только если на val он заметно лучше. fin_test здесь не используется.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

import compare_fast_v5 as cf
from alpr.common import read_meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--candidate', required=True)
    p.add_argument('--baseline', help='текущий OCR (по умолчанию models/ocr_crops_20261024)')
    p.add_argument('--data', default='dataset_composite5000')
    p.add_argument('--split', default='splits/real606.json')
    p.add_argument('--out', default='compare_ocr.json')
    a = p.parse_args()
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)
    split = json.loads(Path(a.split).read_text(encoding='utf-8'))
    meta = defaultdict(list)
    for r in read_meta(a.data):
        meta[r['image']].append(r)
    names = split['val']
    files = [(n, Path(a.data) / n) for n in names]
    config = dict(cf.FAST, cuda_graphs=True)
    report, baseline = {}, None
    for title, ocr in [('прежний OCR', a.baseline or cf.MODELS['ocr']), ('новый OCR', a.candidate)]:
        cf.MODELS['ocr'] = ocr
        pipe = cf.build(config, 'cuda')
        preds, times = cf.run(pipe, files, config['decode_max_side'], 'cuda', prefetch=True)
        by_type, correct, other = cf.score(preds, {n: meta[n] for n in names})
        baseline = baseline or preds
        changed = sum(preds[n] != baseline[n] for n in names)
        report[title] = dict(ocr=ocr, correct=correct, by_type=by_type, other_as_type1=other, images_changed=changed)
        print(f'{title:12s} val: {correct} верно {by_type}, other->type1 {other}, изменилось фото {changed}', flush=True)
        del pipe
        torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
