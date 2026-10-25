"""Подбор порогов итогового режима на val: порог детекции и порог уверенности символа.

    python sweep_thresholds.py

Для каждой пары порогов печатает: верно прочитанных номеров, лишних выводов (выданная строка,
которой нет в разметке этого фото) и other, выданных как type1. fin_test здесь не используется.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

import compare_fast_v5 as cf
from alpr.common import read_meta
from alpr.infer import Pipeline

CYR = str.maketrans('АВЕКМНОРСТУХ', 'ABEKMHOPCTYX')


def extra_outputs(preds, truth):
    """Outputs whose (text, type) is not among the labelled plates of that image."""
    count = 0
    for name, rows in truth.items():
        labelled = {(r['plate_num'].strip().upper().translate(CYR), r['plate_type']) for r in rows}
        count += sum(1 for p in preds.get(name, []) if p not in labelled)
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default='dataset_composite5000')
    p.add_argument('--split', default='splits/real606.json')
    p.add_argument('--ocr', default='models/ocr_synth_20261025/best.pt')
    p.add_argument('--det', default='0.25,0.3,0.35')
    p.add_argument('--char', default='0.5,0.55,0.65')
    p.add_argument('--out', default='sweep_thresholds.json')
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
    truth = {n: meta[n] for n in names}
    report = []
    for det in (float(x) for x in a.det.split(',')):
        for char in (float(x) for x in a.char.split(',')):
            pipe = Pipeline(cf.MODELS['detector'], a.ocr, 'cuda', det_threshold=det, char_threshold=char,
                            keep_nonvehicle=True, detector_backend='yolo', align=True,
                            white_detector=cf.MODELS['white'], type1_ocr=cf.MODELS['type1_ocr'],
                            type1_char_threshold=.5, multiscale_type1=True, direct_yolo=True,
                            gate_second_pass=True, cuda_graphs=True)
            preds, times = cf.run(pipe, files, 1600, 'cuda', prefetch=True)
            by_type, correct, other = cf.score(preds, truth)
            extra = extra_outputs(preds, truth)
            mark = '  <- сейчас' if (det, char) == (.35, .65) else ''
            print(f'детекция {det:.2f}, символ {char:.2f}: верно {correct} {by_type}, лишних {extra}, '
                  f'other->type1 {other}{mark}', flush=True)
            report.append(dict(det=det, char=char, correct=correct, by_type=by_type, extra=extra, other_as_type1=other))
            del pipe
            torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
