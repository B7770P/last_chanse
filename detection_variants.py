"""Варианты детекции на val против пропусков детектора (основа: v7, OCR ocr_synth2, det 0.35, char 0.4).

    python detection_variants.py --out detection_variants.json

V0 как сейчас; V1 основной детектор дополнительно на 960 на каждом кадре; V2 первый проход обоих детекторов на 800;
V3 первый проход на 960; V4 основной на 960, только если длинная сторона >= 1600 или на 640 ничего не найдено;
плюс V0, V1 и V4 с --decode-max-side 2048; V5 = V4 + ремонт длины (--length-repair);
V6 = V4 + decode 2048 + ремонт длины; V7 = V5 + декодирование с маской (--mask-decode). Время: RTX 3060, с чтением в фоне и CUDA graphs, как в итоговом режиме.
fin_test здесь не используется.
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import compare_fast_v5 as cf
from alpr.common import read_image, read_meta
from alpr.infer import Pipeline
from analyze_val_errors import TARGETS, classify, norm
from sweep_thresholds import extra_outputs

VARIANTS = {
    'V0 как сейчас': dict(),
    'V1 основной +960 всегда': dict(base_extra_size=960),
    'V2 первый проход 800': dict(size=800),
    'V3 первый проход 960': dict(size=960),
    'V4 основной +960 (большие или пустые)': dict(base_extra_size=960, base_extra_gate='large_or_empty'),
    'V0 + decode 2048': dict(decode=2048),
    'V1 + decode 2048': dict(base_extra_size=960, decode=2048),
    'V4 + decode 2048': dict(base_extra_size=960, base_extra_gate='large_or_empty', decode=2048),
    'V5 = V4 + length_repair': dict(base_extra_size=960, base_extra_gate='large_or_empty', length_repair=True),
    'V6 = V4 + decode 2048 + length_repair': dict(base_extra_size=960, base_extra_gate='large_or_empty',
                                                  length_repair=True, decode=2048),
    'V7 = V5 + mask_decode': dict(base_extra_size=960, base_extra_gate='large_or_empty', length_repair=True,
                                  mask_decode=True),
    'A1 = V6 + crop_pad 0.04': dict(base_extra_size=960, base_extra_gate='large_or_empty', length_repair=True,
                                    decode=2048, crop_pad=.04),
    'A1b = V6 + crop_pad 0.08': dict(base_extra_size=960, base_extra_gate='large_or_empty', length_repair=True,
                                     decode=2048, crop_pad=.08),
    'A2 = V6 + multi_crop': dict(base_extra_size=960, base_extra_gate='large_or_empty', length_repair=True,
                                 decode=2048, multi_crop=True),
    'B1 = V6 + extra960 weights': dict(base_extra_size=960, base_extra_gate='large_or_empty', length_repair=True,
                                       decode=2048, base_extra_weights='models/yolo11n_extra960_20261025/weights/best.pt'),
    'B2 = V6 + extra960 weights, always': dict(base_extra_size=960, base_extra_gate='always', length_repair=True,
                                               decode=2048, base_extra_weights='models/yolo11n_extra960_20261025/weights/best.pt'),
}


def build(ocr, det, char, size=None, **extra):
    return Pipeline(cf.MODELS['detector'], ocr, 'cuda', size=size, det_threshold=det, char_threshold=char,
                    keep_nonvehicle=True, detector_backend='yolo', align=True,
                    white_detector=cf.MODELS['white'], type1_ocr=cf.MODELS['type1_ocr'],
                    type1_char_threshold=.5, multiscale_type1=True, direct_yolo=True,
                    gate_second_pass=True, cuda_graphs=True, **extra)


def run(pipe, files, max_side):
    """Same timing as compare_fast_v5.run with prefetch; also keeps boxes and detector candidates."""
    pipe(np.full((640, 640, 3), 114, np.uint8))
    pool = ThreadPoolExecutor(1)
    pending = pool.submit(read_image, files[0][1], max_side)
    outputs, times = {}, []
    torch.cuda.synchronize()
    last = time.perf_counter()
    for index, (name, path) in enumerate(files):
        future = pending
        if index + 1 < len(files):
            pending = pool.submit(read_image, files[index + 1][1], max_side)
        im = future.result()
        out = pipe(im)
        candidates = list(pipe._base_candidates)
        torch.cuda.synchronize()
        now = time.perf_counter()
        times.append((now - last) * 1000)
        last = now
        outputs[name] = (im.shape, out, candidates)
    pool.shutdown()
    return outputs, times


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default='dataset_composite5000')
    p.add_argument('--split', default='splits/real606.json')
    p.add_argument('--ocr', default='models/ocr_synth2_20261025/best.pt')
    p.add_argument('--detector', default=cf.MODELS['detector'], help='основной детектор (и для прохода 960, если не задано иное)')
    p.add_argument('--det', type=float, default=.35)
    p.add_argument('--char', type=float, default=.4)
    p.add_argument('--only', help='через запятую: V0,V1,V4+2048,... (по умолчанию все). Варианты в одном процессе '
                                  'влияют друг на друга, для чистого сравнения запускай по одному')
    p.add_argument('--out', default='detection_variants.json')
    a = p.parse_args()
    cf.MODELS['detector'] = a.detector
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)

    split = json.loads(Path(a.split).read_text(encoding='utf-8'))
    meta = defaultdict(list)
    for r in read_meta(a.data):
        meta[r['image']].append(r)
    names = split['val']
    files = [(n, Path(a.data) / n) for n in names]
    sizes = {}
    for n, path in files:
        with Image.open(path) as probe:
            sizes[n] = probe.size
    truth = {n: meta[n] for n in names}
    wanted = a.only.split(',') if a.only else None

    report = {}
    for title, config in VARIANTS.items():
        short = title.split()[0] + ('+2048' if 'decode 2048' in title else '')
        if wanted and short not in wanted:
            continue
        config = dict(config)
        decode = config.pop('decode', 1600)
        pipe = build(a.ocr, a.det, a.char, **config)
        outputs, times = run(pipe, files, decode)
        del pipe
        torch.cuda.empty_cache()

        preds = {n: sorted((o['plate_num'], o['plate_type']) for o in out) for n, (_, out, _) in outputs.items()}
        by_type, correct, other = cf.score(preds, truth)
        extra = extra_outputs(preds, truth)
        cats, plates = Counter(), []
        for n, (shape, out, candidates) in outputs.items():
            scale = sizes[n][0] / shape[1]
            boxes = lambda rows: [dict(r, box=[float(v) * scale for v in r['box'][:4]]) for r in rows]
            out, candidates = boxes(out), boxes(candidates)
            for row in meta[n]:
                text = norm(row['plate_num'])
                if row['plate_type'] not in TARGETS or not text or '#' in text:
                    continue
                x, y, w, h = (float(v) for v in row['bbox'].split(','))
                category, match = classify(row, [x, y, x + w, y + h], out, candidates)
                cats[category] += 1
                plates.append(dict(image=n, truth=text, plate_type=row['plate_type'], category=category,
                                   pred=f'{match["plate_num"]} {match["plate_type"]}' if match else None))
        t = np.array(times)
        result = dict(correct=correct, by_type=by_type, extra=extra, other_as_type1=other,
                      missed_not_detected=cats['missed_not_detected'], missed_no_text=cats['missed_no_text'],
                      categories=dict(cats), mean_ms=round(float(t.mean()), 1),
                      p95_ms=round(float(np.percentile(t, 95)), 1), plates=plates)
        report[title] = result
        print(f'{title:38s} верно {correct} {by_type}, лишних {extra}, other->type1 {other}, '
              f'не найдено {result["missed_not_detected"]}, без текста {result["missed_no_text"]}, '
              f'{result["mean_ms"]} мс / p95 {result["p95_ms"]} мс', flush=True)
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
