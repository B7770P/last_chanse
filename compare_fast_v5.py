"""Сравнение быстрых режимов с текущим итоговым: качество и скорость на val (и на type1 из train).

    python compare_fast.py

val: 91 фото без type1. Чтобы увидеть, не теряет ли что-то type1 (второй проход 960px нужен
только ему), отдельно прогоняются 100 реальных type1 из train: модели их видели, поэтому
абсолютные цифры там завышены, смотреть нужно только на разницу между режимами.
fin_test здесь не используется.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from alpr.common import read_image, read_meta
from alpr.infer import Pipeline

ROOT = Path(__file__).resolve().parent
MODELS = dict(
    detector=str(ROOT / 'models/yolo11n_real_20261024/weights/best.pt'),
    white=str(ROOT / 'models/yolo11n_white_20261024/weights/best.pt'),
    ocr=str(ROOT / 'models/ocr_crops_20261024/best.pt'),
    type1_ocr=str(ROOT / 'models/ocr_type1_small_20261025/best.pt'),
)
FAST = dict(direct_yolo=True, gate_second_pass=True, decode_max_side=1600)
CONFIGS = {
    'итоговый v4 (быстрый)': FAST,
    'квадратный вход без графов': dict(FAST, cuda_graphs=True, drop_graphs=True),
    'квадратный + CUDA graphs': dict(FAST, cuda_graphs=True),
    'квадратный + графы + чтение заранее': dict(FAST, cuda_graphs=True, prefetch=True),
}
CYR = str.maketrans('АВЕКМНОРСТУХ', 'ABEKMHOPCTYX')


def build(config, device):
    white = config.get('white', True)
    return Pipeline(MODELS['detector'], MODELS['ocr'], device, keep_nonvehicle=True, detector_backend='yolo',
                    align=True, white_detector=MODELS['white'] if white else None, type1_ocr=MODELS['type1_ocr'],
                    type1_char_threshold=.5, multiscale_type1=white,
                    direct_yolo=config.get('direct_yolo', False), gate_second_pass=config.get('gate_second_pass', False),
                    cuda_graphs=config.get('cuda_graphs', False))


def run(pipe, files, max_side, device, prefetch=False):
    pipe(np.full((640, 640, 3), 114, np.uint8))
    preds, times = {}, []
    pool = None
    if prefetch:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(1)
        pending = pool.submit(read_image, files[0][1], max_side)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    last = time.perf_counter()
    for index, (name, path) in enumerate(files):
        start = last if pool else time.perf_counter()
        if pool:
            future = pending
            if index + 1 < len(files):
                pending = pool.submit(read_image, files[index + 1][1], max_side)
            image = future.result()
        else:
            image = read_image(path, max_side)
        out = pipe(image)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        last = time.perf_counter()
        times.append((last - start) * 1000)
        preds[name] = sorted((o['plate_num'], o['plate_type']) for o in out)
    return preds, times


def score(preds, truth):
    stats = defaultdict(lambda: [0, 0])
    other_as_type1 = 0
    for name, rows in truth.items():
        got = preds.get(name, [])
        targets = [r for r in rows if r['plate_type'] != 'other']
        for r in targets:
            text = r['plate_num'].strip().upper().translate(CYR)
            if not text or '#' in text:
                continue
            stats[r['plate_type']][1] += 1
            stats[r['plate_type']][0] += (text, r['plate_type']) in got
        if not targets and any(k == 'type1' for _, k in got):
            other_as_type1 += 1
    return {k: f'{c}/{t}' for k, (c, t) in sorted(stats.items())}, sum(c for c, _ in stats.values()), other_as_type1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default='dataset_composite5000')
    p.add_argument('--split', default='splits/real606.json')
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', default='compare_fast_v5.json')
    a = p.parse_args()
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)

    split = json.loads(Path(a.split).read_text(encoding='utf-8'))
    meta = defaultdict(list)
    for r in read_meta(a.data):
        meta[r['image']].append(r)
    sets = {
        'val': split['val'],
        'type1 из train': [n for n in split['train'] if any(r['plate_type'] == 'type1' for r in meta[n])],
        'весь train': split['train'],  # 424 фото разных размеров: проверка, что ничего не падает
    }
    report = {}
    baseline = {}
    for title, config in CONFIGS.items():
        if a.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        pipe = build(config, a.device)
        if config.get('drop_graphs'):
            for model in pipe._direct.values():
                model.graphs = {}
        report[title] = {}
        for part, names in sets.items():
            files = [(n, Path(a.data) / n) for n in names]
            preds, times = run(pipe, files, config.get('decode_max_side', 0), a.device, config.get('prefetch', False))
            by_type, correct, other = score(preds, {n: meta[n] for n in names})
            baseline.setdefault(part, preds)
            changed = sum(preds[n] != baseline[part][n] for n in names)
            report[title][part] = dict(correct=correct, by_type=by_type, other_as_type1=other,
                                       images_changed_vs_current=changed,
                                       mean_ms=round(float(np.mean(times)), 1),
                                       p95_ms=round(float(np.percentile(times, 95)), 1))
        memory = round(torch.cuda.max_memory_reserved() / 2**20) if a.device.startswith('cuda') else 0
        report[title]['gpu_memory_mb'] = memory
        del pipe
        if a.device.startswith('cuda'):
            torch.cuda.empty_cache()
        v, t = report[title]['val'], report[title]['type1 из train']
        print(f"{title:36s} val: {v['correct']:3d} верно {v['by_type']}, other->type1 {v['other_as_type1']}, "
              f"изменилось фото {v['images_changed_vs_current']}, {v['mean_ms']} мс (p95 {v['p95_ms']}) | "
              f"type1 train: {t['correct']} верно, изменилось {t['images_changed_vs_current']}, {t['mean_ms']} мс | "
              f"видеопамять до {memory} МБ | весь train: {report[title]['весь train']['correct']} верно, "
              f"{report[title]['весь train']['mean_ms']} мс",
              flush=True)
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('сохранено в', a.out)


if __name__ == '__main__':
    main()
