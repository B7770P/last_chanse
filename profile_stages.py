"""Время по этапам итоговой конфигурации: на что уходят миллисекунды.

    python profile_stages.py --data dataset_composite5000 --split splits/real606.json --part val
    python profile_stages.py --input test

Меряется на val (или любой папке), НЕ на fin_test. Результат: profile_stages.json и таблица в консоли.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import alpr.alignment as alignment
import alpr.infer as infer
import alpr.type1_crops as type1_crops
from estimate_speed import FINAL_ARGS

times = defaultdict(float)
calls = defaultdict(int)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(name, fn):
    def wrapper(*args, **kwargs):
        sync()
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            sync()
            times[name] += (time.perf_counter() - start) * 1000
            calls[name] += 1
    return wrapper


def option(name, default=None):
    return FINAL_ARGS[FINAL_ARGS.index(name) + 1] if name in FINAL_ARGS else default


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input')
    p.add_argument('--data')
    p.add_argument('--split')
    p.add_argument('--part', default='val')
    p.add_argument('--out', default='profile_stages.json')
    a = p.parse_args()
    if a.input:
        files = sorted(x for x in Path(a.input).iterdir()
                       if x.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'))
    elif a.data and a.split:
        names = json.loads(Path(a.split).read_text(encoding='utf-8'))[a.part]
        files = [Path(a.data) / n for n in names]
    else:
        sys.exit('Укажи --input или --data вместе с --split')
    if 'fin_test' in str(a.input or ''):
        sys.exit('fin_test не трогаем: это итоговый отложенный тест')

    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)
    pipe = infer.Pipeline(option('--detector'), option('--ocr'), 'cuda', None, .35, .65, .5, True, False, False,
                          'yolo', True, option('--white-detector'), option('--type1-ocr'), True,
                          float(option('--type1-char-threshold')), '--multiscale-type1' in FINAL_ARGS,
                          '--direct-yolo' in FINAL_ARGS, '--gate-second-pass' in FINAL_ARGS,
                          cuda_graphs='--cuda-graphs' in FINAL_ARGS)
    max_side = int(option('--decode-max-side', 0))
    pipe(np.full((pipe.size, pipe.size, 3), 114, np.uint8))  # прогрев
    sync()

    read = timed('чтение и декодирование фото', infer.read_image)
    original_yolo = pipe._yolo

    def yolo(which, im, size, conf):
        name = 'детектор 1 (основной YOLO)' if which == 'base' else f'детектор 2 (белые) {size}px'
        return timed(name, original_yolo)(which, im, size, conf)
    pipe._yolo = yolo
    alignment.alignment_quads = timed('поиск контура номера (выравнивание)', alignment.alignment_quads)
    type1_crops.context_crops = timed('доп. вырезки type1', type1_crops.context_crops)
    infer.rectify = timed('вырезка и выпрямление номера', infer.rectify)
    pipe.recognize_crops = timed('OCR (нейросеть)', pipe.recognize_crops)
    infer.decode_text = timed('расшифровка OCR', infer.decode_text)
    single = pipe._single_pass
    pipe._single_pass = lambda *args, **kwargs: (calls.__setitem__('проходов', calls['проходов'] + 1),
                                                 single(*args, **kwargs))[1]

    totals, sizes, crops = [], [], 0
    for path in files:
        start = time.perf_counter()
        im = read(path, max_side)
        out = pipe(im)
        sync()
        totals.append((time.perf_counter() - start) * 1000)
        sizes.append(im.shape[0] * im.shape[1] / 1e6)
        crops += len(out)

    n = len(totals)
    stages = {k: {'mean_ms': round(v / n, 2), 'calls_per_image': round(calls[k] / n, 2)}
              for k, v in sorted(times.items(), key=lambda kv: -kv[1])}
    accounted = sum(times.values()) / n
    result = {
        'images': n,
        'gpu': torch.cuda.get_device_name(),
        'total_mean_ms': round(float(np.mean(totals)), 1),
        'total_p95_ms': round(float(np.percentile(totals, 95)), 1),
        'stages': stages,
        'other_python_ms': round(float(np.mean(totals)) - accounted, 2),
        'second_pass_share': round((calls['проходов'] - n) / n, 2),
        'megapixels_mean': round(float(np.mean(sizes)), 2),
        'megapixels_max': round(float(np.max(sizes)), 2),
        'note': 'Таймеры с синхронизацией CUDA, поэтому сумма немного больше обычного замера.',
    }
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"фото: {n}, в среднем {result['total_mean_ms']} мс, p95 {result['total_p95_ms']} мс, "
          f"{result['megapixels_mean']} Мп в среднем")
    for k, v in stages.items():
        print(f"  {v['mean_ms']:7.2f} мс  {k}  (вызовов на фото: {v['calls_per_image']})")
    print(f"  {result['other_python_ms']:7.2f} мс  прочее (Python, ultralytics вне predict)")
    print(f"второй проход на {result['second_pass_share'] * 100:.0f}% фото; сохранено в {a.out}")


if __name__ == '__main__':
    main()
