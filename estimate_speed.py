r"""Оценка скорости на GTX 1050 Ti + i5-7600 без этой видеокарты.

Два прогона итоговой конфигурации (как run_final.ps1) на твоей видеокарте:
1) обычный замер общего времени на кадр;
2) прогон с профайлером CUDA: сколько времени видеокарта реально считала (ядра CUDA).
Остальное время (чтение, выравнивание, постобработка и запуск ядер из Python) зависит
от процессора. Оценка:
    1050 Ti = время_GPU * k_gpu + время_CPU * k_cpu
k_gpu = 3..5: во сколько раз 1050 Ti медленнее RTX 3060 в вычислениях.
k_cpu: во сколько раз i5-7600 медленнее твоего процессора в одном потоке (по умолчанию 1.5;
для Intel 10-го поколения вроде i5-10400 ставь --cpu-factor 1.2).
Сумма без учёта перекрытия CPU и GPU, то есть оценка сверху.

Это ОЦЕНКА, а не замер: в записке так и пиши.

    python estimate_speed.py --input test --out speed_estimate.json
    python estimate_speed.py --input test --out v.json --base-extra-size 960   # лишние флаги уходят в alpr.infer
"""
import argparse
import json
import platform
import sys
from pathlib import Path

import torch

import alpr.infer as infer

ROOT = Path(__file__).resolve().parent
FINAL_ARGS = [
    '--detector-backend', 'yolo',
    '--detector', str(ROOT / 'models/yolo11n_real_20261024/weights/best.pt'),
    '--white-detector', str(ROOT / 'models/yolo11n_white_20261024/weights/best.pt'),
    '--ocr', str(ROOT / 'models/ocr_synth2_20261025/best.pt'),
    '--char-threshold', '0.4',
    '--type1-ocr', str(ROOT / 'models/ocr_type1_small_20261025/best.pt'),
    '--type1-char-threshold', '0.5',
    '--multiscale-type1',
    '--threads', '1',
    '--keep-nonvehicle',
    '--direct-yolo',
    '--gate-second-pass',
    '--decode-max-side', '2048',
    '--base-extra-size', '960',
    '--base-extra-gate', 'large_or_empty',
    '--length-repair',
    '--cuda-graphs',
    '--prefetch',
    '--device', 'cuda',
]

state = {'started': False, 'profiler': None}
original_read = infer.read_image


def marked_read(path, *args):
    # первый кадр: прогрев в main() уже прошёл
    if not state['started'] and state['profiler'] is not None:
        state['profiler'].start()  # прогрев не профилируем
    state['started'] = True
    return original_read(path, *args)


def cpu_name():
    try:
        import subprocess
        out = subprocess.run(['wmic', 'cpu', 'get', 'name'], capture_output=True, text=True).stdout
        lines = [x.strip() for x in out.splitlines() if x.strip() and x.strip() != 'Name']
        if lines:
            return lines[0]
    except Exception:
        pass
    return platform.processor()


EXTRA_ARGS = []  # флаги alpr.infer поверх FINAL_ARGS (последнее значение флага побеждает)


def run(input_dir, csv_path, bench_path, prefetch=True):
    args = [x for x in FINAL_ARGS + EXTRA_ARGS if prefetch or x != '--prefetch']
    sys.argv = ['alpr.infer', '--input', input_dir, '--output', str(csv_path),
                '--benchmark', str(bench_path)] + args
    infer.read_image = marked_read
    try:
        infer.main()
    finally:
        infer.read_image = original_read
    return json.loads(bench_path.read_text(encoding='utf-8'))


def kernel_ms(profiler):
    # суммируем только события на видеокарте (ядра и копирования), без операций CPU
    from torch.autograd import DeviceType
    total_us = 0.0
    for event in profiler.events():
        if event.device_type == DeviceType.CUDA:
            total_us += event.time_range.elapsed_us()
    return total_us / 1000


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', required=True)
    p.add_argument('--out', default='speed_estimate.json')
    p.add_argument('--gpu-factors', default='3,4,5')
    p.add_argument('--cpu-factor', type=float, default=1.5)
    a, extra = p.parse_known_args()
    EXTRA_ARGS[:] = extra
    if not torch.cuda.is_available():
        sys.exit('Нужна CUDA: оценка строится от замера на твоей видеокарте.')

    out = Path(a.out).resolve()
    csv_path = out.with_suffix('.csv')

    # 1) общее время, без вмешательства
    bench = run(a.input, csv_path, out.with_name(out.stem + '_benchmark.json'))

    # 2) чистое время вычислений на видеокарте
    from torch.profiler import ProfilerActivity, profile
    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
    state.update(started=False, profiler=prof)
    # Без чтения в фоне (профайлер запускается в главном потоке) и без CUDA graphs: профайлер на них
    # зависает. Вход остаётся тем же квадратным, поэтому ядра видеокарты те же, что и в графах.
    original_capture = infer.DirectYolo._capture
    infer.DirectYolo._capture = lambda self, tensor: None
    try:
        run(a.input, csv_path, out.with_name(out.stem + '_profiled.json'), prefetch=False)
    finally:
        infer.DirectYolo._capture = original_capture
    torch.cuda.synchronize()
    prof.stop()
    images = bench['images']
    gpu_mean = kernel_ms(prof) / images if images else 0.0

    if gpu_mean == 0:
        print('ВНИМАНИЕ: профайлер не увидел работы видеокарты, оценка недостоверна.', file=sys.stderr)
    total_mean = bench['mean_ms']
    cpu_mean = max(total_mean - gpu_mean, 0.0)
    p95_ratio = bench['p95_ms'] / total_mean if total_mean else 1.0
    result = {
        'gpu': torch.cuda.get_device_name(),
        'cpu': cpu_name(),
        'images': images,
        'measured_mean_ms': round(total_mean, 1),
        'measured_p95_ms': round(bench['p95_ms'], 1),
        'gpu_kernels_mean_ms': round(gpu_mean, 1),
        'cpu_side_mean_ms': round(cpu_mean, 1),
        'note': 'Estimate = gpu_kernels * k_gpu + cpu_side * k_cpu, no CPU/GPU overlap assumed '
                '(upper bound). This is an estimate, not a GTX 1050 Ti measurement.',
        'cpu_factor': a.cpu_factor,
        'extra_args': extra,
        'estimate_gtx1050ti': {},
    }
    for k in (float(x) for x in a.gpu_factors.split(',')):
        mean = gpu_mean * k + cpu_mean * a.cpu_factor
        result['estimate_gtx1050ti'][f'k_gpu={k:g}'] = {
            'mean_ms': round(mean, 1), 'p95_ms_approx': round(mean * p95_ratio, 1)}
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
