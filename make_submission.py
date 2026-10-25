"""Собирает чистую папку для сдачи: код, итоговые веса, документы, лицензии.

Ничего в проекте не удаляет и не меняет: только копирует в новую папку.

    python make_submission.py                      # submission_<дата>/ без датасета
    python make_submission.py --with-dataset       # + dataset/ (606 реальных + 5000 синтетических)
    python make_submission.py --with-dataset --zip # + zip-архив рядом
    python make_submission.py --with-dataset --dataset dataset_composite5000_blurred --zip
"""
import argparse
import datetime as dt
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Итоговая конфигурация = run_final.ps1. Меняешь здесь — поменяй и там.
MODELS = {
    'models/yolo11n_real_20261024': ['weights/best.pt', 'args.yaml', 'MODEL_PROVENANCE.json',
                                     'TRAINING_CONFIG.json', 'ULTRALYTICS_LICENSE.txt'],
    'models/yolo11n_white_20261024': ['weights/best.pt', 'args.yaml', 'MODEL_PROVENANCE.json',
                                      'ULTRALYTICS_LICENSE.txt'],
    'models/ocr_crops_20261024': ['best.pt', 'history.json', 'MODEL_PROVENANCE.json'],
    'models/ocr_synth_20261025': ['best.pt', 'history.json', 'MODEL_PROVENANCE.json'],
    'models/ocr_synth2_20261025': ['best.pt', 'history.json', 'MODEL_PROVENANCE.json'],
    'models/ocr_type1_small_20261025': ['best.pt', 'history.json', 'MODEL_PROVENANCE.json'],
}
FILES = ['README.md', 'EXPLANATORY_NOTE.md', 'EXPLANATORY_NOTE.pdf', 'TECHNICAL_DETAILS.md', 'REPORT.md',
         'dataset_validation_report.json', 'LICENSE', 'requirements.txt', 'run_final.ps1', 'make_submission.py',
         'blur_faces.py', 'blur_people.py', 'estimate_speed.py', 'profile_stages.py', 'compare_fast_v5.py', 'eval_fin_test.py',
         'label_fin_test.py', 'fin_test_labels.csv', 'fin_test_report_v5.json', 'fin_test_speed_v5.json', 'fin_test_report_v6.json', 'fin_test_speed_v6.json',
         'fin_test_report_v7.json', 'fin_test_speed_v7.json', 'speed_estimate_v7.json',
         'fin_test_report_v8.json', 'fin_test_speed_v8.json', 'speed_estimate_v8.json',
         'analyze_val_errors.py', 'val_errors.json', 'val_errors_train.json', 'val_errors_train_v0.json',
         'val_errors_train_v5.json', 'val_errors_train_v7.json',
         'detection_variants.py', 'detection_variants.json', 'detection_variants_V0.json', 'detection_variants_V5.json',
         'detection_variants_V6_2048.json', 'detection_variants_V7.json',
         'diagnose_no_text.py', 'diagnose_no_text.json',
         'test22_labels.csv', 'test22_report_v8_sub.json', 'test22_pred_v8_sub.csv', 'test22_speed_v8_sub.json',
         'build_yolo_rf.py', 'detection_variants_A1.json', 'detection_variants_A1b.json', 'detection_variants_A2.json',
         'detection_variants_B1.json', 'detection_variants_B2.json', 'detection_variants_V6_cur.json',
         'detection_variants_V6_ocr2.json', 'detection_variants_V6_ocr3_c0.3.json', 'detection_variants_V6_ocr3_c0.4.json',
         'detection_variants_V6_ocr3_c0.5.json', 'detection_variants_V6_rfdet.json',
         'speed_estimate_v5.json', 'compare_fast.json', 'compare_fast_v5.json',
         'compare_ocr.py', 'compare_ocr.json', 'compare_ocr2.json',
         'sweep_thresholds.py', 'sweep_thresholds.json', 'sweep_thresholds2.json', 'sweep_thresholds3.json']
DIRS = ['alpr', 'assets', 'tests']
DATASET_DIR = 'dataset_composite5000'
SPLITS = ['splits/composite5000.json', 'splits/real606.json', 'splits/real606_stats.json']
SKIP = shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache',
                              'face_blur_preview')  # превью размытия только для ручной проверки


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def copy_file(rel, out, required=True):
    src = ROOT / rel
    if not src.is_file():
        if required:
            raise FileNotFoundError(f'нет обязательного файла: {rel}')
        print(f'  пропущен (нет файла): {rel}')
        return
    dst = out / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', help='новая папка (по умолчанию submission_<дата>)')
    p.add_argument('--with-dataset', action='store_true', help='положить датасет (--dataset) как dataset/')
    p.add_argument('--dataset', default=DATASET_DIR,
                   help='папка датасета (например, dataset_composite5000_blurred после blur_faces.py)')
    p.add_argument('--zip', action='store_true', help='дополнительно упаковать в zip')
    a = p.parse_args()

    out = Path(a.out) if a.out else ROOT / f'submission_{dt.date.today():%Y%m%d}'
    if out.exists():
        sys.exit(f'Папка уже существует, выбери другую через --out: {out}')
    out.mkdir(parents=True)
    print(f'Сборка в {out}')

    for rel in FILES:
        copy_file(rel, out)
    for rel in DIRS:
        shutil.copytree(ROOT / rel, out / rel, ignore=SKIP)
    for model_dir, names in MODELS.items():
        for name in names:
            copy_file(f'{model_dir}/{name}', out, required=name.endswith('.pt'))

    if a.with_dataset:
        print('  копирую датасет (может занять несколько минут)...')
        shutil.copytree(ROOT / a.dataset, out / 'dataset', ignore=SKIP)
        for rel in SPLITS:
            copy_file(rel, out, required=False)

    try:
        freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True,
                                text=True, check=True).stdout
        (out / 'requirements-lock.txt').write_text(
            f'# pip freeze, Python {sys.version.split()[0]}\n' + freeze,
            encoding='ascii', errors='replace')  # ASCII: pip -r on Windows reads ANSI codepage
    except Exception as e:  # окружение без pip — не критично
        print(f'  ВНИМАНИЕ: requirements-lock.txt не создан: {e}')

    files = sorted(f for f in out.rglob('*') if f.is_file())
    with open(out / 'SHA256SUMS.txt', 'w', encoding='utf-8') as f:
        for path in files:
            if a.with_dataset and path.is_relative_to(out / 'dataset'):
                continue  # тысячи картинок; хеши источников уже в provenance.json
            f.write(f'{sha256(path)}  {path.relative_to(out).as_posix()}\n')

    size = sum(f.stat().st_size for f in files) / 2**20
    print(f'Готово: {len(files)} файлов, {size:.1f} МБ')

    if a.zip:
        archive = shutil.make_archive(str(out), 'zip', root_dir=out.parent, base_dir=out.name)
        print(f'Архив: {archive}')


if __name__ == '__main__':
    main()
