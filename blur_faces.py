r"""Размывает лица в датасете и пишет копию в новую папку. Исходный датасет не меняется.

    python blur_faces.py --data dataset_composite5000 --out dataset_composite5000_blurred
    python blur_faces.py --data dataset_composite5000 --out dataset_blur_preview --limit 50

Лица ищутся каскадами Хаара из OpenCV (фас и профиль) из assets/face_cascades/,
интернет не нужен. В OpenCV 5 класс CascadeClassifier есть только в contrib-сборке,
поэтому запускай в отдельном окружении, чтобы не трогать окружение решения:

    py -3.9 -m venv .venv_blur
    .venv_blur\Scripts\python -m pip install opencv-contrib-python==5.0.0.93 numpy==2.0.2
    .venv_blur\Scripts\python blur_faces.py --data dataset_composite5000 --out dataset_composite5000_blurred

Номера не трогаются: найденное лицо, которое пересекается с рамкой
номера из labels/, пропускается. Каскады пропускают часть лиц и иногда срабатывают
на фон, поэтому после запуска просмотри face_blur_preview/ в новой папке и при необходимости добавь рамки
вручную в manual_faces.json ({"images/real/x.jpg": [[x, y, w, h], ...]}) и запусти снова.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def plate_boxes(label_path, w, h):
    boxes = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def overlaps(face, plates):
    x, y, fw, fh = face
    for x1, y1, x2, y2 in plates:
        if x < x2 and x + fw > x1 and y < y2 and y + fh > y1:
            return True
    return False


def detect_faces(gray, cascades):
    h, w = gray.shape
    min_side = max(24, min(h, w) // 40)
    found = []
    for cascade, flip in cascades:
        img = cv2.flip(gray, 1) if flip else gray
        for (x, y, fw, fh) in cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=6,
                                                       minSize=(min_side, min_side)):
            if flip:
                x = w - x - fw
            found.append((int(x), int(y), int(fw), int(fh)))
    return found


def blur(img, box):
    x, y, fw, fh = box
    pad = int(0.25 * max(fw, fh))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(img.shape[1], x + fw + pad), min(img.shape[0], y + fh + pad)
    roi = img[y1:y2, x1:x2]
    k = max(15, (max(x2 - x1, y2 - y1) // 3) | 1)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def imread(path):
    data = np.fromfile(str(path), dtype=np.uint8)  # кириллица в путях Windows
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path, img):
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if path.suffix.lower() in ('.jpg', '.jpeg') else []
    ok, buf = cv2.imencode(path.suffix, img, params)
    if not ok:
        raise RuntimeError(f'не удалось закодировать {path}')
    buf.tofile(str(path))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True, help='новая папка')
    p.add_argument('--manual', default='manual_faces.json', help='ручные рамки лиц (необязательно)')
    p.add_argument('--limit', type=int, default=0, help='обработать только N изображений (для пробы)')
    a = p.parse_args()

    src, out = Path(a.data), Path(a.out)
    if out.exists():
        raise SystemExit(f'Папка уже существует, выбери другую: {out}')
    manual = json.loads(Path(a.manual).read_text(encoding='utf-8')) if Path(a.manual).is_file() else {}

    # окружение проверяем до копирования датасета (больше 1 ГБ)
    if not hasattr(cv2, 'CascadeClassifier'):
        raise SystemExit('В этой сборке OpenCV нет CascadeClassifier. '
                         'Установи opencv-contrib-python в отдельное окружение (см. начало файла).')
    data_dir = Path(__file__).resolve().parent / 'assets' / 'face_cascades'
    cascades = []
    for name, flip in [('haarcascade_frontalface_default.xml', False),
                       ('haarcascade_profileface.xml', False), ('haarcascade_profileface.xml', True)]:
        cascade = cv2.CascadeClassifier(str(data_dir / name))
        if cascade.empty():
            raise SystemExit(f'Не загрузился каскад: {data_dir / name}')
        cascades.append((cascade, flip))

    print(f'Копирую {src} -> {out} ...')
    shutil.copytree(src, out)
    preview = out / 'face_blur_preview'
    preview.mkdir()

    images = sorted(f for part in ('real', 'synthetic') for f in (out / 'images' / part).rglob('*')
                    if f.suffix.lower() in EXTS)
    if a.limit:
        images = images[:a.limit]
    log, total = {}, 0
    for i, path in enumerate(images, 1):
        rel = path.relative_to(out).as_posix()
        img = imread(path)
        if img is None:
            print(f'  не читается: {rel}')
            continue
        h, w = img.shape[:2]
        label = out / 'labels' / path.parent.name / (path.stem + '.txt')
        plates = plate_boxes(label, w, h)
        faces = [f for f in detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cascades)
                 if not overlaps(f, plates)]
        faces += [tuple(b) for b in manual.get(rel, [])]
        if faces:
            for f in faces:
                blur(img, f)
            imwrite(path, img)
            log[rel] = [list(map(int, f)) for f in faces]
            total += len(faces)
            if path.parent.name == 'real':
                small = cv2.resize(img, (min(800, w), int(h * min(800, w) / w)))
                cv2.imencode('.jpg', small)[1].tofile(str(preview / (path.stem + '.jpg')))
        if i % 500 == 0:
            print(f'  {i}/{len(images)}')

    (out / 'face_blur_log.json').write_text(json.dumps(
        {'method': 'OpenCV Haar cascades (frontal + profile), Gaussian blur, plates excluded; '
                   'manual boxes from ' + a.manual if manual else
                   'OpenCV Haar cascades (frontal + profile), Gaussian blur, plates excluded',
         'images_processed': len(images), 'images_blurred': len(log), 'faces': total,
         'boxes': log}, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'Готово: обработано {len(images)}, размыто лиц {total} на {len(log)} изображениях.')
    print(f'Проверь глазами: {preview}')


if __name__ == '__main__':
    main()
