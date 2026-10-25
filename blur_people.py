r"""Размывает головы людей в датасете и пишет копию в новую папку. Исходный датасет не меняется.

    python blur_people.py --data dataset_composite5000 --out dataset_composite5000_bl2

Каскады Хаара (blur_faces.py) пропускали людей и размывали деревья, поэтому здесь людей ищет
YOLO11n, обученный на COCO (models/yolo11n_pretrained/yolo11n.pt, класс person), локально,
без интернета. Размывается верхняя часть рамки человека, где голова: так закрываются и мелкие лица,
которые детектор лиц не видит (водители за стеклом, прохожие вдали). Пиксели внутри рамок
номеров из labels/ после размытия возвращаются как были, номера не трогаются.

В face_blur_preview/ лежат уменьшенные реальные кадры, размытые области обведены красным
(рамки только в превью, в датасете их нет). Пропущенное можно добавить вручную в manual_faces.json
({"images/real/x.jpg": [[x, y, w, h], ...]}) и запустить заново.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
ROOT = Path(__file__).resolve().parent


def plate_boxes(label_path, w, h):
    boxes = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        boxes.append((int((cx - bw / 2) * w), int((cy - bh / 2) * h), int(np.ceil((cx + bw / 2) * w)), int(np.ceil((cy + bh / 2) * h))))
    return boxes


def head_region(x1, y1, x2, y2):
    """Upper part of a person box: about one head height, at least as tall as the box is wide."""
    w, h = x2 - x1, y2 - y1
    top = min(h, max(.3 * h, .8 * w))
    return int(x1), int(y1), int(w), int(top)


def blur(img, box):
    x, y, fw, fh = box
    pad = int(.15 * max(fw, fh))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(img.shape[1], x + fw + pad), min(img.shape[0], y + fh + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = img[y1:y2, x1:x2]
    k = max(15, (max(x2 - x1, y2 - y1) // 2) | 1)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return x1, y1, x2, y2


def imread(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)  # кириллица в путях Windows


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
    p.add_argument('--model', default=str(ROOT / 'models/yolo11n_pretrained/yolo11n.pt'))
    p.add_argument('--conf', type=float, default=.2)
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument('--manual', default='manual_faces.json')
    p.add_argument('--device', default='cuda')
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()

    import os
    os.environ['YOLO_OFFLINE'] = 'true'
    from ultralytics import YOLO
    model = YOLO(a.model)
    if model.names.get(0) != 'person':
        raise SystemExit(f'В {a.model} класс 0 не person: нужна модель COCO')

    src, out = Path(a.data), Path(a.out)
    if out.exists():
        raise SystemExit(f'Папка уже существует, выбери другую: {out}')
    manual = json.loads(Path(a.manual).read_text(encoding='utf-8')) if Path(a.manual).is_file() else {}
    print(f'Копирую {src} -> {out} ...')
    shutil.copytree(src, out, ignore=shutil.ignore_patterns('face_blur_preview', 'face_blur_log.json'))
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
        result = model.predict(img, imgsz=a.imgsz, conf=a.conf, classes=[0], device=a.device, verbose=False)[0]
        regions = [head_region(*row[:4]) for row in result.boxes.xyxy.cpu().numpy()]
        regions += [tuple(int(v) for v in b) for b in manual.get(rel, [])]
        if regions:
            original = img.copy()
            done = [r for r in (blur(img, r) for r in regions) if r]
            for x1, y1, x2, y2 in plate_boxes(out / 'labels' / path.parent.name / (path.stem + '.txt'), w, h):
                img[max(0, y1):y2, max(0, x1):x2] = original[max(0, y1):y2, max(0, x1):x2]  # номер не трогаем
            imwrite(path, img)
            log[rel] = [list(map(int, r)) for r in done]
            total += len(done)
            if path.parent.name == 'real':
                shown = img.copy()
                for x1, y1, x2, y2 in done:
                    cv2.rectangle(shown, (x1, y1), (x2, y2), (0, 0, 255), max(2, w // 400))
                scale = min(1., 1000 / w)
                small = cv2.resize(shown, (int(w * scale), int(h * scale)))
                cv2.imencode('.jpg', small)[1].tofile(str(preview / (path.stem + '.jpg')))
        if i % 500 == 0:
            print(f'  {i}/{len(images)}', flush=True)

    (out / 'face_blur_log.json').write_text(json.dumps(
        {'method': f'YOLO11n COCO person (conf {a.conf}, imgsz {a.imgsz}), blur of the upper (head) part of '
                   'each person box, plate boxes restored' + (f'; manual boxes from {a.manual}' if manual else ''),
         'images_processed': len(images), 'images_blurred': len(log), 'regions': total,
         'boxes_xyxy': log}, ensure_ascii=False, indent=1), encoding='utf-8')
    real = sum(1 for k in log if k.startswith('images/real/'))
    print(f'Готово: обработано {len(images)}, размыто областей {total} на {len(log)} кадрах (реальных {real}).')
    print(f'Проверь глазами: {preview}')


if __name__ == '__main__':
    main()
