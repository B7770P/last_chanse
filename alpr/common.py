from pathlib import Path
import csv
import re
import cv2
import numpy as np

TYPES = ('type1', 'type1a', 'type1b', 'other')
LETTERS = 'ABEKMHOPCTYX'
ALPHABET = '0123456789' + LETTERS
FIELDS = ('image', 'plate_num', 'plate_type', 'bbox', 'quad', 'is_vehicle',
          'is_synthetic', 'source', 'license', 'conditions')
PATTERN = re.compile(r'^[ABEKMHOPCTYX#][0-9#]{3}[ABEKMHOPCTYX#]{2}[0-9#]{2,3}$')
YELLOW_PATTERN = re.compile(r'^[ABEKMHOPCTYX#]{2}[0-9#]{5}$')


def valid_number(text, kind='type1'):
    """Validate a transcription by plate type; # marks one unreadable glyph."""
    if kind == 'other':
        return True
    if kind == 'type1b':
        return YELLOW_PATTERN.fullmatch(text) is not None
    if kind in ('type1', 'type1a'):
        return PATTERN.fullmatch(text) is not None
    return False


def number_format_hint(kind):
    if kind == 'type1b':
        return 'AA12377 (2 letters, 3 digits, 2 region digits); # for unreadable positions; no spaces'
    if kind in ('type1', 'type1a'):
        return 'A123BC77 / A123BC777; # for unreadable positions; no spaces'
    return 'Free text; blank allowed for other'


def read_image(path, max_side=0):
    """imdecode/fromfile also handles Cyrillic paths on Windows.

    max_side > 0: large JPEGs are decoded at 1/2, 1/4 or 1/8 size (faster), keeping the long side >= max_side.
    """
    flags = cv2.IMREAD_COLOR
    if max_side:
        from PIL import Image
        with Image.open(path) as probe:
            side = max(probe.size)
        for factor, flag in ((8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4), (2, cv2.IMREAD_REDUCED_COLOR_2)):
            if side / factor >= max_side:
                flags = flag
                break
    im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if im is None:
        raise ValueError(f'Cannot read image: {path}')
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def write_image(path, image, quality=88):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if path.suffix.lower() in ('.jpg', '.jpeg') else []
    ok, data = cv2.imencode(path.suffix, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), params)
    if not ok:
        raise ValueError(f'Cannot encode {path}')
    data.tofile(str(path))


def read_meta(root):
    with (Path(root) / 'meta.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f'meta.csv must have columns: {FIELDS}')
        return list(reader)


def write_meta(root, rows):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'meta.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter=';')
        writer.writeheader()
        writer.writerows(rows)


def numbers(text, count):
    result = np.asarray([float(x) for x in text.split(',')], np.float32)
    if result.size != count or not np.isfinite(result).all():
        raise ValueError(f'Expected {count} finite numbers: {text!r}')
    return result


def label_path(root, image):
    return Path(root) / 'labels' / Path(image).relative_to('images').with_suffix('.txt')


def write_labels(root, rows, empty_images=()):
    grouped = {x: [] for x in empty_images}
    for row in rows:
        grouped.setdefault(row['image'], []).append(row)
    for image, plates in grouped.items():
        h, w = read_image(Path(root) / image).shape[:2]
        lines = []
        for row in plates:
            x, y, bw, bh = numbers(row['bbox'], 4)
            q = numbers(row['quad'], 8).reshape(4, 2) / [w, h]
            vals = [(x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h, *q.ravel()]
            lines.append(str(TYPES.index(row['plate_type'])) + ' ' +
                         ' '.join(f'{v:.7f}' for v in vals) + ' ' + row['is_vehicle'])
        out = label_path(root, image)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def letterbox(im, size):
    h, w = im.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    sx, sy = nw / w, nh / h
    px, py = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[py:py + nh, px:px + nw] = cv2.resize(im, (nw, nh))
    return out, (sx, sy, px, py)


def rectify(im, quad, kind):
    q = np.asarray(quad, np.float32).reshape(4, 2)
    if not cv2.isContourConvex(q) or abs(cv2.contourArea(q)) < 4:
        raise ValueError('Degenerate or non-convex plate quadrilateral')
    width, height = (290, 170) if kind == 'type1a' else (520, 112)
    dest = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    transform = cv2.getPerspectiveTransform(q, dest)
    plate = cv2.warpPerspective(im, transform, (width, height), borderValue=(180, 180, 180))
    # Layout follows the competition brief. Remove frame, RUS and flag area.
    if kind == 'type1a':
        a = cv2.resize(plate[8:82, 9:281], (160, 48))
        b = cv2.resize(plate[87:153, 9:281], (160, 48))
        strip = np.concatenate([a, b], axis=1)
    else:
        strip = cv2.resize(plate[6:88, 8:510], (320, 48))
    return cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)


def iou(a, b):
    # Boxes are x1,y1,x2,y2, not x,y,w,h.
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(area - inter, 1e-8)


def format_number(chars, confidences, threshold=.65, kind='type1'):
    lengths = (7,) if kind == 'type1b' else (8, 9)
    if kind not in TYPES[:3] or len(chars) not in lengths or len(chars) != len(confidences):
        return None
    letter_positions = (0, 1) if kind == 'type1b' else (0, 4, 5)
    out = []
    for i, (char, confidence) in enumerate(zip(chars, confidences)):
        allowed = LETTERS if i in letter_positions else '0123456789'
        out.append(char if char in allowed and confidence >= threshold else '#')
    return ''.join(out)
