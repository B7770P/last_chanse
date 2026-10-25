"""Reproducible procedural scenes. No web images, no pretrained generators."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .common import LETTERS, FIELDS, write_meta, write_labels, write_image, valid_number

FONT = Path(__file__).resolve().parents[1] / 'assets' / 'DejaVuSans-Bold.ttf'


def draw_text(im, text, box):
    x, y, w, h = map(int, box)
    font = ImageFont.truetype(str(FONT), 100)
    b = font.getbbox(text)
    glyph = Image.new('L', (max(1, b[2] - b[0]), max(1, b[3] - b[1])), 0)
    ImageDraw.Draw(glyph).text((-b[0], -b[1]), text, font=font, fill=255)
    glyph = glyph.resize((w, h), Image.Resampling.LANCZOS)
    im.paste((15, 15, 15), (x, y, x + w, y + h), glyph)


def random_number(rng, kind='type1'):
    letter = lambda: str(rng.choice(list(LETTERS)))
    if kind == 'type1b':
        return letter() + letter() + f'{rng.integers(0, 1000):03d}' + f'{rng.integers(1, 100):02d}'
    if kind not in ('type1', 'type1a'):
        raise ValueError('random_number expects type1, type1a or type1b')
    region = f'{rng.integers(1, 100):02d}' if rng.random() < .5 else str(rng.choice([1, 2, 7])) + f'{rng.integers(0, 100):02d}'
    return letter() + f'{rng.integers(0, 1000):03d}' + letter() + letter() + region


def render_plate(text, kind, rng):
    if not valid_number(text, kind):
        raise ValueError(f'Invalid {kind} number: {text!r}')
    square = kind == 'type1a'
    w, h = (290, 170) if square else (520, 112)
    color = (246, 197, 37) if kind == 'type1b' else (235, 235, 228)
    if kind == 'other':
        color = [(140, 24, 25), (22, 26, 25), (213, 219, 233)][int(rng.integers(3))]
    im = Image.new('RGB', (w, h), color)
    draw = ImageDraw.Draw(im)
    draw.rounded_rectangle((1, 1, w - 2, h - 2), radius=8, outline=(30, 30, 30), width=3)
    if square:
        for i, c in enumerate(text[:4]):
            draw_text(im, c, (15 + i * 67, 14, 54, 64))
        draw_text(im, text[4:6], (15, 95, 102, 53))
        draw_text(im, text[6:], (145, 95, 128, 50))
        draw_text(im, 'RUS', (152, 155, 47, 10))
        draw.rectangle((233, 155, 267, 158), fill='white')
        draw.rectangle((233, 159, 267, 162), fill=(30, 65, 190))
        draw.rectangle((233, 163, 267, 166), fill=(200, 30, 30))
    elif kind == 'other':
        draw_text(im, text, (18, 20, 475, 65))
    else:
        prefix = 5 if kind == 'type1b' else 6
        letter_positions = (0, 1) if kind == 'type1b' else (0, 4, 5)
        step, glyph_width = (73, 58) if kind == 'type1b' else (61, 48)
        for i, c in enumerate(text[:prefix]):
            is_letter = i in letter_positions
            draw_text(im, c, (15 + i * step, 20 if is_letter else 12, glyph_width, 61 if is_letter else 69))
        draw.line((385, 2, 385, 109), fill=(30, 30, 30), width=2)
        draw_text(im, text[prefix:], (397, 13, 110, 65))
        draw_text(im, 'RUS', (396, 89, 53, 12))
        draw.rectangle((462, 88, 499, 91), fill='white')
        draw.rectangle((462, 92, 499, 95), fill=(30, 65, 190))
        draw.rectangle((462, 96, 499, 99), fill=(200, 30, 30))
    # Simulated bolts and dirt are visual, not evidence of true occlusion labels.
    for x in (7, w - 8):
        draw.ellipse((x - 2, h // 2 - 2, x + 2, h // 2 + 2), fill=(90, 90, 90))
    return np.array(im)


def scene(rng, width=512, height=320):
    sky = np.array([rng.integers(100, 210), rng.integers(100, 210), rng.integers(100, 210)])
    arr = np.empty((height, width, 3), np.uint8)
    arr[:height // 2] = sky
    arr[height // 2:] = rng.integers(45, 110)
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    for _ in range(8):
        x = int(rng.integers(width))
        y = int(rng.integers(10, height // 2))
        d.rectangle((x, y, x + int(rng.integers(10, 80)), height // 2), fill=tuple(map(int, rng.integers(50, 190, 3))))
    return np.array(im)


def generate(root, count=5000, seed=42):
    root = Path(root)
    if (root / 'meta.csv').exists():
        raise ValueError('Choose a new output directory; refusing to overwrite an existing dataset')
    rng = np.random.default_rng(seed)
    rows, empty = [], []
    for i in range(count):
        im = scene(rng)
        h, w = im.shape[:2]
        name = f'images/synthetic/synth_{i:06d}.jpg'
        plates = []
        # Explicit negatives: 12% of frames have no plates.
        n = 0 if rng.random() < .12 else (2 if rng.random() < .2 else 1)
        for j in range(n):
            kind = str(rng.choice(['type1', 'type1a', 'type1b', 'other'], p=[.25, .30, .30, .15]))
            text = random_number(rng, kind) if kind != 'other' else str(rng.choice(['CD12345', 'AB1234', '1234AB', 'TR9999']))
            # Avoid overlap by assigning separate horizontal bands.
            band = w // max(n, 1)
            pw = float(rng.uniform(60, min(180, band - 24)))
            ph = pw * (170 / 290 if kind == 'type1a' else 112 / 520)
            cx = float(rng.uniform(j * band + pw / 2 + 8, (j + 1) * band - pw / 2 - 8))
            cy = float(rng.uniform(175, 263))
            vehicle = int(rng.random() > .15)
            canvas = Image.fromarray(im)
            d = ImageDraw.Draw(canvas)
            car_color = tuple(map(int, rng.integers(35, 205, 3)))
            if vehicle:
                cw = min(band - 8, max(110, pw * 1.4))
                d.rounded_rectangle((cx - cw / 2, cy - 70, cx + cw / 2, min(h - 5, cy + 40)), radius=14, fill=car_color, outline=(25, 25, 25), width=3)
                d.polygon([(cx - cw * .38, cy - 66), (cx + cw * .38, cy - 66), (cx + cw * .3, cy - 30), (cx - cw * .3, cy - 30)], fill=(43, 57, 69))
                for side in (-1, 1):
                    x = cx + side * cw * .38
                    d.rectangle((x - 12, cy - 20, x + 12, cy - 10), fill=(235, 224, 145))
                    d.rectangle((x - 9, cy + 35, x + 9, min(h - 1, cy + 50)), fill=(18, 18, 18))
            else:
                d.rectangle((cx - pw / 2 - 5, cy - ph / 2 - 12, cx + pw / 2 + 5, cy + ph / 2 + 12), fill=(90, 78, 64))
                d.line((cx, cy + ph / 2, cx, h - 1), fill=(40, 40, 40), width=5)
            im = np.array(canvas)
            q = np.float32([[cx - pw / 2, cy - ph / 2], [cx + pw / 2, cy - ph / 2],
                            [cx + pw / 2, cy + ph / 2], [cx - pw / 2, cy + ph / 2]])
            q += rng.uniform(-1, 1, (4, 2)).astype(np.float32) * [pw * .05, ph * .12]
            plate = render_plate(text, kind, rng)
            ah, aw = plate.shape[:2]
            transform = cv2.getPerspectiveTransform(np.float32([[0, 0], [aw - 1, 0], [aw - 1, ah - 1], [0, ah - 1]]), q)
            warped = cv2.warpPerspective(plate, transform, (w, h))
            mask = cv2.warpPerspective(np.full((ah, aw), 255, np.uint8), transform, (w, h)) / 255.
            im = (warped * mask[..., None] + im * (1 - mask[..., None])).astype(np.uint8)
            lo, hi = q.min(0), q.max(0)
            plates.append({'image': name, 'plate_num': text, 'plate_type': kind,
                           'bbox': ','.join(f'{v:.3f}' for v in [*lo, *(hi - lo)]),
                           'quad': ','.join(f'{v:.3f}' for v in q.ravel()), 'is_vehicle': str(vehicle),
                           'is_synthetic': '1', 'source': f'own_generator:v2:seed={seed}',
                           'license': 'CC-BY-4.0', 'conditions': 'day,angle'})
        conditions = ['day', 'angle']
        if rng.random() < .25:
            sigma = float(rng.uniform(.3, 1.1))
            im = cv2.GaussianBlur(im, (5, 5), sigma)
            conditions.append('blur')
        if rng.random() < .20:
            im = np.clip(im.astype(float) * rng.uniform(.35, .65), 0, 255).astype(np.uint8)
            conditions[0] = 'night'
        noise = rng.normal(0, rng.uniform(0, 5), im.shape)
        im = np.clip(im.astype(float) + noise, 0, 255).astype(np.uint8)
        for row in plates:
            row['conditions'] = ','.join(conditions)
        rows.extend(plates)
        if not plates:
            empty.append(name)
        write_image(root / name, im, quality=int(rng.integers(72, 92)))
        if (i + 1) % 500 == 0:
            print(f'Generated {i + 1}/{count}', flush=True)
    write_meta(root, rows)
    write_labels(root, rows, empty)
    for folder in ['images/real', 'labels/real', 'generator']:
        (root / folder).mkdir(parents=True, exist_ok=True)
    recipe = {'version': 2, 'count': count, 'seed': seed, 'numpy': np.__version__, 'opencv': cv2.__version__,
              'font_sha256': hashlib.sha256(FONT.read_bytes()).hexdigest(),
              'command': f'python -m alpr.generate --out dataset --count {count} --seed {seed}'}
    (root / 'generator' / 'recipe.json').write_text(json.dumps(recipe, indent=2), encoding='utf-8')
    # Make dataset/generator self-contained, including font and its license.
    generator_dir = root / 'generator'
    (generator_dir / 'alpr').mkdir(exist_ok=True)
    (generator_dir / 'assets').mkdir(exist_ok=True)
    for filename in ('__init__.py', 'common.py', 'generate.py'):
        shutil.copy2(Path(__file__).parent / filename, generator_dir / 'alpr' / filename)
    shutil.copy2(FONT, generator_dir / 'assets' / FONT.name)
    shutil.copy2(FONT.parent / 'FONT-LICENSE.txt', generator_dir / 'assets' / 'FONT-LICENSE.txt')
    (generator_dir / 'generate.py').write_text('from alpr.generate import main\nif __name__ == "__main__":\n    main()\n', encoding='utf-8')
    (root / 'generator' / 'requirements.txt').write_text('numpy==1.26.4\nPillow==11.0.0\nopencv-python-headless==4.10.0.84\n', encoding='utf-8')
    (root / 'README.md').write_text(
        '# Synthetic dataset\n\nProcedural vector-like vehicles, signs and backgrounds. No real photographs.\n'
        'Formats: type1/type1a A123BC77 or A123BC777; yellow type1b AA12377.\n'
        'Author: project participant; fill in your name before publication. Licence: CC BY 4.0.\n'
        'Plate font: bundled DejaVu, see ../assets/FONT-LICENSE.txt. Geometry and font approximate, not certified GOST.\n'
        'Scene and annotation are generated together, so this is not manual annotation.\n'
        'Negative frames have empty TXT files and no rows in meta.csv.\n'
        'Self-contained generator: python generator/generate.py --out NEW_DIRECTORY --count COUNT --seed SEED. Recipe in generator/recipe.json.\n'
        'Do not claim procedural pictures satisfy the real-image quotas. Inspect tiny/blurred plates and use # when unobservable.\n'
        'Label TXT: class cx cy w h qx1 qy1 qx2 qy2 qx3 qy3 qx4 qy4 is_vehicle; coordinates normalized.\n'
        'Order of corners: top-left, top-right, bottom-right, bottom-left; clockwise in image coordinates.\n', encoding='utf-8')
    (root / 'LICENSE').write_text('Creative Commons Attribution 4.0 International\nhttps://creativecommons.org/licenses/by/4.0/legalcode\nAttribution: project participant (fill in name before publication).\n', encoding='utf-8')
    print(f'Generated {count} images, {len(rows)} plates; real images: 0', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--count', type=int, default=5000)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.count < 1:
        p.error('--count must be positive')
    generate(a.out, a.count, a.seed)


if __name__ == '__main__':
    main()
