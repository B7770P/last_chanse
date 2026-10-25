"""Local checks; this is NOT the missing organizer's validator."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from .common import read_meta, read_image, label_path, numbers, TYPES, valid_number, number_format_hint


def validate(root):
    root = Path(root).resolve()
    errors, warnings = [], []
    rows = read_meta(root)
    grouped, counts, unique = {}, Counter(), {t: set() for t in TYPES}
    for index, row in enumerate(rows, 2):
        try:
            path = (root / row['image']).resolve()
            if not path.is_relative_to(root / 'images') or path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                raise ValueError('image must point inside images/ to a JPG or PNG')
            if row['plate_type'] not in TYPES:
                raise ValueError('unknown plate_type')
            if row['is_vehicle'] not in ('0', '1') or row['is_synthetic'] not in ('0', '1'):
                raise ValueError('flags must be 0 or 1')
            if not valid_number(row['plate_num'], row['plate_type']):
                raise ValueError('plate_num: ' + number_format_hint(row['plate_type']))
            if not row['source'].strip() or not row['license'].strip():
                raise ValueError('source and license are required')
            if row['license'].upper() in ('UNKNOWN', 'TODO', 'UNSPECIFIED'):
                raise ValueError('placeholder license is not permitted')
            actual_synthetic = Path(row['image']).parts[1] == 'synthetic'
            if actual_synthetic != bool(int(row['is_synthetic'])):
                raise ValueError('is_synthetic disagrees with image directory')
            h, w = read_image(path).shape[:2]
            x, y, bw, bh = numbers(row['bbox'], 4)
            q = numbers(row['quad'], 8).reshape(4, 2)
            if not (bw > 0 and bh > 0 and x >= -.01 and y >= -.01 and x + bw <= w + .01 and y + bh <= h + .01):
                raise ValueError('bbox outside image or empty')
            if (q < -.01).any() or (q[:, 0] > w + .01).any() or (q[:, 1] > h + .01).any():
                raise ValueError('quad outside image')
            if not cv2.isContourConvex(q) or cv2.contourArea(q, oriented=True) <= 1:
                raise ValueError('quad must be convex and clockwise in image coordinates')
            if np.linalg.norm(q[0] - q[np.argmin(q.sum(axis=1))]) > max(bw, bh) * .3:
                warnings.append(f'Line {index}: check top-left corner order')
            if np.any(q.min(0) < np.array([x, y]) - .1) or np.any(q.max(0) > np.array([x + bw, y + bh]) + .1):
                raise ValueError('bbox does not enclose quad')
            grouped.setdefault(row['image'], []).append(row)
            origin = 'synthetic' if actual_synthetic else 'real'
            counts[f'{origin}_{row["plate_type"]}_plates'] += 1
            if origin == 'real':
                unique[row['plate_type']].add(row['plate_num'])
        except (ValueError, OSError, IndexError) as exc:
            errors.append(f'Line {index}: {exc}')
    images = sorted(p for p in (root / 'images').rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    seen = {}
    for path in images:
        image = path.relative_to(root).as_posix()
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                warnings.append(f'Exact duplicate: {image} and {seen[digest]}')
            seen[digest] = image
            h, w = read_image(path).shape[:2]
            lp = label_path(root, image)
            if not lp.exists():
                raise ValueError('missing labels TXT, including for negative frames')
            lines = [s.split() for s in lp.read_text(encoding='utf-8').splitlines() if s.strip()]
            expected = grouped.get(image, [])
            if len(lines) != len(expected):
                raise ValueError('TXT row count differs from meta.csv')
            for tokens, row in zip(lines, expected):
                if len(tokens) != 14:
                    raise ValueError('TXT needs 14 tokens: class, bbox(4), quad(8), vehicle')
                vals = np.asarray(list(map(float, tokens[1:13])))
                x, y, bw, bh = numbers(row['bbox'], 4)
                target = np.array([(x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h,
                                   *(numbers(row['quad'], 8).reshape(4, 2) / [w, h]).ravel()])
                if not np.isfinite(vals).all() or not np.allclose(vals, target, atol=2e-6):
                    raise ValueError('TXT coordinates differ from meta.csv')
                if int(tokens[0]) != TYPES.index(row['plate_type']) or tokens[-1] != row['is_vehicle']:
                    raise ValueError('TXT class/vehicle differs from meta.csv')
            counts['synthetic_images' if Path(image).parts[1] == 'synthetic' else 'real_images'] += 1
        except (ValueError, OSError) as exc:
            errors.append(f'{image}: {exc}')
    minima = {'type1a': (150, 50), 'type1b': (300, 100), 'other': (50, 0)}
    for t, (num_images, num_unique) in minima.items():
        actual = len({r['image'] for r in rows if r['plate_type'] == t and r['is_synthetic'] == '0'})
        counts[f'real_{t}_images'] = actual
        counts[f'real_{t}_unique_numbers'] = len(unique[t])
        if actual < num_images or len(unique[t]) < num_unique:
            warnings.append(f'Real {t}: {actual}/{num_images} images, {len(unique[t])}/{num_unique} unique numbers')
    if counts['synthetic_images'] < 5000:
        warnings.append(f'Synthetic images: {counts["synthetic_images"]}/5000 recommended')
    warnings.append('Legal rights, faces, image observability and near-duplicates require human review. Organizer validation still required.')
    return {'validator': 'local-v2 (type-aware masks; not organizer validator)', 'valid_structure': not errors,
            'statistics': dict(counts), 'errors': errors, 'warnings': warnings}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--report', default='validation_report.json')
    a = p.parse_args()
    report = validate(a.data)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['valid_structure'] else 1)


if __name__ == '__main__':
    main()
