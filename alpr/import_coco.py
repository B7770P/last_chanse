"""Import Roboflow COCO as geometry-only training data, without inventing OCR/types."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import zipfile

import cv2
import numpy as np
from .common import read_image

SOURCE = 'https://universe.roboflow.com/ru-anrp/russian-license-plates-detector/dataset/2'
PARTS = {'train': 'train', 'valid': 'val', 'val': 'val', 'test': 'test'}


class CocoSource:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.archive = zipfile.ZipFile(self.path) if self.path.is_file() else None
        if self.archive:
            items = [x for x in self.archive.infolist() if not x.is_dir()]
            if len({x.filename for x in items}) != len(items):
                raise ValueError('Duplicate ZIP member names')
            if sum(x.file_size for x in items) > 20_000_000_000:
                raise ValueError('Archive exceeds 20 GB uncompressed')
            self.names = [x.filename for x in items]
        elif self.path.is_dir():
            self.names = [p.relative_to(self.path).as_posix() for p in self.path.rglob('*') if p.is_file()]
        else:
            raise ValueError('--input must be a COCO ZIP or directory')
        for name in self.names:
            self.safe_name(name)

    @staticmethod
    def safe_name(name):
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
            raise ValueError(f'Unsafe archive or image path: {name}')
        return p

    def read(self, name):
        self.safe_name(name)
        if self.archive:
            info = self.archive.getinfo(name)
            if info.file_size > 250_000_000:
                raise ValueError(f'Oversized member: {name}')
            return self.archive.read(name)
        path = (self.path / name).resolve()
        if not path.is_relative_to(self.path):
            raise ValueError(f'Image escapes input directory: {name}')
        return path.read_bytes()

    def close(self):
        if self.archive:
            self.archive.close()


def plate_geometry(annotation, width, height):
    """Return verified bbox and optional quad. Missing polygons never become fake corners."""
    box = np.asarray(annotation.get('bbox', []), dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or min(box[2:]) <= 0:
        raise ValueError('Invalid or empty COCO bbox')
    lo, hi = box[:2], box[:2] + box[2:]
    if np.any(lo < -1) or np.any(hi > [width + 1, height + 1]):
        raise ValueError('COCO bbox is outside image by more than one pixel')
    lo, hi = np.maximum(lo, 0), np.minimum(hi, [width, height])
    if np.any(hi <= lo):
        raise ValueError('Clipped bbox is empty')
    quad, method, clipped = None, 'bbox_only', bool(np.any(box[:2] != lo) or np.any(box[:2] + box[2:] != hi))
    polygons = annotation.get('segmentation', [])
    if not isinstance(polygons, list):
        raise ValueError('RLE segmentation is unsupported; export polygon COCO')
    if len(polygons) > 1:
        method = 'multiple_polygons_bbox_only'
    elif polygons:
        flat = np.asarray(polygons[0], dtype=float)
        if flat.ndim != 1 or flat.size < 6 or flat.size % 2 or not np.isfinite(flat).all():
            raise ValueError('Invalid polygon')
        points = flat.reshape(-1, 2)
        if (points < -1).any() or (points[:, 0] > width + 1).any() or (points[:, 1] > height + 1).any():
            raise ValueError('Polygon outside image by more than one pixel')
        bounded = np.clip(points, [0, 0], [width, height])
        clipped = clipped or bool(np.any(points != bounded))
        points = np.unique(bounded, axis=0).astype(np.float32)
        hull = cv2.convexHull(points).reshape(-1, 2)
        if len(hull) < 3 or cv2.contourArea(hull) <= 1:
            raise ValueError('Degenerate polygon')
        lo, hi = points.min(0), points.max(0)
        if len(hull) == 4:
            quad, method = hull, 'polygon_four_corners' if len(points) == 4 else 'polygon_convex_hull'
        else:
            approx = cv2.approxPolyDP(hull, .02 * cv2.arcLength(hull, True), True).reshape(-1, 2)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad, method = approx, 'polygon_approximated'
            else:
                method = 'polygon_bbox_only'
    if quad is not None:
        center = quad.mean(0)
        quad = quad[np.argsort(np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0]))]
        quad = np.roll(quad, -int(np.argmin(quad.sum(1))), axis=0)
        if cv2.contourArea(quad, oriented=True) <= 1 or not cv2.isContourConvex(quad):
            raise ValueError('Invalid ordered quad')
    return [float(v) for v in np.r_[lo, hi - lo]], quad.tolist() if quad is not None else None, method, clipped


def grouped_split(records):
    """Keep byte/pixel-identical images and source-name hints in the same partition."""
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i, record in enumerate(records):
        base = record['original_filename'].split('.rf.')[0]
        hint = re.match(r'^([ABEKMHOPCTYX][0-9]{3}[ABEKMHOPCTYX]{2})-', base)
        keys = [('sha256', record['sha256']), ('pixels', record['pixel_sha256']), ('source_name', base)]
        if hint:
            # A conservative grouping hint, never an OCR training label.
            keys.append(('plate_prefix_hint', hint.group(1)))
        for key in keys:
            if key in owners:
                parent[find(i)] = find(owners[key])
            else:
                owners[key] = i
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(i)
    priority = {'train': 0, 'val': 1, 'test': 2}
    result = {'train': [], 'val': [], 'test': []}
    moved = []
    for indices in groups.values():
        part = max((records[i]['original_split'] for i in indices), key=priority.get)
        for i in indices:
            record = records[i]
            result[part].append(record['image'])
            if part != record['original_split']:
                moved.append({'image': record['image'], 'from': record['original_split'], 'to': part})
    for images in result.values():
        images.sort()
    result.update(groups=len(groups), note='Grouped by byte/pixel duplicates, source filename and conservative plate-prefix hints. Unknown plate identities/near-duplicates still need review.')
    return result, moved


def import_coco(source_path, output, source_url=SOURCE, attribution='RU ANRP'):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Output already exists; choose a new directory to preserve existing data')
    output.parent.mkdir(parents=True, exist_ok=True)
    source = CocoSource(source_path)
    stage = Path(tempfile.mkdtemp(prefix=output.name + '_import_', dir=output.parent))
    report = {'schema': 'alpr_geometry_v1', 'source_url': source_url, 'attribution': attribution,
              'source_image_count': 0, 'source_annotation_count': 0, 'excluded_images': [],
              'duplicate_annotations_removed': 0, 'geometry_clipped': 0,
              'geometry_methods': {}, 'ocr_labels_available': False,
              'plate_types_available': False, 'vehicle_labels_available': False}
    records, methods = [], Counter()
    try:
        jsons = sorted(name for name in source.names if name.endswith('_annotations.coco.json'))
        if not jsons:
            raise ValueError('No _annotations.coco.json found under train/valid/test')
        for name in jsons:
            folder = PurePosixPath(name).parent
            if folder.name not in PARTS:
                raise ValueError(f'Unknown split folder: {folder}')
            part = PARTS[folder.name]
            coco = json.loads(source.read(name))
            if any(a.get('iscrowd', 0) for a in coco['annotations']):
                raise ValueError('Crowd annotations need separate handling')
            licenses = {x['id']: x for x in coco.get('licenses', [])}
            categories = {x['id']: x['name'] for x in coco['categories']}
            annotations = defaultdict(list)
            ids = [x['id'] for x in coco['images']]
            id_set = set(ids)
            if len(set(ids)) != len(ids):
                raise ValueError('Duplicate COCO image IDs')
            for a in coco['annotations']:
                if a['image_id'] not in id_set:
                    raise ValueError('Annotation refers to missing image ID')
                if categories.get(a['category_id']) not in ('license-plate', 'license-plates'):
                    raise ValueError('This importer expects generic license-plate categories only')
                annotations[a['image_id']].append(a)
            report['source_image_count'] += len(coco['images'])
            report['source_annotation_count'] += len(coco['annotations'])
            raw_dest = stage / 'source_files' / folder.name
            raw_dest.mkdir(parents=True, exist_ok=True)
            (raw_dest / '_annotations.coco.json').write_bytes(source.read(name))
            for image_info in coco['images']:
                original = image_info['file_name']
                source.safe_name(original)
                original_path = (folder / original).as_posix()
                lic = licenses.get(image_info.get('license'), {})
                if lic.get('name') != 'CC BY 4.0' or 'creativecommons.org/licenses/by/4.0' not in lic.get('url', ''):
                    raise ValueError(f'Unverified/different image license: {original_path}')
                extension = PurePosixPath(original).suffix.lower()
                if extension not in ('.jpg', '.jpeg', '.png'):
                    raise ValueError(f'Unsupported image extension: {original}')
                new_name = f'rf_{folder.name}_{len(records):05d}{extension}'
                dest = stage / 'images' / 'real' / new_name
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    content = source.read(original_path)
                    dest.write_bytes(content)
                    image = read_image(dest)
                    height, width = image.shape[:2]
                    if (width, height) != (image_info['width'], image_info['height']):
                        raise ValueError('COCO dimensions differ from decoded image')
                    objects, unique = [], set()
                    for a in annotations[image_info['id']]:
                        bbox, quad, method, clipped = plate_geometry(a, width, height)
                        key = json.dumps([bbox, quad])
                        if key in unique:
                            report['duplicate_annotations_removed'] += 1
                            continue
                        unique.add(key)
                        objects.append({'annotation_id': a['id'], 'bbox': bbox, 'quad': quad,
                                        'quad_method': method, 'plate_num': None, 'plate_type': None,
                                        'is_vehicle': None, 'original_bbox': a['bbox'],
                                        'geometry_clipped': clipped,
                                        'original_segmentation': a.get('segmentation', [])})
                    records.append({'image': dest.relative_to(stage).as_posix(), 'width': width, 'height': height,
                                    'original_filename': original, 'original_path': original_path,
                                    'original_split': part, 'sha256': hashlib.sha256(content).hexdigest(),
                                    'pixel_sha256': hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest(),
                                    'source': source_url, 'license': lic['name'], 'license_url': lic['url'],
                                    'is_synthetic': False, 'objects': objects})
                    methods.update(x['quad_method'] for x in objects)
                    report['geometry_clipped'] += sum(x['geometry_clipped'] for x in objects)
                except (ValueError, OSError, KeyError) as exc:
                    dest.unlink(missing_ok=True)
                    report['excluded_images'].append({'image': original_path, 'reason': str(exc)})
                if len(records) and len(records) % 500 == 0:
                    print(f'Imported {len(records)} images', flush=True)
        if not records:
            raise ValueError('No valid images imported')
        split, moved = grouped_split(records)
        if any(not split[k] for k in ('train', 'val', 'test')):
            raise ValueError('An imported split is empty')
        report.update(imported_images=len(records), imported_annotations=sum(len(x['objects']) for x in records),
                      images_without_annotations=sum(not x['objects'] for x in records),
                      geometry_methods=dict(methods), split_sizes={k: len(split[k]) for k in ('train', 'val', 'test')},
                      split_changes=moved, identity_groups=split['groups'])
        manifest = {'schema': 'alpr_geometry_v1', 'source': source_url, 'attribution': attribution,
                    'source_annotation_exhaustiveness': 'not independently verified', 'images': records}
        for filename, value in [('geometry.json', manifest), ('split.json', split), ('import_report.json', report)]:
            (stage / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        for name in source.names:
            if PurePosixPath(name).name in ('README.dataset.txt', 'README.roboflow.txt'):
                (stage / 'source_files' / PurePosixPath(name).name).write_bytes(source.read(name))
        (stage / 'ATTRIBUTION.txt').write_text(
            f'Russian License Plates Detector, v2 accurate polygon annotation\nDataset creator: {attribution}\n'
            f'Source: {source_url}\nLicense: CC BY 4.0\nhttps://creativecommons.org/licenses/by/4.0/\n'
            'Changes: image bytes retained, filenames shortened; annotation coordinates reordered; '
            'convex hull/quad approximation used when flagged; <=1 px boundary errors clipped; '
            'invalid frames excluded; source grouping applied across splits. '
            'Per-photo creators are not listed in the supplied archive.\n', encoding='utf-8')
        (stage / 'README.txt').write_text(
            'Geometry-only Roboflow import. Not a complete competition dataset.\n'
            'geometry.json retains images, bbox, optional quad, sources and licenses. '
            'OCR text, competition types and vehicle flags are null, not inferred from names.\n'
            'For bbox-only objects, corner loss is disabled. Their corners have not been invented.\n'
            'Use alpr.train --task detector --generic-plates --data THIS_FOLDER --split THIS_FOLDER/split.json.\n'
            'Then fine-tune without --generic-plates on fully annotated competition data. '
            'A generic checkpoint cannot be used by alpr.infer before such fine-tuning.\n'
            'Grouped splits prevent exact/pixel duplicate and known source-name hint leakage. '
            'Unidentified matching plates and near-duplicate scenes still need review.\n'
            'Missing annotations in source frames and recognisable faces still need visual review. '
            'Do not include these raw frames in the final submission without that review.\n', encoding='utf-8')
        stage.rename(output)
        return report
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        source.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='Roboflow COCO ZIP or extracted directory')
    p.add_argument('--out', required=True, help='New geometry-only data directory')
    p.add_argument('--source', default=SOURCE)
    p.add_argument('--attribution', default='RU ANRP')
    a = p.parse_args()
    report = import_coco(a.input, a.out, a.source, a.attribution)
    print(json.dumps({k: report[k] for k in ('imported_images', 'imported_annotations', 'geometry_methods', 'split_sizes')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
