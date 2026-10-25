import json
from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .common import read_meta, read_image, letterbox, numbers, rectify, TYPES, ALPHABET, valid_number


def gaussian(heat, cx, cy, radius):
    h, w = heat.shape
    radius = max(1, int(radius))
    x1, x2 = max(0, cx - radius), min(w, cx + radius + 1)
    y1, y2 = max(0, cy - radius), min(h, cy + radius + 1)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * max(radius / 3, .5) ** 2))
    heat[y1:y2, x1:x2] = np.maximum(heat[y1:y2, x1:x2], g)


class PlateDataset(Dataset):
    def __init__(self, root, split_file, part, task, size=512, augment=False):
        self.root, self.task, self.size, self.augment = Path(root), task, size, augment
        self.ocr_box_probability = 0.
        split = json.loads(Path(split_file).read_text(encoding='utf-8'))
        selected = set(split[part])
        grouped = {image: [] for image in selected}
        for row in read_meta(root):
            if row['image'] in grouped:
                grouped[row['image']].append(row)
        if task == 'detector':
            self.samples = sorted(grouped.items())
        else:
            for image in sorted(grouped):
                for row in grouped[image]:
                    if not valid_number(row['plate_num'], row['plate_type']):
                        raise ValueError(f'Invalid {row["plate_type"]} plate_num {row["plate_num"]!r} in {image}; '
                                         'correct the transcription/type or regenerate old synthetic type1b data')
            self.samples = [(im, r) for im in sorted(grouped) for r in grouped[im]
                            if r['plate_type'] != 'other' and '#' not in r['plate_num']]
        if not self.samples:
            raise ValueError(f'No {task} samples in {part}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, rows = self.samples[index]
        im = read_image(self.root / image)
        if self.task == 'ocr':
            quad = numbers(rows['quad'], 8)
            if self.augment and random.random() < self.ocr_box_probability:
                x,y,w,h = numbers(rows['bbox'], 4)
                # Train on rectangular detector-like crops as well as manually rectified crops.
                jitter = np.array([random.uniform(-.03,.03)*w,random.uniform(-.03,.03)*h])
                lo = np.maximum([x,y]+jitter,0)
                hi = np.minimum([x+w,y+h]+jitter,[im.shape[1],im.shape[0]])
                quad = np.array([lo,[hi[0],lo[1]],hi,[lo[0],hi[1]]])
            crop = rectify(im, quad, rows['plate_type'])
            if self.augment:
                crop = np.clip(crop.astype(float) * random.uniform(.7, 1.25) + random.uniform(-12, 12), 0, 255).astype(np.uint8)
                if random.random() < .2:
                    crop = cv2.GaussianBlur(crop, (3, 3), .5)
            x = torch.from_numpy(crop.copy()).float().unsqueeze(0) / 127.5 - 1
            target = torch.tensor([ALPHABET.index(c) + 1 for c in rows['plate_num']], dtype=torch.long)
            return x, target, rows['plate_num']
        im, (sx, sy, px, py) = letterbox(im, self.size)
        if self.augment:
            im = np.clip(im.astype(float) * random.uniform(.65, 1.25) + random.uniform(-12, 12), 0, 255).astype(np.uint8)
        grid = self.size // 8
        heat = np.zeros((4, grid, grid), np.float32)
        reg = np.zeros((13, grid, grid), np.float32)
        mask = np.zeros((grid, grid), np.float32)
        for row in rows:
            q = numbers(row['quad'], 8).reshape(4, 2) * [sx, sy] + [px, py]
            lo, hi = q.min(0) / 8, q.max(0) / 8
            center, wh = (lo + hi) / 2, hi - lo
            ix, iy = np.floor(center).astype(int)
            if not (0 <= ix < grid and 0 <= iy < grid):
                continue
            if mask[iy, ix]:
                raise ValueError(f'Two plate centres in one stride-8 cell: {image}; increase --size')
            gaussian(heat[TYPES.index(row['plate_type'])], ix, iy, min(wh) / 2)
            reg[:2, iy, ix] = center - [ix, iy]
            reg[2:4, iy, ix] = wh
            reg[4:12, iy, ix] = (q / 8 - [ix, iy]).ravel()
            reg[12, iy, ix] = float(row['is_vehicle'])
            mask[iy, ix] = 1
        return (torch.from_numpy(im.transpose(2, 0, 1).copy()).float() / 255,
                torch.from_numpy(heat), torch.from_numpy(reg), torch.from_numpy(mask))


def ocr_collate(batch):
    images, targets, texts = zip(*batch)
    return torch.stack(images), torch.cat(targets), torch.tensor([len(x) for x in targets], dtype=torch.long), list(texts)


class GenericPlateDataset(Dataset):
    """Generic plate locations; no target type, OCR text or vehicle supervision."""
    def __init__(self, root, split_file, part, size=512, augment=False):
        self.root, self.size, self.augment = Path(root), size, augment
        manifest = json.loads((self.root / 'geometry.json').read_text(encoding='utf-8'))
        if manifest.get('schema') != 'alpr_geometry_v1':
            raise ValueError('Expected geometry-only data from alpr.import_coco')
        selected = set(json.loads(Path(split_file).read_text(encoding='utf-8'))[part])
        by_image = {record['image']: record for record in manifest['images']}
        if selected - by_image.keys():
            raise ValueError('Split contains images missing from geometry.json')
        self.samples, self.skipped = [], []
        for name in sorted(selected):
            record = by_image[name]
            width, height = record['width'], record['height']
            scale = min(size / width, size / height)
            nw, nh = round(width * scale), round(height * scale)
            sx, sy, px, py = nw / width, nh / height, (size - nw) // 2, (size - nh) // 2
            centers = []
            for obj in record['objects']:
                x, y, w, h = obj['bbox']
                centers.append((int(((x + w / 2) * sx + px) // 8), int(((y + h / 2) * sy + py) // 8)))
            if len(set(centers)) != len(centers):
                self.skipped.append({'image': name, 'reason': f'Multiple plate centres in one stride-8 cell at size={size}'})
                continue
            self.samples.append(record)
        if not self.samples:
            raise ValueError(f'No usable generic detector samples in {part}; inspect sampling report or increase --size')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record = self.samples[index]
        im, (sx, sy, px, py) = letterbox(read_image(self.root / record['image']), self.size)
        if self.augment:
            im = np.clip(im.astype(float) * random.uniform(.65, 1.25) + random.uniform(-12, 12), 0, 255).astype(np.uint8)
        grid = self.size // 8
        heat, reg, mask = np.zeros((1, grid, grid), np.float32), np.zeros((13, grid, grid), np.float32), np.zeros((grid, grid), np.float32)
        for obj in record['objects']:
            x, y, w, h = obj['bbox']
            lo = (np.array([x, y]) * [sx, sy] + [px, py]) / 8
            hi = (np.array([x + w, y + h]) * [sx, sy] + [px, py]) / 8
            center, wh = (lo + hi) / 2, hi - lo
            ix, iy = np.floor(center).astype(int)
            if not (0 <= ix < grid and 0 <= iy < grid) or mask[iy, ix]:
                raise ValueError(f'Invalid or colliding centre: {record["image"]}')
            gaussian(heat[0], ix, iy, min(wh) / 2)
            reg[:2, iy, ix], reg[2:4, iy, ix] = center - [ix, iy], wh
            if obj['quad'] is not None:
                q = np.asarray(obj['quad'], np.float32) * [sx, sy] + [px, py]
                reg[4:12, iy, ix] = (q / 8 - [ix, iy]).ravel()
                # Generic-only target slot: corner availability, NOT a vehicle flag.
                reg[12, iy, ix] = 1
            mask[iy, ix] = 1
        return (torch.from_numpy(im.transpose(2, 0, 1).copy()).float() / 255,
                torch.from_numpy(heat), torch.from_numpy(reg), torch.from_numpy(mask))
