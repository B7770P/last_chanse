"""Fully procedural OCR crops: a rendered plate placed in a random scene, then degraded like a real photo.

Nothing here comes from real photos. Each crop goes through the same rectify() as in inference,
with a slightly wrong quad, to imitate detector and alignment errors.
"""
import random

import cv2
import numpy as np

from .common import rectify
from .generate import random_number, render_plate, valid_number


def background(rng, w, h):
    base = rng.uniform(20, 230, 3)
    im = np.empty((h, w, 3), np.float32)
    im[:] = base
    gx = np.linspace(-1, 1, w)[None, :, None] * rng.uniform(-60, 60, 3)
    gy = np.linspace(-1, 1, h)[:, None, None] * rng.uniform(-60, 60, 3)
    im += gx + gy
    for _ in range(int(rng.integers(3, 12))):  # bumpers, grilles, shadows
        x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x2, y2 = x1 + int(rng.integers(10, w)), y1 + int(rng.integers(5, h // 2 + 6))
        cv2.rectangle(im, (x1, y1), (x2, y2), rng.uniform(0, 255, 3).tolist(), -1)
    im += rng.normal(0, rng.uniform(2, 15), im.shape)
    return np.clip(im, 0, 255).astype(np.uint8)


def plate_quad(rng, plate_w, plate_h, canvas_w, canvas_h, width):
    """Plate corners in the canvas: random size, tilt, perspective."""
    height = width * plate_h / plate_w
    cx, cy = canvas_w / 2 + rng.uniform(-.1, .1) * canvas_w, canvas_h / 2 + rng.uniform(-.1, .1) * canvas_h
    q = np.array([[-width / 2, -height / 2], [width / 2, -height / 2], [width / 2, height / 2], [-width / 2, height / 2]])
    yaw = rng.uniform(-.35, .35)  # side view: one edge shorter
    q[[1, 2] if yaw > 0 else [0, 3], 1] *= 1 - abs(yaw)
    pitch = rng.uniform(-.2, .2)  # camera above or below: top or bottom edge shorter
    q[[0, 1] if pitch > 0 else [2, 3], 0] *= 1 - abs(pitch)
    angle = np.deg2rad(rng.uniform(-12, 12))
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return (q @ rot.T + [cx, cy]).astype(np.float32)


def degrade(im, rng):
    im = im.astype(np.float32)
    im = im * rng.uniform(.65, 1.25) + rng.uniform(-30, 30)  # exposure
    im *= rng.uniform(.85, 1.15, 3)  # colour cast
    if rng.random() < .3:  # shadow over part of the plate
        h, w = im.shape[:2]
        x = int(rng.integers(0, w))
        im[:, x:] *= rng.uniform(.4, .8) if rng.random() < .5 else 1
    im = np.clip(im, 0, 255).astype(np.uint8)
    if rng.random() < .3:  # motion blur
        k = int(rng.integers(3, 6))
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2] = 1 / k
        m = cv2.getRotationMatrix2D((k / 2 - .5, k / 2 - .5), rng.uniform(0, 180), 1)
        im = cv2.filter2D(im, -1, cv2.warpAffine(kernel, m, (k, k)))
    if rng.random() < .5:
        im = cv2.GaussianBlur(im, (0, 0), rng.uniform(.3, 1.0))
    noise = rng.normal(0, rng.uniform(0, 8), im.shape)
    im = np.clip(im + noise, 0, 255).astype(np.uint8)
    ok, data = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(25, 92))])
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def repeat_digits(rng, text, kind):
    """Copy one digit onto its right neighbour (77, 33, 00...): CTC tends to merge such pairs."""
    pairs = [i for i in range(len(text) - 1) if text[i].isdigit() and text[i + 1].isdigit()]
    i = pairs[int(rng.integers(len(pairs)))]
    repeated = text[:i + 1] + text[i] + text[i + 2:]
    return repeated if valid_number(repeated, kind) else text


def synth_crop(rng, kind, repeat_bias=0., small_bias=False):
    """Return (gray crop as in inference, text). Defaults reproduce the original crops exactly."""
    text = random_number(rng, kind)
    if repeat_bias and rng.random() < repeat_bias:
        text = repeat_digits(rng, text, kind)
    plate = render_plate(text, kind, rng)
    if rng.random() < .3:  # dirt and wear on the plate itself
        plate = cv2.addWeighted(plate, 1, np.full_like(plate, int(rng.integers(60, 140))), -rng.uniform(.05, .3), 0)
    ph, pw = plate.shape[:2]
    # small plates matter most: widths in the canvas from 55 to 260 px
    if small_bias and rng.random() < .5:  # oversample the small plates that fail most often
        width = float(np.exp(rng.uniform(np.log(45), np.log(90))))
    else:
        width = float(np.exp(rng.uniform(np.log(55), np.log(260))))
    canvas_w, canvas_h = int(width * 2.2), int(width * 2.2 * ph / pw + width * .6)
    canvas = background(rng, canvas_w, canvas_h)
    quad = plate_quad(rng, pw, ph, canvas_w, canvas_h, width)
    src = np.float32([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]])
    m = cv2.getPerspectiveTransform(src, quad)
    warped = cv2.warpPerspective(plate, m, (canvas_w, canvas_h), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), m, (canvas_w, canvas_h))
    canvas[mask > 127] = warped[mask > 127]
    canvas = degrade(canvas, rng)
    # inference never gets exact corners: jitter them by up to ~4 % of the plate size
    jitter = rng.normal(0, .015, (4, 2)) * [width, width * ph / pw * 2]
    if rng.random() < .3:  # axis-aligned box instead of corners, as when alignment fails
        x1, y1 = quad.min(0)
        x2, y2 = quad.max(0)
        guess = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    else:
        guess = (quad + jitter).astype(np.float32)
    try:
        return rectify(canvas, guess, kind), text
    except ValueError:
        return rectify(canvas, quad, kind), text


def make_rng(index, epoch, seed=20261025):
    return np.random.default_rng([seed, epoch, index])


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    p = argparse.ArgumentParser(description='Save sample crops for a visual check')
    p.add_argument('--out', default='synth_preview')
    p.add_argument('--count', type=int, default=24)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(a.count):
        kind = ['type1a', 'type1b', 'type1'][i % 3]
        crop, text = synth_crop(make_rng(i, 0), kind)
        rows.append(cv2.copyMakeBorder(crop, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255))
        random.seed(i)
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(out / 'sheet.png'), sheet)
    print('saved', out / 'sheet.png')
