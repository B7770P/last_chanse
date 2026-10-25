"""Почему найденный номер остался без текста (missed_no_text) — разбор на val, конфигурация V4 по умолчанию.

    python diagnose_no_text.py                      # печать + diagnose_no_text.json + обрезки в debug_no_text/
    python diagnose_no_text.py --extra-size 0       # то же для v7 без добавочного прохода

Для каждого такого номера: кандидаты детектора рядом с рамкой (тип, score), прошли ли они через слияние со
специалистом белых номеров, и для каждой обрезки (рамка и варианты по контуру) сырой текст OCR до маски
с уверенностью каждого символа и причина, по которой текст отброшен. fin_test здесь не используется.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import alpr.infer as infer
import compare_fast_v5 as cf
from alpr.common import ALPHABET, format_number, read_image, read_meta, rectify
from analyze_val_errors import TARGETS, classify, iou, norm, overlaps

merged_log = []
original_merge = infer.merge_specialist


def logged_merge(base, white):
    result = original_merge(base, white)
    merged_log.append(result)
    return result


def raw_decode(seq):
    """Greedy CTC exactly as decode_text, but keeps every character with its confidence."""
    tokens = seq.argmax(-1)
    chars, confidences, prev = [], [], -1
    for t, token in enumerate(tokens):
        if token and token != prev:
            chars.append(ALPHABET[token - 1])
            confidences.append(float(seq[t, token]))
        elif token and token == prev:
            confidences[-1] = max(confidences[-1], float(seq[t, token]))
        prev = token
    return chars, confidences


def save_png(path, crop):
    ok, data = cv2.imencode('.png', crop)
    if ok:
        data.tofile(str(path))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default='dataset_composite5000')
    p.add_argument('--split', default='splits/real606.json')
    p.add_argument('--ocr', default='models/ocr_synth2_20261025/best.pt')
    p.add_argument('--extra-size', type=int, default=960, help='0 = без добавочного прохода (v7)')
    p.add_argument('--crops', default='debug_no_text')
    p.add_argument('--out', default='diagnose_no_text.json')
    a = p.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    infer.merge_specialist = logged_merge

    split = json.loads(Path(a.split).read_text(encoding='utf-8'))
    meta = defaultdict(list)
    for r in read_meta(a.data):
        meta[r['image']].append(r)
    extra = dict(base_extra_size=a.extra_size, base_extra_gate='large_or_empty') if a.extra_size else {}
    pipe = infer.Pipeline(cf.MODELS['detector'], a.ocr, 'cuda', det_threshold=.35, char_threshold=.4,
                          keep_nonvehicle=True, detector_backend='yolo', align=True,
                          white_detector=cf.MODELS['white'], type1_ocr=cf.MODELS['type1_ocr'],
                          type1_char_threshold=.5, multiscale_type1=True, direct_yolo=True,
                          gate_second_pass=True, cuda_graphs=True, **extra)
    pipe(np.full((640, 640, 3), 114, np.uint8))
    crops_dir = Path(a.crops)
    crops_dir.mkdir(exist_ok=True)

    from alpr.alignment import alignment_quads
    records, reasons = [], Counter()
    for name in split['val']:
        path = Path(a.data) / name
        with Image.open(path) as probe:
            width = probe.size[0]
        im = read_image(path, 1600)
        scale = width / im.shape[1]
        merged_log.clear()
        out = pipe(im)
        base = list(pipe._base_candidates)
        merged = merged_log[0] if merged_log else base  # first pass only
        to_orig = lambda rows: [dict(r, box=[float(v) * scale for v in r['box'][:4]]) for r in rows]
        for row in meta[name]:
            truth = norm(row['plate_num'])
            if row['plate_type'] not in TARGETS or not truth or '#' in truth:
                continue
            x, y, w, h = (float(v) for v in row['bbox'].split(','))
            label = [x, y, x + w, y + h]
            category, _ = classify(row, label, to_orig(out), to_orig(base))
            if category != 'missed_no_text':
                continue
            near_base = [c for c in base if overlaps(label, [float(v) * scale for v in c['box'][:4]])]
            near_merged = [c for c in merged if overlaps(label, [float(v) * scale for v in c['box'][:4]])]
            rec = dict(image=name, truth=truth, plate_type=row['plate_type'], width=round(w), height=round(h),
                       base_candidates=[f'{c["plate_type"]} {c["score"]:.2f}' for c in near_base],
                       after_merge=[f'{c["plate_type"]} {c["score"]:.2f} {c.get("detector_source", "base")}'
                                    for c in near_merged], crops=[])
            reason = None
            if not near_merged:
                reason = 'dropped_in_white_merge'
            elif all(c['plate_type'] == 'other' for c in near_merged):
                reason = 'detected_as_other'
            for ci, cand in enumerate(c for c in near_merged if c['plate_type'] != 'other'):
                kind = cand['plate_type']
                quads = [('box', cand['quad'])] + [('contour', q) for q in alignment_quads(im, cand['box'])]
                tensors, labels = [], []
                for label_name, quad in quads:
                    try:
                        crop = rectify(im, quad, kind)
                    except ValueError:
                        rec['crops'].append(dict(alignment=label_name, result='rectify_failed'))
                        continue
                    index = len(records) * 100 + ci * 10 + len(tensors)
                    save_png(crops_dir / f'{index:05d}_{truth}_{kind}_{label_name}.png', crop)
                    tensors.append(torch.from_numpy(crop.copy()).float().unsqueeze(0) / 127.5 - 1)
                    labels.append(label_name)
                if not tensors:
                    continue
                with torch.inference_mode():
                    logits = pipe.recognize_crops(torch.stack(tensors).to(pipe.device), [kind] * len(tensors))
                probs = logits.softmax(-1).cpu().numpy()
                cutoff = .5 if kind == 'type1' else .4
                need = (7,) if kind == 'type1b' else (8, 9)
                for label_name, seq in zip(labels, probs):
                    chars, confs = raw_decode(seq)
                    text = format_number(chars, confs, cutoff, kind=kind)
                    if text is None:
                        result = f'length_{len(chars)}_need_{"/".join(map(str, need))}'
                    elif text.count('#') == len(text):
                        result = 'all_hash'
                    else:
                        result = f'text_ok:{text}'
                    fixed, fixed_conf = infer.repair_length(chars, confs, kind)
                    if not infer.mask_fits(fixed, kind):
                        decoded = infer.mask_beam_decode(seq, kind)
                        fixed, fixed_conf = decoded if decoded else (fixed, fixed_conf)
                    chain = format_number(fixed, fixed_conf, cutoff, kind=kind)
                    rec['crops'].append(dict(kind=kind, score=round(cand['score'], 2), alignment=label_name,
                                             raw=''.join(chars), conf=[round(c, 2) for c in confs], result=result,
                                             repair_and_mask=chain, repair_and_mask_raw=''.join(fixed)))
            if reason is None:
                results = [c['result'] for c in rec['crops']]
                if any(r.startswith('text_ok') for r in results):
                    reason = 'text_ok_but_dropped_later'
                elif any(r.startswith('length_') for r in results):
                    reason = 'wrong_length' + ('_type_mismatch' if any(c.get('kind') != row['plate_type']
                                                                      for c in rec['crops'] if 'kind' in c) else '')
                elif results and all(r == 'rectify_failed' for r in results):
                    reason = 'rectify_failed'
                else:
                    reason = 'all_hash'
            rec['reason'] = reason
            reasons[reason] += 1
            records.append(rec)

    for r in records:
        print(f'{Path(r["image"]).name[:50]:50s} {r["truth"]:10s} {r["plate_type"]:6s} w={r["width"]} '
              f'детектор: {", ".join(r["base_candidates"]) or "-"} | после слияния: {", ".join(r["after_merge"]) or "-"}')
        for c in r['crops']:
            if 'raw' in c:
                conf = ' '.join(f'{ch}{v:.2f}' for ch, v in zip(c['raw'], c['conf']))
                print(f'    {c["kind"]:6s} {c["alignment"]:7s} raw={c["raw"]:12s} [{conf}] -> {c["result"]}')
                print(f'        ремонт длины + маска: {c["repair_and_mask_raw"]} -> {c["repair_and_mask"]}  '
                      f'(правильно {r["truth"]}{", СОВПАЛО" if c["repair_and_mask"] == r["truth"] else ""})')
            else:
                print(f'    {c["alignment"]:7s} -> {c["result"]}')
        print(f'    ПРИЧИНА: {r["reason"]}')
    print('\nИтого по причинам: ' + ', '.join(f'{k} {v}' for k, v in reasons.most_common()))
    Path(a.out).write_text(json.dumps(dict(reasons=dict(reasons), records=records), ensure_ascii=False, indent=2),
                           encoding='utf-8')
    print(f'подробности: {a.out}, обрезки: {crops_dir}/')


if __name__ == '__main__':
    main()
