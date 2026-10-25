"""Offline image directory -> UTF-8 semicolon CSV. No network calls."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from .common import read_image, letterbox, rectify, iou, TYPES, ALPHABET, LETTERS, format_number
from .models import load_model


def decode_detections(heat, reg, transform, shape, threshold=.35, topk=60):
    scores = heat.sigmoid()
    peaks = (F.max_pool2d(scores, 3, 1, 1) == scores) & (scores >= threshold)
    positions = peaks[0].nonzero()
    if not len(positions):
        return []
    values = scores[0, positions[:, 0], positions[:, 1], positions[:, 2]]
    order = values.argsort(descending=True)[:topk]
    positions, values = positions[order].cpu().numpy(), values[order].cpu().numpy()
    r = reg[0].cpu().numpy()
    sx, sy, px, py = transform
    h, w = shape[:2]
    result = []
    for (kind, iy, ix), confidence in zip(positions, values):
        raw = r[:, iy, ix]
        offset = 1 / (1 + np.exp(-np.clip(raw[:2], -50, 50)))
        wh = np.logaddexp(0, raw[2:4]) * 8
        center = (np.array([ix, iy]) + offset) * 8
        box = np.concatenate([(center - wh / 2 - [px, py]) / [sx, sy],
                              (center + wh / 2 - [px, py]) / [sx, sy]])
        box[[0, 2]] = np.clip(box[[0, 2]], 0, w)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, h)
        quad = ((raw[4:12].reshape(4, 2) + [ix, iy]) * 8 - [px, py]) / [sx, sy]
        vehicle = float(1 / (1 + np.exp(-np.clip(raw[12], -50, 50))))
        if box[2] - box[0] < 8 or box[3] - box[1] < 5:
            continue
        if any(iou(box, item['box']) > .4 for item in result):
            continue
        result.append({'plate_type': TYPES[kind], 'score': float(confidence), 'box': box,
                       'quad': quad, 'vehicle': vehicle})
    return result


@torch.inference_mode()
def detect_plates(model, im, device='cpu', size=512, threshold=.35,
                  refine=False, proposal_threshold=.15, max_refinements=12):
    """Optional bounded coarse-to-fine localization; retain full-frame type evidence.

    Crop detections are only accepted when overlapping an existing proposal.
    Scores are geometric means of full-frame and crop scores, not probabilities.
    """
    if size < 128 or size % 32:
        raise ValueError('size must be >=128 and divisible by 32')
    if not 0 <= proposal_threshold <= threshold <= 1 or max_refinements < 1:
        raise ValueError('Require 0 <= proposal_threshold <= threshold <= 1 and positive refinement limit')

    def forward(image, cutoff):
        padded, transform = letterbox(image, size)
        tensor = torch.from_numpy(padded.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255
        heat, reg = model(tensor)
        return decode_detections(heat, reg, transform, image.shape, cutoff)

    coarse = forward(im, proposal_threshold if refine else threshold)
    if not refine:
        return coarse
    height, width = im.shape[:2]
    candidates, regions = [], []
    for original in coarse[:max_refinements]:
        box = original['box']
        center = (box[:2] + box[2:]) / 2
        side = min(max((box[2] - box[0]) * 5, (box[3] - box[1]) * 8, 100), max(height, width))
        x1, y1 = max(0, int(center[0] - side / 2)), max(0, int(center[1] - side / 2))
        x2, y2 = min(width, int(center[0] + side / 2)), min(height, int(center[1] + side / 2))
        region = np.array([x1, y1, x2, y2])
        if x2 <= x1 or y2 <= y1 or any(iou(region, other) > .8 for other in regions):
            continue
        regions.append(region)
        for item in forward(im[y1:y2, x1:x2], proposal_threshold):
            item['box'] += np.array([x1, y1, x1, y1])
            item['quad'] += np.array([x1, y1])
            if iou(item['box'], box) < .1:
                continue
            item['crop_type'] = item['plate_type']
            item['plate_type'] = original['plate_type']
            item['vehicle'] = original['vehicle']
            item['coarse_score'] = original['score']
            item['score'] = float(np.sqrt(item['score'] * original['score']))
            candidates.append(item)
    for original in coarse:
        if original['score'] >= threshold and not any(iou(original['box'], item['box']) > .1 for item in candidates):
            candidates.append(original)
    kept = []
    for item in sorted(candidates, key=lambda item: -item['score']):
        if item['score'] >= threshold and not any(iou(item['box'], other['box']) > .4 for other in kept):
            kept.append(item)
    return kept


def pad_quad(quad, pad):
    """Grow a plate quadrilateral outward by pad of its size on every side."""
    q = np.asarray(quad, np.float32).reshape(4, 2)
    center = q.mean(0)
    return center + (q - center) * (1 + 2 * pad)


def repair_length(chars, confidences, kind):
    """One character too many (doubled CTC output): drop the least confident one that leaves a valid mask."""
    lengths = (7,) if kind == 'type1b' else (8, 9)
    if len(chars) in lengths or len(chars) - 1 not in lengths:
        return chars, confidences
    letter_positions = (0, 1) if kind == 'type1b' else (0, 4, 5)
    best = None
    for i in range(len(chars)):
        rest = chars[:i] + chars[i + 1:]
        if all((c in LETTERS) if j in letter_positions else c.isdigit() for j, c in enumerate(rest)):
            if best is None or confidences[i] < confidences[best]:
                best = i
    if best is None:
        return chars, confidences
    return chars[:best] + chars[best + 1:], confidences[:best] + confidences[best + 1:]


def mask_fits(chars, kind):
    """Valid length and every character of the right class (letter/digit) for the plate type."""
    lengths = (7,) if kind == 'type1b' else (8, 9)
    letter_positions = (0, 1) if kind == 'type1b' else (0, 4, 5)
    return len(chars) in lengths and all((c in LETTERS) if i in letter_positions else c.isdigit()
                                         for i, c in enumerate(chars))


def mask_beam_decode(seq, kind, beam=20, min_prob=1e-3):
    """CTC prefix beam search over softmax frames, keeping only prefixes consistent with the plate mask.

    Returns (chars, confidences) of the most probable complete valid string, or None. A character's confidence
    is the largest frame probability with which it was emitted on the kept path."""
    lengths = (7,) if kind == 'type1b' else (8, 9)
    letter_positions = (0, 1) if kind == 'type1b' else (0, 4, 5)
    max_len = max(lengths)
    allowed = [{k for k in range(1, len(ALPHABET) + 1) if (ALPHABET[k - 1] in LETTERS) == (i in letter_positions)}
               for i in range(max_len)]
    beams = {(): [1., 0., ()]}  # prefix -> [p_blank, p_nonblank, confidences]
    for row in seq:
        tokens = [k for k in np.flatnonzero(row > min_prob) if k]
        new = {}
        def entry(prefix, confs):
            if prefix not in new:
                new[prefix] = [0., 0., confs]
            return new[prefix]
        for prefix, (pb, pnb, confs) in beams.items():
            total = pb + pnb
            entry(prefix, confs)[0] += total * row[0]
            if prefix:
                last = prefix[-1]
                e = entry(prefix, confs)
                e[1] += pnb * row[last]
                if e[2] and row[last] > e[2][-1]:
                    e[2] = e[2][:-1] + (float(row[last]),)
            if len(prefix) >= max_len:
                continue
            for k in tokens:
                if k not in allowed[len(prefix)]:
                    continue
                e = entry(prefix + (k,), confs + (float(row[k]),))
                e[1] += (pb if prefix and k == prefix[-1] else total) * row[k]
        best = sorted(new.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:beam]
        norm = sum(v[0] + v[1] for _, v in best) or 1.
        beams = {k: [v[0] / norm, v[1] / norm, v[2]] for k, v in best}
    done = [(v[0] + v[1], k, v[2]) for k, v in beams.items() if len(k) in lengths]
    if not done:
        return None
    _, prefix, confs = max(done, key=lambda x: x[0])
    return [ALPHABET[k - 1] for k in prefix], list(confs)


def decode_text(logits, threshold, kinds=None, type1_threshold=None, length_repair=False, mask_decode=False):
    """CTC decoding with a separate format for each detected plate type."""
    if kinds is None:
        kinds = ['type1'] * logits.shape[0]
    if len(kinds) != logits.shape[0] or any(kind not in TYPES[:3] for kind in kinds):
        raise ValueError('Provide one supported OCR plate type per crop')
    probs = logits.softmax(-1).cpu().numpy()
    decoded = []
    for seq, kind in zip(probs, kinds):
        tokens = seq.argmax(-1)
        chars, confidences = [], []
        prev = -1
        for t, token in enumerate(tokens):
            if token and token != prev:
                chars.append(ALPHABET[token - 1])
                confidences.append(float(seq[t, token]))
            elif token and token == prev:
                confidences[-1] = max(confidences[-1], float(seq[t, token]))
            prev = token
        cutoff = type1_threshold if kind=='type1' and type1_threshold is not None else threshold
        if length_repair:
            chars, confidences = repair_length(chars, confidences, kind)
        if mask_decode and not mask_fits(chars, kind):
            constrained = mask_beam_decode(seq, kind)
            if constrained is not None:
                chars, confidences = constrained
        text = format_number(chars, confidences, cutoff, kind=kind)
        conf = float(np.exp(np.mean(np.log(np.maximum(confidences, 1e-8))))) if confidences else 0.
        if text is not None:
            # '#' is uncertainty, not a confidently recognized glyph.
            conf *= sum(c != '#' for c in text) / len(text)
        decoded.append((text, conf))
    return decoded


def merge_specialist(base,white):
    yellow=[p for p in base if p['plate_type']=='type1b']
    merged=list(yellow)
    for p in sorted(white,key=lambda p:-p['score']):
        if not any(iou(p['box'],q['box'])>.4 for q in merged):merged.append(p)
    # Preserve old white detections when the specialist misses, and keep other only outside accepted plates.
    for p in sorted((p for p in base if p['plate_type']!='type1b'),key=lambda p:(p['plate_type']=='other',-p['score'])):
        if not any(iou(p['box'],q['box'])>.4 for q in merged):merged.append(p)
    return merged


def merge_type1_results(base,high):
    """Choose overlapping type1 outputs by confidence; preserve other types."""
    merged=list(base)
    for item in high:
        if item['plate_type']!='type1':continue
        matches=[j for j,b in enumerate(merged) if iou(b['box'],item['box'])>.4]
        if any(merged[j]['plate_type']!='type1' for j in matches):continue
        if not matches:merged.append({**item,'resolution_source':960});continue
        j=max(matches,key=lambda j:iou(merged[j]['box'],item['box']))
        if item['confidence']>merged[j]['confidence']:
            merged[j]={**item,'resolution_source':960}
    return sorted(merged,key=lambda x:(x['box'][1],x['box'][0]))


def yolo_boxes(result):
    """Copy detector boxes from GPU once, avoiding per-box synchronizations."""
    boxes=result.boxes
    if hasattr(boxes,'data'):
        for row in boxes.data.cpu().numpy():
            yield row[:4],int(row[-1]),float(row[-2])
    else:
        for b in boxes:
            yield b.xyxy[0].cpu().numpy(),int(b.cls.item()),float(b.conf.item())


class DirectYolo:
    """Same backend, letterbox and NMS as ultralytics predict, without its per-call overhead."""

    def __init__(self, yolo, device, graph_sizes=()):
        yolo.predict(np.full((64, 64, 3), 114, np.uint8), device=device, imgsz=640, verbose=False)
        self.backend = yolo.predictor.model  # AutoBackend that predict() itself uses
        self.stride = int(getattr(self.backend, 'stride', 32))
        self.square = bool(graph_sizes)
        self.graphs = {}
        if graph_sizes:
            # Anchors must live inside each graph: a cached tensor would be freed when another size is used.
            for m in self.backend.modules():
                if hasattr(m, 'anchors') and hasattr(m, 'dynamic'):
                    m.dynamic = True
            for size in graph_sizes:
                self._capture(torch.zeros(1, 3, size, size, device=device))

    @torch.inference_mode()
    def _capture(self, tensor):
        """CUDA graph for one fixed input shape: removes Python and kernel launch overhead."""
        static_in = tensor.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.backend(static_in)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = self.backend(static_in)
        self.graphs[tuple(tensor.shape)] = (graph, static_in, static_out)

    def forward(self, tensor):
        key = tuple(tensor.shape)
        if key not in self.graphs:
            return self.backend(tensor)
        graph, static_in, static_out = self.graphs[key]
        static_in.copy_(tensor)
        graph.replay()
        return static_out

    def detect(self, tensor, shape, conf):
        from ultralytics.utils import nms, ops
        preds = self.forward(tensor)
        det = nms.non_max_suppression(preds, conf, .4, None, True, max_det=60, nc=0,
                                      end2end=getattr(self.backend, 'end2end', False))[0]
        det[:, :4] = ops.scale_boxes(tensor.shape[2:], det[:, :4], shape)
        return [(row[:4], int(row[5]), float(row[4])) for row in det.cpu().numpy()]


def yolo_tensor(im, size, device, stride=32, square=False):
    """RGB image -> letterboxed tensor exactly as in ultralytics predict (rect=True); square for CUDA graphs."""
    from ultralytics.data.augment import LetterBox
    padded = LetterBox((size, size), auto=not square, stride=stride)(image=im)
    return torch.from_numpy(padded).to(device).permute(2, 0, 1).unsqueeze(0).contiguous().float().div_(255)


class Pipeline:
    def __init__(self, detector, ocr, device='cpu', size=None, det_threshold=.35,
                 char_threshold=.65, vehicle_threshold=.5, keep_nonvehicle=False, emit_other=False, refine=False,
                 detector_backend='custom', align=True, white_detector=None, type1_ocr=None, type1_crop=True, type1_char_threshold=None, multiscale_type1=False,
                 direct_yolo=False, gate_second_pass=False, gate_conf=.1, cuda_graphs=False,
                 base_extra_size=None, base_extra_gate='always', base_extra_min_side=1600, length_repair=False,
                 mask_decode=False, crop_pad=0., multi_crop=False, base_extra_weights=None):
        self.type1_crop = type1_crop and detector_backend == 'yolo'
        if type1_char_threshold is not None and not 0 <= type1_char_threshold <= 1:
            raise ValueError('type1_char_threshold must be in [0,1]')
        self.type1_char_threshold = type1_char_threshold
        if multiscale_type1 and (detector_backend!='yolo' or not white_detector):
            raise ValueError('multiscale type1 requires YOLO and white_detector')
        self.multiscale_type1 = multiscale_type1
        self.device = device
        self.detector_backend = detector_backend
        if detector_backend == 'yolo':
            if not keep_nonvehicle:
                raise ValueError('YOLO has no vehicle flag head; explicitly pass --keep-nonvehicle')
            if refine:
                raise ValueError('--refine applies only to the custom detector')
            if not Path(detector).is_file():
                raise ValueError('A local YOLO checkpoint is required')
            import os
            os.environ['YOLO_OFFLINE'] = 'true'
            os.environ['YOLO_AUTOINSTALL'] = 'false'
            from ultralytics import YOLO
            self.detector = YOLO(str(detector))
            if [self.detector.names[i] for i in range(4)] != list(TYPES):
                raise ValueError('YOLO class mapping does not match plate types')
            ckpt = {'size': 640}
        else:
            self.detector, ckpt = load_model(detector, 'detector', device)
        if ckpt.get('generic_plate_pretraining'):
            raise ValueError('Generic COCO checkpoint needs fine-tuning on plate types and vehicle flags before final inference')
        self.ocr, _ = load_model(ocr, 'ocr', device)
        self.type1_ocr = load_model(type1_ocr,'ocr',device)[0] if type1_ocr else None
        self.size = size or ckpt.get('size', 512)
        if self.size < 128 or self.size % 32:
            raise ValueError('size must be >=128 and divisible by 32')
        self.det_threshold, self.char_threshold = det_threshold, char_threshold
        self.vehicle_threshold = vehicle_threshold
        self.keep_nonvehicle, self.emit_other = keep_nonvehicle, emit_other
        self.refine = refine
        self.align = align and detector_backend == 'yolo'
        self.white_detector = None
        if white_detector:
            if detector_backend != 'yolo' or not Path(white_detector).is_file():
                raise ValueError('White specialist requires YOLO backend and a local checkpoint')
            from ultralytics import YOLO
            self.white_detector=YOLO(str(white_detector))
            if [self.white_detector.names[i] for i in range(4)] != list(TYPES):
                raise ValueError('Specialist class mapping mismatch')
        if gate_second_pass and not (direct_yolo and multiscale_type1):
            raise ValueError('gate_second_pass requires direct_yolo and multiscale_type1')
        self.direct_yolo, self.gate_second_pass, self.gate_conf = direct_yolo, gate_second_pass, gate_conf
        # Optional extra pass of the base detector at another size (None = off, the default behaviour).
        if base_extra_size is not None and (not direct_yolo or base_extra_size < 128 or base_extra_size % 32):
            raise ValueError('base_extra_size requires direct_yolo and a size >=128 divisible by 32')
        if base_extra_gate not in ('always', 'large_or_empty'):
            raise ValueError("base_extra_gate must be 'always' or 'large_or_empty'")
        self.base_extra_size, self.base_extra_gate, self.base_extra_min_side = base_extra_size, base_extra_gate, base_extra_min_side
        self.length_repair, self.mask_decode = length_repair, mask_decode
        self.crop_pad, self.multi_crop = crop_pad, multi_crop  # 0 / False = the default crops
        if direct_yolo:
            if detector_backend != 'yolo':
                raise ValueError('direct_yolo requires the YOLO backend')
            if cuda_graphs and not str(device).startswith('cuda'):
                raise ValueError('cuda_graphs requires a CUDA device')
            self._direct = {'base': DirectYolo(self.detector, device, (self.size,) if cuda_graphs else ())}
            if base_extra_size:
                # Separate copy of the base weights: a second input size on the same modules corrupts the first CUDA graph.
                from ultralytics import YOLO
                extra_model = YOLO(str(base_extra_weights or detector))  # default: same weights as the base detector
                if [extra_model.names[i] for i in range(4)] != list(TYPES):
                    raise ValueError('Extra detector class mapping does not match plate types')
                self._direct['base_extra'] = DirectYolo(extra_model, device, (base_extra_size,) if cuda_graphs else ())
            if self.white_detector is not None:
                sizes = (self.size, 960) if multiscale_type1 else (self.size,)
                self._direct['white'] = DirectYolo(self.white_detector, device, sizes if cuda_graphs else ())
        elif cuda_graphs:
            raise ValueError('cuda_graphs requires direct_yolo')

    def _yolo(self, which, im, size, conf):
        """(box, class, score) rows of the base or white YOLO detector."""
        if getattr(self, 'direct_yolo', False):
            model = self._direct[which]
            key = (size, model.stride, model.square)
            if key not in self._tensors:  # one letterbox per size, shared by both detectors
                self._tensors[key] = yolo_tensor(im, size, self.device, model.stride, model.square)
            return model.detect(self._tensors[key], im.shape, conf)
        from PIL import Image
        model = self.detector if which == 'base' else self.white_detector
        source = self._yolo_input if getattr(self, '_yolo_input', None) is not None else Image.fromarray(im)
        prediction = model.predict(source, device=self.device, imgsz=size, conf=conf, iou=.4, agnostic_nms=True,
                                   max_det=60, verbose=False)[0]
        return list(yolo_boxes(prediction))

    def recognize_crops(self,tensors,kinds):
        specialized=getattr(self,'type1_ocr',None)
        if specialized is None:return self.ocr(tensors)
        indices=[i for i,k in enumerate(kinds) if k=='type1']
        others=[i for i,k in enumerate(kinds) if k!='type1']
        if not others:return specialized(tensors)
        if not indices:return self.ocr(tensors)
        base=self.ocr(tensors[others]);special=specialized(tensors[indices])
        result=base.new_empty((len(tensors),*base.shape[1:]));result[others]=base;result[indices]=special
        return result

    @torch.inference_mode()
    def __call__(self, im):
        direct=getattr(self,'direct_yolo',False)
        self._yolo_input=np.ascontiguousarray(im[:,:,::-1]) if getattr(self,'multiscale_type1',False) and not direct else None
        self._tensors,self._type1_hint={},False
        base=self._single_pass(im)
        if not getattr(self,'multiscale_type1',False):return base
        if base and all(x['plate_type']=='type1' and '#' not in x['plate_num'] and x['confidence']/max(x['score'],1e-8)>=.98 for x in base):return base
        # The 960px pass only adds type1; skip it when neither detector sees any type1 even at low confidence.
        if getattr(self,'gate_second_pass',False) and not self._type1_hint and not any(x['plate_type']=='type1' for x in self._base_candidates):return base
        high=self._single_pass(im,white_size=960,base_candidates=self._base_candidates)
        return merge_type1_results(base,high)

    @torch.inference_mode()
    def _single_pass(self, im, white_size=None, base_candidates=None):
        if base_candidates is not None:
            candidates=list(base_candidates)
        elif self.detector_backend == 'yolo':
            candidates = []
            for box,kind,score in self._yolo('base', im, self.size, self.det_threshold):
                x1,y1,x2,y2 = box
                candidates.append({'box':box,'quad':np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]),
                                   'plate_type':TYPES[kind],'score':score,
                                   'vehicle':None})
            extra=getattr(self,'base_extra_size',None)
            if extra and (self.base_extra_gate=='always' or not candidates or max(im.shape[:2])>=self.base_extra_min_side):
                for box,kind,score in self._yolo('base_extra', im, extra, self.det_threshold):
                    if any(iou(box,c['box'])>.4 for c in candidates):continue
                    x1,y1,x2,y2 = box
                    candidates.append({'box':box,'quad':np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]),
                                       'plate_type':TYPES[kind],'score':score,
                                       'vehicle':None,'detector_source':f'base_{extra}'})
        else:
            candidates = detect_plates(self.detector, im, self.device, self.size, self.det_threshold,
                                       refine=self.refine, proposal_threshold=min(.15, self.det_threshold))
        self._base_candidates=list(candidates)
        if getattr(self,'white_detector',None) is not None:
            gate=getattr(self,'gate_second_pass',False) and white_size is None
            rows=self._yolo('white',im,white_size or self.size,min(self.gate_conf,self.det_threshold) if gate else self.det_threshold)
            if gate:
                self._type1_hint=any(TYPES[k]=='type1' for _,k,_ in rows)
                # NMS keeps the same boxes above the threshold, so filtering afterwards changes nothing else
                rows=[r for r in rows if r[2]>=self.det_threshold]
            white=[]
            for box,kind_index,score in rows:
                kind=TYPES[kind_index]
                if kind not in ('type1','type1a'):continue
                x1,y1,x2,y2=box
                white.append({'box':box,'quad':np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]),
                              'plate_type':kind,'score':score,'vehicle':None,'detector_source':'white_specialist'})
            candidates=merge_specialist(candidates,white)
        crops, selected, output = [], [], []
        for candidate_id,item in enumerate(candidates):
            if white_size is not None and item['plate_type']!='type1':continue
            if not self.keep_nonvehicle and item['vehicle'] < self.vehicle_threshold:
                continue
            if item['plate_type'] == 'other':
                if self.emit_other:
                    output.append({**item, 'plate_num': '', 'confidence': item['score']})
                continue
            pad=getattr(self,'crop_pad',0.)
            try:
                crop = rectify(im, pad_quad(item['quad'],pad) if pad else item['quad'], item['plate_type'])
            except ValueError:
                continue
            crops.append(torch.from_numpy(crop.copy()).float().unsqueeze(0) / 127.5 - 1)
            selected.append({**item,'candidate_id':candidate_id,'alignment':'box'})
            if getattr(self,'multi_crop',False):
                try:
                    crops.append(torch.from_numpy(rectify(im,pad_quad(item['quad'],.06),item['plate_type']).copy()).float().unsqueeze(0)/127.5-1)
                    selected.append({**item,'candidate_id':candidate_id,'alignment':'box_pad'})
                except ValueError:
                    pass
            if item['plate_type']=='type1' and getattr(self,'type1_crop',False):
                from .type1_crops import context_crops
                for extra,quad,label in context_crops(im,item['quad']):
                    crops.append(torch.from_numpy(extra.copy()).float().unsqueeze(0)/127.5-1)
                    selected.append({**item,'quad':quad,'candidate_id':candidate_id,'alignment':label})
            if getattr(self,'align',False):
                from .alignment import alignment_quads
                for quad in alignment_quads(im,item['box']):
                    if pad:quad=pad_quad(quad,pad)
                    try:crop=rectify(im,quad,item['plate_type'])
                    except ValueError:continue
                    crops.append(torch.from_numpy(crop.copy()).float().unsqueeze(0)/127.5-1)
                    selected.append({**item,'quad':quad,'candidate_id':candidate_id,'alignment':'contour'})
        if crops:
            kinds = [item['plate_type'] for item in selected]
            logits = self.recognize_crops(torch.stack(crops).to(self.device),kinds)
            recognized,readings={},{}
            for item, (text, ocr_conf) in zip(selected, decode_text(logits, self.char_threshold, kinds, getattr(self,'type1_char_threshold',None),
                                                                  getattr(self,'length_repair',False),
                                                                  getattr(self,'mask_decode',False))):
                if text is None or text.count('#') == len(text):
                    continue
                confidence = item['score'] * ocr_conf
                if not self.keep_nonvehicle:
                    confidence *= item['vehicle']
                result={**item, 'plate_num': text, 'confidence': float(np.clip(confidence, 0, 1))}
                index=item['candidate_id']
                # Require a confidence margin before replacing a readable original crop.
                if index not in recognized or result['confidence']>recognized[index]['confidence']+.03:
                    recognized[index]=result
                if '#' not in text:readings.setdefault(index,[]).append(result)
            if getattr(self,'multi_crop',False):
                # Prefer a complete reading (no '#') over a partial one; otherwise keep the choice above.
                for index,complete in readings.items():
                    if '#' in recognized[index]['plate_num']:
                        recognized[index]=max(complete,key=lambda r:r['confidence'])
            output.extend(recognized.values())
        return sorted(output, key=lambda x: (x['box'][1], x['box'][0]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--detector', required=True)
    p.add_argument('--ocr', required=True)
    p.add_argument('--no-type1-crop',action='store_false',dest='type1_crop',help='Disable additional full-height type1 crops')
    p.add_argument('--type1-ocr',help='Optional specialized OCR checkpoint for type1 only')
    p.add_argument('--detector-backend', choices=['custom','yolo'], default='custom')
    p.add_argument('--white-detector',help='Optional local YOLO specialist for type1/type1a; keeps base yellow crops')
    p.add_argument('--align', action='store_true', default=True, help='Try image-derived plate corners before OCR (YOLO default)')
    p.add_argument('--no-align', action='store_false', dest='align', help='Disable YOLO crop alignment for comparison')
    p.add_argument('--device', default='cpu')
    p.add_argument('--size', type=int)
    p.add_argument('--det-threshold', type=float, default=.35)
    p.add_argument('--multiscale-type1',action='store_true',help='Slower: combine normal and 960px specialist OCR outputs for type1')
    p.add_argument('--type1-char-threshold',type=float,help='Override character confidence threshold for type1 only')
    p.add_argument('--char-threshold', type=float, default=.65)
    p.add_argument('--vehicle-threshold', type=float, default=.5)
    p.add_argument('--keep-nonvehicle', action='store_true')
    p.add_argument('--emit-other', action='store_true')
    p.add_argument('--refine', action='store_true', help='Refine plate boxes on up to 12 enlarged crops; additional inference cost')
    p.add_argument('--benchmark', help='Save timings as JSON; includes reading + inference + CSV writing, excludes startup/warmup')
    p.add_argument('--details', help='Optional JSON with boxes and corners for local detection evaluation')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--direct-yolo', action='store_true', help='Faster: call the YOLO backend directly, one shared letterbox')
    p.add_argument('--gate-second-pass', action='store_true', help='Skip the 960px pass when no type1 is seen at 640px')
    p.add_argument('--cuda-graphs', action='store_true', help='With --direct-yolo: square 640/960 input and CUDA graphs (less CPU work)')
    p.add_argument('--prefetch', action='store_true', help='Read the next image in a background thread')
    p.add_argument('--decode-max-side', type=int, default=0, help='Decode large JPEGs at reduced size, long side >= N')
    p.add_argument('--skip-broken', action='store_true', help='Log unreadable images and continue (default: fail)')
    p.add_argument('--base-extra-size', type=int, help='Extra pass of the base detector at this size (default: off)')
    p.add_argument('--base-extra-gate', choices=['always', 'large_or_empty'], default='always',
                   help='large_or_empty: extra pass only if long side >= --base-extra-min-side or no plate found')
    p.add_argument('--base-extra-min-side', type=int, default=1600)
    p.add_argument('--length-repair', action='store_true', help='Drop one surplus OCR character if that fits the plate mask')
    p.add_argument('--mask-decode', action='store_true', help='If greedy text breaks the plate mask, use mask-constrained CTC beam search')
    p.add_argument('--crop-pad', type=float, default=0., help='Grow plate quads by this fraction before OCR (default 0)')
    p.add_argument('--multi-crop', action='store_true', help='Also OCR a 6%% padded crop; prefer a complete reading')
    p.add_argument('--base-extra-weights', help='Separate weights for the --base-extra-size pass (default: --detector)')
    a = p.parse_args()
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA unavailable; use --device cpu or install a compatible torch wheel')
    if a.threads < 1 or any(not 0 <= v <= 1 for v in (a.det_threshold, a.char_threshold, a.vehicle_threshold)):
        p.error('threads must be positive and thresholds in [0,1]')
    folder = Path(a.input)
    if not folder.is_dir():
        p.error('--input must be an existing directory')
    files = sorted(x for x in folder.iterdir() if x.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp') and x.is_file())
    if not files:
        p.error('No supported images directly inside --input directory; expected JPG, JPEG, PNG or WebP')
    torch.set_num_threads(a.threads)
    import cv2
    cv2.setNumThreads(1)
    pipeline = Pipeline(a.detector, a.ocr, a.device, a.size, a.det_threshold, a.char_threshold,
                        a.vehicle_threshold, a.keep_nonvehicle, a.emit_other, a.refine, a.detector_backend, a.align,a.white_detector,a.type1_ocr,a.type1_crop,a.type1_char_threshold,a.multiscale_type1,
                        a.direct_yolo,a.gate_second_pass,cuda_graphs=a.cuda_graphs,
                        base_extra_size=a.base_extra_size,base_extra_gate=a.base_extra_gate,
                        base_extra_min_side=a.base_extra_min_side, length_repair=a.length_repair,
                        mask_decode=a.mask_decode, crop_pad=a.crop_pad, multi_crop=a.multi_crop,
                        base_extra_weights=a.base_extra_weights)
    pipeline(np.full((pipeline.size, pipeline.size, 3), 114, np.uint8))
    if a.device.startswith('cuda'):
        torch.cuda.synchronize()
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    timings, details, failures = [], {}, []
    # Atomic replacement: a failed run must not leave a misleading partial CSV.
    temporary = out.with_suffix(out.suffix + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image', 'plate_num', 'plate_type', 'confidence'], delimiter=';')
            writer.writeheader()
            pool = None
            if a.prefetch:
                from concurrent.futures import ThreadPoolExecutor
                pool = ThreadPoolExecutor(1)  # cv2.imdecode releases the GIL, so reading overlaps the models
                pending = pool.submit(read_image, files[0], a.decode_max_side)
            last = time.perf_counter()
            for index, path in enumerate(files):
                start = last if pool else time.perf_counter()
                try:
                    if pool:
                        future = pending
                        if index + 1 < len(files):
                            pending = pool.submit(read_image, files[index + 1], a.decode_max_side)
                        image = future.result()
                    else:
                        image = read_image(path, a.decode_max_side)
                    predictions = pipeline(image)
                except (ValueError, OSError) as exc:
                    if not a.skip_broken:
                        raise
                    failures.append(path.name)
                    print(f'Skipped {path.name}: {exc}', file=sys.stderr)
                    continue
                details[path.name] = [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in r.items()} for r in predictions]
                for r in predictions:
                    writer.writerow({'image': path.name, 'plate_num': r['plate_num'], 'plate_type': r['plate_type'],
                                     'confidence': f'{r["confidence"]:.6f}'})
                if a.device.startswith('cuda'):
                    torch.cuda.synchronize()
                last = time.perf_counter()
                timings.append((last - start) * 1000)
        temporary.replace(out)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    stats = {'images': len(timings), 'failed_images': failures, 'device': a.device, 'size': pipeline.size,
             'mean_ms': float(np.mean(timings)) if timings else None,
             'p95_ms': float(np.percentile(timings, 95)) if timings else None,
             'includes': 'image decoding, preprocessing, models, postprocessing and CSV write; startup/warmup excluded',
             'timing': ('wall time between consecutive results; the next image is read in parallel' if a.prefetch
                        else 'per image, sequential'),
             'hardware': torch.cuda.get_device_name() if a.device.startswith('cuda') else 'CPU; record processor separately',
             'torch': torch.__version__}
    for path, payload in [(a.benchmark, stats), (a.details, details)]:
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == '__main__':
    main()



