import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .common import ALPHABET
from .data import PlateDataset, GenericPlateDataset, ocr_collate
from .models import Detector, Recognizer, save_model, load_model


def box_overlap_loss(decoded, target, mask):
    # Aligned GIoU in cell-relative coordinates. The common cell origin cancels.
    pc, pw = decoded[:, :2], decoded[:, 2:4]
    tc, tw = target[:, :2], target[:, 2:4]
    plo, phi = pc - pw / 2, pc + pw / 2
    tlo, thi = tc - tw / 2, tc + tw / 2
    intersection = (torch.minimum(phi, thi) - torch.maximum(plo, tlo)).clamp_min(0).prod(1)
    union = pw.prod(1) + tw.prod(1) - intersection
    enclosing = (torch.maximum(phi, thi) - torch.minimum(plo, tlo)).clamp_min(0).prod(1)
    giou = intersection / union.clamp_min(1e-6) - (enclosing - union) / enclosing.clamp_min(1e-6)
    return ((1 - giou) * mask).sum() / mask.sum().clamp_min(1)


def detector_loss(outputs, target, box_iou_weight=2.0):
    heat_logits, predicted = outputs
    heat, reg, mask = target
    generic = heat.shape[1] == 1
    if generic:
        # One generic foreground against implicit background. No class is assigned.
        heat_logits = heat_logits.logsumexp(dim=1, keepdim=True)
    p = heat_logits.sigmoid().clamp(1e-5, 1 - 1e-5)
    positive = heat.eq(1).float()
    negative = heat.lt(1).float()
    focal = -(positive * (1 - p).pow(2) * p.log() + negative * (1 - heat).pow(4) * p.pow(2) * (1 - p).log())
    count = mask.sum().clamp(min=1)
    loss_heat = focal.sum() / count
    decoded = torch.cat([predicted[:, :2].sigmoid(), F.softplus(predicted[:, 2:4]), predicted[:, 4:12]], dim=1)
    spatial = F.smooth_l1_loss(decoded, reg[:, :12], reduction='none')
    if generic:
        spatial = torch.cat([spatial[:, :4], spatial[:, 4:] * reg[:, 12:13]], dim=1)
    # Quad coordinates and box sizes are in stride-8 cells.
    loss_reg = (spatial * mask[:, None]).sum() / count
    loss_vehicle = predicted[:, 12].sum() * 0 if generic else (F.binary_cross_entropy_with_logits(predicted[:, 12], reg[:, 12], reduction='none') * mask).sum() / count
    return loss_heat + .2 * loss_reg + .5 * loss_vehicle + box_iou_weight * box_overlap_loss(decoded, reg, mask)


def ctc_text(logits):
    result = []
    for seq in logits.argmax(-1).tolist():
        prev, text = -1, []
        for token in seq:
            if token != prev and token:
                text.append(ALPHABET[token - 1])
            prev = token
        result.append(''.join(text))
    return result


def main():
    p = argparse.ArgumentParser(description='Train a detector or CRNN; models initialized from scratch')
    p.add_argument('--task', choices=['detector', 'ocr'], required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--split', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--ocr-box-probability', type=float, default=0., help='OCR augmentation: fraction of training crops using slightly shifted bounding boxes')
    p.add_argument('--box-iou-weight', type=float, default=2.0, help='GIoU localization weight; 0 reproduces the old detector loss')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', help='Model warm start (optimizer is reset; useful for real-data finetuning)')
    p.add_argument('--limit-batches', type=int, default=0, help='Debug only; 0 means all batches')
    p.add_argument('--generic-plates', action='store_true', help='Geometry-only pretraining on imported COCO; no type/vehicle/OCR labels')
    a = p.parse_args()
    if not 0 <= a.ocr_box_probability <= 1:
        p.error('--ocr-box-probability must be in [0,1]')
    if a.generic_plates and a.task != 'detector':
        p.error('--generic-plates is only for detector pretraining, not OCR')
    if a.size < 128 or a.size % 32 or min(a.batch, a.epochs, a.threads) < 1 or a.lr <= 0 or a.box_iou_weight < 0:
        p.error('size must be >=128 and divisible by 32; batch/epochs/threads/lr must be positive')
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA is unavailable. Install the CUDA torch wheel or explicitly use --device cpu')
    torch.set_num_threads(a.threads)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    if a.generic_plates:
        train = GenericPlateDataset(a.data, a.split, 'train', a.size, True)
        val = GenericPlateDataset(a.data, a.split, 'val', a.size)
    else:
        train = PlateDataset(a.data, a.split, 'train', a.task, a.size, True)
        val = PlateDataset(a.data, a.split, 'val', a.task, a.size)
    collate = ocr_collate if a.task == 'ocr' else None
    if a.task == 'ocr':
        train.ocr_box_probability = a.ocr_box_probability
    generator = torch.Generator().manual_seed(a.seed)
    tr = DataLoader(train, a.batch, shuffle=True, num_workers=a.workers, collate_fn=collate, generator=generator)
    va = DataLoader(val, a.batch, num_workers=a.workers, collate_fn=collate)
    model = (Detector() if a.task == 'detector' else Recognizer()).to(a.device)
    if a.resume:
        model, _ = load_model(a.resume, a.task, a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.0001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.generic_plates:
        sampling = {'size': a.size, 'train_images': len(train), 'val_images': len(val),
                    'skipped_train': train.skipped, 'skipped_val': val.skipped}
        (out / 'sampling_report.json').write_text(json.dumps(sampling, indent=2), encoding='utf-8')
        print(json.dumps({'generic_plate_pretraining': True, 'train_images': len(train), 'val_images': len(val),
                          'skipped_train': len(train.skipped), 'skipped_val': len(val.skipped)}), flush=True)
    best = (float('inf'),float('inf'))
    history = []
    for epoch in range(a.epochs):
        metrics = {}
        for part, loader in [('train', tr), ('val', va)]:
            model.train(part == 'train')
            total, seen, exact = 0., 0, 0
            with torch.set_grad_enabled(part == 'train'):
                for bi, batch in enumerate(loader):
                    if a.limit_batches and bi >= a.limit_batches:
                        break
                    if a.task == 'detector':
                        images, heat, reg, mask = [x.to(a.device) for x in batch]
                        loss = detector_loss(model(images), (heat, reg, mask), a.box_iou_weight)
                    else:
                        images, targets, lengths, texts = batch
                        images, targets = images.to(a.device), targets.to(a.device)
                        logits = model(images)
                        t = logits.shape[1]
                        # CPU lengths are accepted by CTC; log probabilities have T,N,C shape.
                        loss = ctc(logits.log_softmax(-1).transpose(0, 1), targets,
                                   torch.full((len(images),), t, dtype=torch.long), lengths)
                        exact += sum(x == y for x, y in zip(ctc_text(logits.detach()), texts))
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Non-finite loss at epoch {epoch}, batch {bi}; inspect annotations')
                    if part == 'train':
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                        opt.step()
                    total += float(loss.detach()) * len(images)
                    seen += len(images)
            metrics[f'{part}_loss'] = total / max(seen, 1)
            if a.task == 'ocr':
                metrics[f'{part}_exact'] = exact / max(seen, 1)
        schedule.step()
        record = {'epoch': epoch + 1, **metrics}
        history.append(record)
        print(json.dumps(record), flush=True)
        info = {'epoch': epoch + 1, 'size': a.size, 'seed': a.seed, 'debug': bool(a.limit_batches),
                'generic_plate_pretraining': a.generic_plates,
                'detector_loss_version': 'giou-v1', 'box_iou_weight': a.box_iou_weight,
                'metrics': metrics, 'config': vars(a)}
        save_model(out / 'last.pt', model, a.task, **info)
        selection = (-metrics['val_exact'],metrics['val_loss']) if a.task == 'ocr' else (metrics['val_loss'],0.)
        info['checkpoint_selection'] = 'val_exact, then val_loss' if a.task == 'ocr' else 'val_loss'
        if selection < best:
            best = selection
            save_model(out / 'best.pt', model, a.task, **info)
        (out / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
