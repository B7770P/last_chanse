"""Fine-tune the general OCR on real train crops plus fully procedural crops (alpr/synth_crops.py).

    python -m alpr.train_ocr_synth --data dataset_composite5000 --split splits/real606.json --resume models/ocr_crops_20261024/best.pt --out models/ocr_synth_20261025

Checkpoint is chosen on real val crops only. Synthetic crops are new every epoch.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .common import ALPHABET, read_meta
from .data import ocr_collate
from .models import load_model, save_model
from .synth_crops import make_rng, synth_crop
from .train import ctc_text
from .train_ocr_crops import Crops


class Synth(Dataset):
    def __init__(self, count, kinds=('type1a', 'type1b'), repeat_bias=0., small_bias=False):
        self.count, self.kinds, self.epoch = count, kinds, 0
        self.repeat_bias, self.small_bias = repeat_bias, small_bias

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        rng = make_rng(i, self.epoch)
        crop, text = synth_crop(rng, self.kinds[i % len(self.kinds)], self.repeat_bias, self.small_bias)
        return torch.from_numpy(crop.copy()).float()[None] / 127.5 - 1, torch.tensor([ALPHABET.index(c) + 1 for c in text]), text


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key in ['data', 'split', 'resume', 'out']:
        p.add_argument('--' + key, required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=.0002)
    p.add_argument('--synth-per-epoch', type=int, default=1600)
    p.add_argument('--real-repeat', type=int, default=3, help='real train crops are shown this many times per epoch')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--repeat-bias', type=float, default=0., help='share of synthetic numbers with a repeated adjacent digit')
    p.add_argument('--small-bias', action='store_true', help='half of synthetic plates 45-90 px wide')
    a = p.parse_args()
    out = Path(a.out)
    assert not out.exists(), f'{out} already exists'
    out.mkdir(parents=True)
    random.seed(20261025)
    np.random.seed(20261025)
    torch.manual_seed(20261025)
    split = json.loads(Path(a.split).read_text(encoding='utf-8'))
    assert not set(split['train']) & set(split['val'] + split['test'])
    rows = [r for r in read_meta(a.data) if r['plate_type'] != 'other' and '#' not in r['plate_num']]
    real = {part: Crops([r for r in rows if r['image'] in split[part]], a.data, part == 'train') for part in ['train', 'val']}
    synth = Synth(a.synth_per_epoch, repeat_bias=a.repeat_bias, small_bias=a.small_bias)
    train = ConcatDataset([real['train']] * a.real_repeat + [synth])
    print(json.dumps({'real_train': len(real['train']), 'real_val': len(real['val']), 'synth_per_epoch': len(synth)}), flush=True)
    loaders = {
        'train': DataLoader(train, batch_size=32, shuffle=True, collate_fn=ocr_collate, num_workers=a.workers,
                            persistent_workers=False),
        'val': DataLoader(real['val'], batch_size=64, shuffle=False, collate_fn=ocr_collate),
    }
    model, _ = load_model(a.resume, 'ocr', 'cuda')
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.0001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    ctc = torch.nn.CTCLoss(blank=0)
    best = (float('inf'), float('inf'))
    history = []
    for epoch in range(a.epochs + 1):
        synth.epoch = epoch  # workers are recreated every epoch, so they see the new value
        metrics = {}
        for part in (['val'] if epoch == 0 else ['train', 'val']):
            model.train(part == 'train')
            total = seen = exact = 0
            with torch.set_grad_enabled(part == 'train'):
                for images, targets, lengths, texts in loaders[part]:
                    logits = model(images.cuda())
                    loss = ctc(logits.log_softmax(-1).transpose(0, 1), targets.cuda(),
                               torch.full((len(images),), logits.shape[1], dtype=torch.long), lengths)
                    assert torch.isfinite(loss)
                    if part == 'train':
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                        opt.step()
                    seen += len(images)
                    total += float(loss.detach()) * len(images)
                    exact += sum(x == y for x, y in zip(ctc_text(logits.detach()), texts))
            metrics[part + '_loss'] = total / seen
            metrics[part + '_exact'] = exact / seen
        if epoch:
            schedule.step()
        record = {'epoch': epoch, **metrics}
        history.append(record)
        print(json.dumps(record), flush=True)
        info = {'epoch': epoch, 'metrics': metrics, 'config': vars(a), 'checkpoint_selection': 'val_exact then val_loss',
                'seed': 20261025, 'crop_version': 'real-contour-v1 + procedural-synth-v1'}
        selection = (-metrics['val_exact'], metrics['val_loss'])
        if selection < best:
            best = selection
            save_model(out / 'best.pt', model, 'ocr', **info)
        save_model(out / 'last.pt', model, 'ocr', **info)
        (out / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    print(json.dumps({'best_val_exact': -best[0], 'baseline_val_exact': history[0]['val_exact']}), flush=True)


if __name__ == '__main__':
    main()
