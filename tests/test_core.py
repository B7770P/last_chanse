import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from alpr.common import ALPHABET, PATTERN, format_number, rectify, letterbox, read_meta, write_labels, write_meta
from alpr.generate import generate, render_plate
from alpr.validate import validate
from alpr.split import split_dataset
from alpr.models import Detector, Recognizer, save_model, load_model
from alpr.train import detector_loss
from alpr.infer import decode_text, decode_detections
from alpr.evaluate import score


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_mask_and_unknown_positions(self):
        self.assertTrue(PATTERN.fullmatch('A123BC777'))
        self.assertTrue(PATTERN.fullmatch('#123##7##'))
        self.assertTrue(PATTERN.fullmatch('A123BC977'))
        self.assertIsNone(format_number('A123BC7', [.99] * 7))
        self.assertEqual(format_number('A123BC77', [.99] * 7 + [.1]), 'A123BC7#')
        self.assertEqual(format_number('A123BC977', [.99] * 9), 'A123BC977')

    def test_reproducible_generation_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a', Path(tmp) / 'b'
            generate(a, count=24, seed=13)
            generate(b, count=24, seed=13)
            hashes = lambda p: [hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted((p / 'images').rglob('*.jpg'))]
            self.assertEqual(hashes(a), hashes(b))
            self.assertEqual((a / 'meta.csv').read_bytes(), (b / 'meta.csv').read_bytes())
            self.assertTrue(validate(a)['valid_structure'])
            rows = read_meta(a)
            rows[0]['quad'] = '0,0,0,0,0,0,0,0'
            write_meta(a, rows)
            self.assertFalse(validate(a)['valid_structure'])

    def test_split_has_no_plate_or_image_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data'
            generate(root, count=40, seed=12)
            rows = read_meta(root)
            # Deliberately repeat a plate identity in another image.
            target = [r for r in rows if r['plate_type'] != 'other']
            target[1]['plate_num'] = target[0]['plate_num']
            write_meta(root, rows)
            result = split_dataset(root, Path(tmp) / 'split.json')
            owner = {image: part for part in ('train', 'val', 'test') for image in result[part]}
            self.assertEqual(owner[target[0]['image']], owner[target[1]['image']])
            nums = {part: {r['plate_num'] for r in rows if r['image'] in result[part] and r['plate_type'] != 'other'} for part in ('train', 'val', 'test')}
            self.assertFalse(nums['train'] & nums['val'])
            self.assertFalse(nums['train'] & nums['test'])
            self.assertFalse(nums['val'] & nums['test'])

    def test_geometry_and_two_rows(self):
        rng = np.random.default_rng(1)
        plate = render_plate('A123BC777', 'type1a', rng)
        q = [[0, 0], [289, 0], [289, 169], [0, 169]]
        strip = rectify(plate, q, 'type1a')
        self.assertEqual(strip.shape, (48, 320))
        self.assertLess(float(strip[:, :160].mean()), 235)
        self.assertLess(float(strip[:, 160:].mean()), 235)
        with self.assertRaises(ValueError):
            rectify(plate, [[0, 0]] * 4, 'type1a')
        _, (sx, sy, px, py) = letterbox(np.zeros((127, 319, 3), np.uint8), 512)
        point = np.array([120., 70.])
        self.assertTrue(np.allclose((point * [sx, sy] + [px, py] - [px, py]) / [sx, sy], point))

    def test_ctc_repeated_digits_and_mask(self):
        tokens = [ALPHABET.index(c) + 1 for c in 'A111BC77']
        seq = [0]
        for token in tokens:
            seq += [token, token, 0]
        logits = torch.full((1, len(seq), len(ALPHABET) + 1), -12.)
        for t, token in enumerate(seq):
            logits[0, t, token] = 12.
        text, conf = decode_text(logits, .65)[0]
        self.assertEqual(text, 'A111BC77')
        self.assertGreater(conf, .99)

    def test_detection_decode_multiple_plates(self):
        heat = torch.full((1, 4, 16, 16), -20.)
        reg = torch.zeros((1, 13, 16, 16))
        for kind, ix, iy in [(0, 4, 4), (1, 12, 11)]:
            heat[0, kind, iy, ix] = 10.
            reg[0, 2:4, iy, ix] = torch.tensor([3., 2.])
            reg[0, 4:12, iy, ix] = torch.tensor([-1., -1., 2., -1., 2., 2., -1., 2.])
            reg[0, 12, iy, ix] = 10.
        result = decode_detections(heat, reg, (1, 1, 0, 0), (128, 128, 3))
        self.assertEqual(len(result), 2)
        self.assertEqual({r['plate_type'] for r in result}, {'type1', 'type1a'})
        self.assertFalse(decode_detections(torch.full_like(heat, -20), reg, (1, 1, 0, 0), (128, 128, 3)))

    def test_models_loss_backward_checkpoint(self):
        detector = Detector()
        x = torch.rand(2, 3, 128, 128)
        heat, reg = detector(x)
        target_heat = torch.zeros_like(heat)
        target_reg = torch.zeros_like(reg)
        mask = torch.zeros(2, 16, 16)
        target_heat[:, 0, 6, 6] = 1
        target_reg[:, 2:4, 6, 6] = 3
        mask[:, 6, 6] = 1
        loss = detector_loss((heat, reg), (target_heat, target_reg, mask))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(detector.heat.weight.grad.abs().sum()), 0)
        ocr = Recognizer()
        self.assertEqual(ocr(torch.rand(2, 1, 48, 320)).shape, (2, 80, 23))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'detector.pt'
            save_model(path, detector, 'detector', size=128)
            restored, _ = load_model(path, 'detector')
            detector.eval()
            self.assertTrue(torch.allclose(detector(x)[0], restored(x)[0]))
            with self.assertRaises(ValueError):
                load_model(path, 'ocr')

    def test_metric_partial_truth_matching_and_false_positives(self):
        truth = [{'image': 'a.jpg', 'plate_num': 'A123BC7#', 'plate_type': 'type1', 'is_vehicle': '1'},
                 {'image': 'a.jpg', 'plate_num': 'A123BC77', 'plate_type': 'type1', 'is_vehicle': '1'}]
        pred = [{'image': 'a.jpg', 'plate_num': 'A123BC77', 'plate_type': 'type1'},
                {'image': 'a.jpg', 'plate_num': 'A123BC78', 'plate_type': 'type1'},
                {'image': 'negative.jpg', 'plate_num': 'K123BC77', 'plate_type': 'type1'}]
        result = score(truth, pred)['per_type']['type1']
        self.assertEqual((result['tp'], result['fp'], result['fn']), (2, 1, 0))
        result = score(truth[:1], [{'image': 'a.jpg', 'plate_num': 'A123BC##', 'plate_type': 'type1'}])
        self.assertEqual(result['per_type']['type1']['tp'], 0)


if __name__ == '__main__':
    unittest.main()
