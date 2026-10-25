import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image
import torch

from alpr.data import GenericPlateDataset
from alpr.import_coco import import_coco, plate_geometry
from alpr.infer import Pipeline
from alpr.models import Detector, save_model, load_model
from alpr.train import detector_loss


def fixture(path):
    with zipfile.ZipFile(path, 'w') as z:
        for split, prefix, seed in [('train', 'A123BC', 1), ('valid', 'B456EK', 2), ('test', 'A123BC', 3)]:
            pixels = np.random.default_rng(seed).integers(0, 256, (80, 160, 3), np.uint8)
            image = io.BytesIO()
            Image.fromarray(pixels).save(image, format='JPEG')
            name = prefix + '-77_jpg.rf.' + split + '.jpg'
            z.writestr(split + '/' + name, image.getvalue())
            # An extra independent image keeps train nonempty after grouping.
            other_name = 'independent_' + split + '.jpg'
            second = io.BytesIO()
            Image.fromarray(pixels // 2).save(second, format='JPEG')
            z.writestr(split + '/' + other_name, second.getvalue())
            d = {'licenses': [{'id': 1, 'name': 'CC BY 4.0', 'url': 'https://creativecommons.org/licenses/by/4.0/'}],
                 'categories': [{'id': 1, 'name': 'license-plate'}],
                 'images': [{'id': 0, 'file_name': name, 'width': 160, 'height': 80, 'license': 1},
                            {'id': 1, 'file_name': other_name, 'width': 160, 'height': 80, 'license': 1}],
                 'annotations': [{'id': 0, 'image_id': 0, 'category_id': 1, 'iscrowd': 0,
                                  'bbox': [20, 20, 100, 30], 'segmentation': [[20, 50, 120, 50, 120, 20, 20, 20, 20, 50]]},
                                 {'id': 1, 'image_id': 1, 'category_id': 1, 'iscrowd': 0,
                                  'bbox': [20, 20, 100, 30], 'segmentation': []}]}
            z.writestr(split + '/_annotations.coco.json', json.dumps(d))


class CocoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_polygon_order_missing_corners_and_invalid_bounds(self):
        a = {'bbox': [20, 20, 100, 30], 'segmentation': [[20, 50, 120, 50, 120, 20, 20, 20, 20, 50]]}
        bbox, quad, method, clipped = plate_geometry(a, 160, 80)
        self.assertEqual(quad, [[20., 20.], [120., 20.], [120., 50.], [20., 50.]])
        self.assertEqual(method, 'polygon_four_corners')
        self.assertFalse(clipped)
        self.assertIsNone(plate_geometry({'bbox': bbox, 'segmentation': []}, 160, 80)[1])
        with self.assertRaises(ValueError):
            plate_geometry({'bbox': [158, 0, 10, 20]}, 160, 80)

    def test_archive_and_directory_import_preserve_unknowns_and_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / 'data.zip'
            fixture(archive)
            report = import_coco(archive, base / 'converted')
            self.assertEqual((report['imported_images'], report['imported_annotations']), (6, 6))
            self.assertEqual(report['geometry_methods']['bbox_only'], 3)
            manifest = json.loads((base / 'converted' / 'geometry.json').read_text())
            for record in manifest['images']:
                for obj in record['objects']:
                    self.assertIsNone(obj['plate_num'])
                    self.assertIsNone(obj['plate_type'])
                    self.assertIsNone(obj['is_vehicle'])
            split = json.loads((base / 'converted' / 'split.json').read_text())
            owner = {im: part for part in ('train', 'val', 'test') for im in split[part]}
            linked = [r for r in manifest['images'] if r['original_filename'].startswith('A123BC-')]
            self.assertEqual({owner[r['image']] for r in linked}, {'test'})
            # Same public API works on the user's already extracted folder.
            with zipfile.ZipFile(archive) as z:
                z.extractall(base / 'raw')
            second = import_coco(base / 'raw', base / 'converted2')
            self.assertEqual(report, second)
            with self.assertRaises(ValueError):
                import_coco(archive, base / 'converted')
            dataset = GenericPlateDataset(base / 'converted', base / 'converted' / 'split.json', 'train', 128)
            image, heat, reg, mask = dataset[0]
            self.assertEqual(tuple(heat.shape), (1, 16, 16))
            self.assertEqual(float(reg[12].sum()), 0.)  # bbox only, no invented corners
            model = Detector()
            loss = detector_loss(model(image.unsqueeze(0)), (heat.unsqueeze(0), reg.unsqueeze(0), mask.unsqueeze(0)))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(float(model.reg.weight.grad[12].abs().sum()), 0.)  # no vehicle-label gradients
            self.assertEqual(float(model.reg.weight.grad[4:12].abs().sum()), 0.)  # missing corners ignored
            self.assertGreater(float(model.heat.weight.grad.abs().sum()), 0.)

    def test_generic_checkpoint_requires_finetuning_but_can_warm_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'generic.pt'
            save_model(checkpoint, Detector(), 'detector', generic_plate_pretraining=True)
            net, info = load_model(checkpoint, 'detector')
            self.assertTrue(info['generic_plate_pretraining'])
            with self.assertRaisesRegex(ValueError, 'fine-tuning'):
                Pipeline(checkpoint, Path(tmp) / 'missing_ocr.pt')

    def test_unsafe_zip_is_rejected_and_output_is_not_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / 'unsafe.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('../outside.txt', 'no')
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                import_coco(archive, base / 'out')
            self.assertFalse((base / 'out').exists())


if __name__ == '__main__':
    unittest.main()
