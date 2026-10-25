import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from alpr.common import read_image, write_image, write_meta, read_meta, write_labels
from alpr.rights import prepare_jobs, metadata, matches_saved, read_annotation_image
from alpr.sync_rights import sync_rights


class RightsTests(unittest.TestCase):
    def fixture(self, root, names):
        images = root / 'input'
        records = []
        for i, name in enumerate(names):
            p = images / name
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.random.default_rng(i).integers(0, 256, (40, 80, 3), dtype=np.uint8)).save(p)
            records.append({'file': name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                            'source_url': 'https://example.org/' + str(i),
                            'license': 'CC BY 4.0' if i % 2 else 'CC BY-SA 4.0'})
        registry = root / 'rights.json'
        registry.write_text(json.dumps(records), encoding='utf-8')
        return images, registry, records

    def test_nested_sources_licenses_and_exclusions(self):
        with tempfile.TemporaryDirectory() as temp:
            folder, registry, records = self.fixture(Path(temp), ['a/a.png', 'b/b.png', 'b/excluded.png'])
            records[-1]['excluded'] = True
            registry.write_text(json.dumps({'files': records}), encoding='utf-8')
            jobs = prepare_jobs(folder, registry, recursive=True)
            self.assertEqual(len(jobs), 2)
            self.assertEqual([metadata(j['record'])[1] for j in jobs], ['CC BY-SA 4.0', 'CC BY 4.0'])
            self.assertEqual([j['relative'] for j in jobs], ['a/a.png', 'b/b.png'])

    def test_missing_changed_and_unlicensed_files_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder, registry, records = self.fixture(Path(temp), ['a.png'])
            records[0]['license'] = 'UNKNOWN'
            registry.write_text(json.dumps(records), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'License not assigned'):
                prepare_jobs(folder, registry)
            records[0]['license'] = 'CC BY 4.0'
            records[0]['sha256'] = 'bad'
            registry.write_text(json.dumps(records), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'differs'):
                prepare_jobs(folder, registry)
            registry.write_text('[]', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'No rights record'):
                prepare_jobs(folder, registry)

    def test_collision_names_have_distinct_label_stems(self):
        with tempfile.TemporaryDirectory() as temp:
            folder, registry, _ = self.fixture(Path(temp), ['a/same.jpg', 'b/same.jpg', 'b/same.png'])
            jobs = prepare_jobs(folder, registry, True)
            self.assertEqual(len({Path(j['dest']).stem.casefold() for j in jobs}), 3)
            self.assertEqual([j['dest'] for j in jobs], [j['dest'] for j in prepare_jobs(folder, registry, True)])

    def test_own_photo_source(self):
        record = {'file': 'a.png', 'author': 'My Name', 'rights_status': 'own_photo_user_granted_cc_by_4_0', 'license': 'CC BY 4.0'}
        self.assertEqual(metadata(record), ('own_photo:My Name', 'CC BY 4.0'))

    def test_sync_preserves_annotations_and_backups(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder, registry, records = self.fixture(root, ['photo.webp'])
            data = root / 'dataset'
            dest = 'images/real/photo.webp.jpg'
            write_image(data / dest, read_image(folder / 'photo.webp'))
            row = dict(image=dest, plate_num='AA12377', plate_type='type1b', bbox='1,1,20,10',
                       quad='1,1,21,1,21,11,1,11', is_vehicle='1', is_synthetic='0',
                       source='UNVERIFIED', license='UNKNOWN', conditions='day')
            write_meta(data, [row, dict(row, plate_num='BB12377')])
            write_labels(data, [row, dict(row, plate_num='BB12377')])
            (data / 'annotation_audit.jsonl').write_text(json.dumps({'image': dest, 'covered_faces': []})+'\n')
            before = (data / 'meta.csv').read_bytes()
            image_before = (data / dest).read_bytes()
            label = data / 'labels/real/photo.webp.txt'
            label_before = label.read_bytes()
            preview = sync_rights(data, folder, registry)
            self.assertEqual(preview['changed_rows'], 2)
            self.assertEqual((data / 'meta.csv').read_bytes(), before)
            result = sync_rights(data, folder, registry, True)
            self.assertEqual((Path(result['backup'])/'meta.csv').read_bytes(), before)
            self.assertEqual((data / dest).read_bytes(), image_before)
            self.assertEqual(label.read_bytes(), label_before)
            for updated, number in zip(read_meta(data), ['AA12377', 'BB12377']):
                expected = dict(row, plate_num=number, source=records[0]['source_url'], license=records[0]['license'])
                self.assertEqual(updated, expected)
            self.assertEqual(sync_rights(data, folder, registry)['changed_rows'], 0)
            self.assertIn(dest, json.loads((data / 'rights_provenance.json').read_text()))
            (data / dest).write_bytes(b'changed')
            current = (data / 'meta.csv').read_bytes()
            with self.assertRaisesRegex(ValueError, 'verified source'):
                sync_rights(data, folder, registry, True)
            self.assertEqual((data / 'meta.csv').read_bytes(), current)

    def test_face_cover_is_reproduced(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder, _, _ = self.fixture(root, ['photo.png'])
            pixels = read_image(folder/'photo.png')
            pixels[3:11, 2:16] = 110
            write_image(root/'covered.jpg', pixels)
            self.assertTrue(matches_saved(folder/'photo.png', root/'covered.jpg', [[2, 3, 15, 10]]))
            self.assertFalse(matches_saved(folder/'photo.png', root/'covered.jpg'))

    def test_pillow_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            folder, _, _ = self.fixture(Path(temp), ['photo.png'])
            with patch('alpr.common.read_image', side_effect=ValueError('Unsupported codec')):
                image = read_annotation_image(folder/'photo.png')
            self.assertEqual(image.shape, (40, 80, 3))

    def test_annotator_switches_rights_per_image(self):
        from alpr.annotate import Annotator
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder, registry, records = self.fixture(root, ['one.png', 'two.png'])
            annotator = Annotator.__new__(Annotator)
            annotator.root = root/'data'
            annotator.jobs = prepare_jobs(folder, registry)
            annotator.files = [j['path'] for j in annotator.jobs]
            annotator.rights_path = str(registry)
            annotator.audit = {}
            annotator.display = lambda: None
            for i in range(2):
                annotator.index = i
                annotator.load()
                self.assertEqual((annotator.source, annotator.license), metadata(records[i]))
                self.assertEqual(annotator.dest, annotator.jobs[i]['dest'])


if __name__ == '__main__':
    unittest.main()
