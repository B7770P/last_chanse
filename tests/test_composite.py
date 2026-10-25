import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from alpr.common import read_meta, write_image, write_meta
from alpr.generate_composite import generate, replace, render
from alpr.validate import validate


class CompositeTests(unittest.TestCase):
    def fixture(self, root):
        data=root/'source'
        rows=[]
        for i,(kind,text) in enumerate([('type1','A123BC77'),('type1a','B234CE77'),
                                        ('type1b','AB12377'),('other','1234AB')]):
            name=f'images/real/{i}.jpg'
            im=np.full((300,500,3),110+i*10,np.uint8)
            write_image(data/name,im)
            q=[100,100,300,100,300,220 if kind=='type1a' else 150,100,220 if kind=='type1a' else 150]
            rows.append(dict(image=name,plate_num=text,plate_type=kind,
                             bbox=f'100,100,200,{120 if kind=="type1a" else 50}',
                             quad=','.join(map(str,q)),is_vehicle='1',is_synthetic='0',
                             source=f'fixture:{i}',license='CC-BY-SA-4.0',conditions='day'))
        # A tiny secondary plate must keep its original text and annotation.
        rows.append(dict(rows[0],plate_num='C345EX77',bbox='10,10,20,8',quad='10,10,30,10,30,18,10,18'))
        held=dict(rows[0],image='images/real/held.jpg',plate_num='K999XX77')
        write_image(data/held['image'],np.full((300,500,3),200,np.uint8))
        rows.append(held)
        write_meta(data,rows)
        split=root/'split.json'
        split.write_text(json.dumps({'train':[f'images/real/{i}.jpg' for i in range(4)],
                                     'val':[held['image']],'test':[]}),encoding='utf-8')
        return data,split

    def test_reproducible_valid_derivatives_and_no_leakage(self):
        with tempfile.TemporaryDirectory(prefix='composite_test_', dir=Path.cwd()) as tmp:
            root=Path(tmp);data,split=self.fixture(root)
            a,b=root/'a',root/'b'
            generate(data,split,a,20,123)
            generate(data,split,b,20,123)
            for p in a.rglob('*'):
                if p.is_file():
                    self.assertEqual(p.read_bytes(),(b/p.relative_to(a)).read_bytes())
            self.assertTrue(validate(a)['valid_structure'])
            provenance=json.loads((a/'provenance.json').read_text(encoding='utf-8'))
            self.assertNotIn('images/real/held.jpg',provenance['parents'])
            rows=read_meta(a)
            self.assertNotIn('K999XX77',{r['plate_num'] for r in rows})
            self.assertTrue(all(r['license']=='CC-BY-SA-4.0' for r in rows))
            self.assertIn('C345EX77',{r['plate_num'] for r in rows})
            with self.assertRaises(ValueError):
                generate(data,split,a,20,123)
            s=json.loads(split.read_text());s['val'].append(s['train'][0]);split.write_text(json.dumps(s))
            with self.assertRaises(ValueError):
                generate(data,split,root/'bad',20,123)
            self.assertFalse((root/'bad').exists())

    def test_compositing_leaves_distant_pixels_unchanged(self):
        im=np.full((200,400,3),180,np.uint8)
        q=np.float32([[100,80],[300,70],[300,120],[100,130]])
        plate=render('AA12377','type1b',np.random.default_rng(7))
        result=replace(im,q,plate)
        self.assertTrue(np.array_equal(im[:50],result[:50]))
        self.assertFalse(np.array_equal(im[80:120,110:290],result[80:120,110:290]))


if __name__=='__main__':
    unittest.main()
