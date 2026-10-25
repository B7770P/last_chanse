import unittest
import numpy as np
import torch
from alpr.infer import Pipeline,yolo_boxes
class FastTests(unittest.TestCase):
 def test_specialist_does_not_run_base_ocr_for_type1(self):
  p=Pipeline.__new__(Pipeline)
  def fail(x):raise AssertionError('base OCR unnecessarily called')
  p.ocr=fail;p.type1_ocr=lambda x:torch.ones(len(x),80,23)
  self.assertEqual(p.recognize_crops(torch.zeros(2,1,48,320),['type1','type1']).shape,(2,80,23))
 def test_base_candidates_reused_in_second_pass(self):
  p=Pipeline.__new__(Pipeline);p.multiscale_type1=True;calls=[]
  def single(im,white_size=None,base_candidates=None):
   calls.append((white_size,base_candidates));p._base_candidates=['sentinel'];return []
  p._single_pass=single;p(np.zeros((10,10,3),np.uint8))
  self.assertEqual(calls,[(None,None),(960,['sentinel'])])
 def test_high_confidence_skips_second_pass(self):
  p=Pipeline.__new__(Pipeline);p.multiscale_type1=True;calls=[]
  def single(im,**kwargs):
   calls.append(1);return [dict(plate_type='type1',plate_num='A123BC77',confidence=.99,score=1)]
  p._single_pass=single;p(np.zeros((10,10,3),np.uint8));self.assertEqual(len(calls),1)
 def test_bulk_box_fields(self):
  class B:data=torch.tensor([[1,2,3,4,.8,2]])
  class R:boxes=B()
  box,kind,score=list(yolo_boxes(R()))[0]
  np.testing.assert_array_equal(box,[1,2,3,4]);self.assertEqual(kind,2);self.assertAlmostEqual(score,.8)
if __name__=='__main__':unittest.main()
