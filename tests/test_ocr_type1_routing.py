import unittest
import torch
from alpr.infer import Pipeline
class RouteTests(unittest.TestCase):
 def test_only_type1_replaced(self):
  p=Pipeline.__new__(Pipeline);p.ocr=lambda x:torch.ones(len(x),80,23);p.type1_ocr=lambda x:torch.full((len(x),80,23),2.)
  out=p.recognize_crops(torch.zeros(3,1,48,320),['type1a','type1','type1b'])
  self.assertTrue(torch.all(out[0]==1));self.assertTrue(torch.all(out[1]==2));self.assertTrue(torch.all(out[2]==1))
 def test_no_specialist_preserves_results(self):
  p=Pipeline.__new__(Pipeline);p.ocr=lambda x:torch.ones(len(x),80,23)
  self.assertTrue(torch.all(p.recognize_crops(torch.zeros(1,1,48,320),['type1'])==1))
if __name__=='__main__':unittest.main()
