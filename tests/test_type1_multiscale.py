import unittest
from alpr.infer import merge_type1_results
def item(kind,text,confidence,box=(0,0,100,25)):
 return dict(plate_type=kind,plate_num=text,confidence=confidence,box=list(box))
class MergeTests(unittest.TestCase):
 def test_higher_confidence_and_no_duplicate(self):
  base=[item('type1','A123BC77',.5)];high=[item('type1','A123BC777',.7)]
  r=merge_type1_results(base,high);self.assertEqual(len(r),1);self.assertEqual(r[0]['plate_num'],'A123BC777');self.assertEqual(base[0]['plate_num'],'A123BC77')
 def test_other_types_preserved(self):
  for kind in ['type1a','type1b','other']:
   base=[item(kind,'original',.4)]
   self.assertEqual(merge_type1_results(base,[item('type1','wrong',.99)]),base)
 def test_new_detection_added_and_high_square_ignored(self):
  r=merge_type1_results([],[item('type1','A123BC77',.8),item('type1a','X',.99)])
  self.assertEqual(len(r),1)
if __name__=='__main__':unittest.main()
