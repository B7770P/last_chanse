import unittest
from alpr.infer import merge_specialist
def item(kind,box,score=.9):return {'plate_type':kind,'box':box,'score':score}
class MergeTest(unittest.TestCase):
 def test_keep_yellow_and_replace_wrong_other(self):
  yellow=item('type1b',[0,0,100,30]);other=item('other',[200,0,300,30]);white=item('type1',[200,0,300,30])
  result=merge_specialist([yellow,other],[white,item('type1',[0,0,100,30])])
  self.assertEqual(result,[yellow,white]);self.assertIs(result[0],yellow)
 def test_fallback_and_no_duplicate(self):
  old=item('type1a',[0,0,50,40]);new=item('type1a',[1,0,51,40]);distant=item('type1',[100,0,160,30])
  result=merge_specialist([old,distant],[new]);self.assertEqual(result,[new,distant])
if __name__=='__main__':unittest.main()
