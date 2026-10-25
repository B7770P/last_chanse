import unittest,torch,math
from alpr.infer import decode_text
from alpr.common import ALPHABET
class ThresholdTest(unittest.TestCase):
 def test_only_type1_changes_and_default_preserved(self):
  text='A123BC77';tokens=[0]
  for c in text:tokens.extend([ALPHABET.index(c)+1,0])
  logits=torch.full((3,len(tokens),len(ALPHABET)+1),-30.)
  for t,k in enumerate(tokens):logits[:,t,k]=12
  logits[:,1,:]=-30;logits[:,1,ALPHABET.index('A')+1]=math.log(.6);logits[:,1,0]=math.log(.4)
  kinds=['type1','type1a','type1b']
  old=decode_text(logits,.65,kinds);new=decode_text(logits,.65,kinds,.5)
  self.assertEqual(old[0][0],'#123BC77');self.assertEqual(new[0][0],text)
  self.assertEqual(new[1:],old[1:])
if __name__=='__main__':unittest.main()
