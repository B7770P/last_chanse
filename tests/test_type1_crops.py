import unittest
import numpy as np
from alpr.type1_crops import context_crops
from alpr.common import rectify

class CropTests(unittest.TestCase):
    def test_preserves_lower_glyph_strokes(self):
        image=np.full((112,520,3),255,np.uint8)
        image[92:101,80:440]=0
        q=np.float32([[0,0],[519,0],[519,111],[0,111]])
        old=rectify(image,q,'type1')
        crops=context_crops(image,q)
        self.assertGreater(old.mean(),250)
        self.assertLess(crops[0][0].mean(),245)
        for crop,quad,label in crops:
            self.assertEqual(crop.shape,(48,320))
            self.assertEqual(crop.dtype,np.uint8)
            np.testing.assert_array_equal(quad,q)
    def test_degenerate_quad_rejected(self):
        self.assertEqual(context_crops(np.zeros((10,10,3),np.uint8),np.zeros((4,2))),[])

if __name__=='__main__':unittest.main()
