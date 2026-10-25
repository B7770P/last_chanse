import unittest
import cv2
import numpy as np
from alpr.alignment import alignment_quads,order_quad

class AlignmentTests(unittest.TestCase):
    def test_perspective_plate_boundary(self):
        image=np.full((180,260,3),30,np.uint8)
        quad=np.float32([[50,35],[215,50],[200,140],[40,125]])
        cv2.fillConvexPoly(image,quad.astype(int),(240,240,240))
        cv2.putText(image,'123',(65,100),cv2.FONT_HERSHEY_SIMPLEX,1,(20,20,20),2)
        candidates=alignment_quads(image,[40,35,215,140])
        self.assertTrue(candidates)
        self.assertLess(min(np.mean(np.linalg.norm(q-quad,axis=1)) for q in candidates),5)
    def test_order_and_invalid_inputs(self):
        expected=np.float32([[10,10],[80,20],[75,70],[5,60]])
        np.testing.assert_array_equal(order_quad(expected[[2,0,3,1]]),expected)
        self.assertEqual(alignment_quads(np.zeros((100,100,3),np.uint8),[0,0,3,3]),[])
        self.assertEqual(alignment_quads(np.zeros((100,100,3),np.uint8),[10,10,90,80]),[])

if __name__=='__main__':unittest.main()
