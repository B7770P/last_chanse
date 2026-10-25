import unittest
from unittest.mock import patch
import numpy as np
import torch
from alpr.infer import Pipeline
from alpr.common import ALPHABET

class YoloPipelineTests(unittest.TestCase):
    def test_yellow_number_rgb_crop_and_coordinates(self):
        class Box:
            xyxy=torch.tensor([[10.,10.,110.,40.]])
            cls=torch.tensor([2]);conf=torch.tensor([.9])
        class Result:boxes=[Box()]
        class Detector:
            def predict(self,image,**kwargs):
                assert image.mode=='RGB'
                assert image.getpixel((0,0))==(255,0,0)
                return [Result()]
        def ocr(crops):
            assert crops.shape==(1,1,48,320)
            seq=[0]
            for char in 'AA12377':seq.extend([ALPHABET.index(char)+1,0])
            logits=torch.full((1,len(seq),len(ALPHABET)+1),-12.)
            for t,token in enumerate(seq):logits[0,t,token]=12.
            return logits
        pipe=Pipeline.__new__(Pipeline)
        pipe.device='cpu';pipe.detector_backend='yolo';pipe.detector=Detector();pipe.ocr=ocr
        pipe.size=640;pipe.det_threshold=.35;pipe.char_threshold=.65
        pipe.keep_nonvehicle=True;pipe.emit_other=False
        image=np.zeros((80,120,3),np.uint8);image[:,:,0]=255
        result=pipe(image)
        self.assertEqual(result[0]['plate_num'],'AA12377')
        self.assertEqual(result[0]['plate_type'],'type1b')
        self.assertIsNone(result[0]['vehicle'])
        np.testing.assert_array_equal(result[0]['quad'],[[10,10],[110,10],[110,40],[10,40]])
    def test_yolo_requires_explicit_vehicle_filter_opt_out(self):
        with self.assertRaisesRegex(ValueError,'vehicle flag'):
            Pipeline('unused','unused',detector_backend='yolo')

if __name__=='__main__':unittest.main()
