import unittest
from unittest.mock import patch
import numpy as np
import torch
from alpr.common import letterbox, iou
from alpr.infer import decode_detections, detect_plates
from alpr.train import box_overlap_loss
from alpr.review_detector import score_frame


class GeometryTests(unittest.TestCase):
    def test_overlap_loss_and_gradients(self):
        target=torch.zeros(1,12,1,1);target[:,:2]=.5
        target[:,2:4]=torch.tensor([2.,1.]).reshape(1,2,1,1)
        mask=torch.ones(1,1,1)
        self.assertAlmostEqual(float(box_overlap_loss(target,target,mask)),0,places=6)
        predicted=target.clone();predicted[:,0]+=.5;predicted.requires_grad_(True)
        loss=box_overlap_loss(predicted,target,mask);self.assertGreater(float(loss),0)
        loss.backward();self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertGreater(float(predicted.grad.abs().sum()),0)
        self.assertEqual(float(box_overlap_loss(predicted,target,torch.zeros_like(mask))),0)

    def test_geometry_roundtrip_non_square_images(self):
        for h,w in [(301,997),(997,301),(513,777)]:
            im=np.zeros((h,w,3),np.uint8)
            _,(sx,sy,px,py)=letterbox(im,512)
            q=np.float32([[w*.3,h*.4],[w*.6,h*.4],[w*.6,h*.6],[w*.3,h*.6]])
            scaled=q*np.array([sx,sy])+[px,py]
            lo,hi=scaled.min(0)/8,scaled.max(0)/8
            center=(lo+hi)/2;ix,iy=np.floor(center).astype(int)
            off=np.clip(center-[ix,iy],1e-7,1-1e-7);wh=hi-lo
            reg=torch.zeros(1,13,64,64);heat=torch.full((1,4,64,64),-20.)
            reg[0,:2,iy,ix]=torch.tensor(np.log(off/(1-off)))
            reg[0,2:4,iy,ix]=torch.tensor(np.log(np.expm1(wh)))
            reg[0,4:12,iy,ix]=torch.tensor((scaled/8-[ix,iy]).ravel())
            heat[0,1,iy,ix]=10
            detections=decode_detections(heat,reg,(sx,sy,px,py),im.shape)
            self.assertEqual(len(detections),1)
            self.assertGreater(iou(detections[0]['box'],np.r_[q.min(0),q.max(0)]),.99999)
            self.assertTrue(np.allclose(detections[0]['quad'],q,atol=.001))

    def test_refinement_coordinates_and_context_class(self):
        class Dummy:
            def __call__(self,x):return None,None
        coarse={'box':np.array([180.,90.,220.,100.]),'quad':np.array([[180.,90.],[220.,90.],[220.,100.],[180.,100.]]),
                'plate_type':'type1b','score':.64,'vehicle':.9}
        crop={'box':np.array([80.,90.,120.,100.]),'quad':np.array([[80.,90.],[120.,90.],[120.,100.],[80.,100.]]),
              'plate_type':'type1a','score':.81,'vehicle':.6}
        with patch('alpr.infer.decode_detections',side_effect=[[coarse],[crop]]) as decode:
            results=detect_plates(Dummy(),np.zeros((200,400,3),np.uint8),refine=True)
        self.assertEqual(decode.call_count,2)
        self.assertEqual(len(results),1)
        self.assertEqual(results[0]['plate_type'],'type1b')
        self.assertAlmostEqual(results[0]['score'],.72)
        self.assertTrue(np.allclose(results[0]['box'],coarse['box']))
        self.assertTrue(np.allclose(results[0]['quad'],coarse['quad']))

    def test_one_to_one_matching(self):
        truth=[{'box':[0,0,20,10]}]
        predictions=[{'box':[0,0,20,10],'score':.8},{'box':[0,0,20,10],'score':.6}]
        self.assertEqual(score_frame(truth,predictions),[(0,0)])
        self.assertEqual(score_frame([],predictions),[])


if __name__=='__main__':unittest.main()
