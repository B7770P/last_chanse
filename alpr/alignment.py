"""Image-only plate corner proposals. No annotation or recognized text is used."""
import cv2
import numpy as np
from .common import iou

def order_quad(points):
    p=np.asarray(points,np.float32).reshape(4,2)
    # Sort around the centre, then start at the top-left corner.
    c=p.mean(0);p=p[np.argsort(np.arctan2(p[:,1]-c[1],p[:,0]-c[0]))]
    p=np.roll(p,-int(np.argmin(p.sum(1))),axis=0)
    return p

def alignment_quads(image,box,max_candidates=4):
    x1,y1,x2,y2=map(float,box);w=x2-x1;h=y2-y1
    if min(w,h)<8:return []
    lo=np.maximum(np.floor([x1-.15*w,y1-.15*h]),0).astype(int)
    hi=np.minimum(np.ceil([x2+.15*w,y2+.15*h]),[image.shape[1],image.shape[0]]).astype(int)
    roi=image[lo[1]:hi[1],lo[0]:hi[0]]
    if roi.size==0:return []
    scale=min(3.,400/max(roi.shape[:2]));roi=cv2.resize(roi,None,fx=scale,fy=scale)
    gray=cv2.cvtColor(roi,cv2.COLOR_RGB2GRAY)
    gray=cv2.GaussianBlur(gray,(3,3),0)
    _,bright=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    masks=[bright,cv2.morphologyEx(bright,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)),cv2.Canny(gray,50,150)]
    proposals=[]
    for mask in masks:
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour)<w*h*scale*scale*.3:continue
            perimeter=cv2.arcLength(contour,True)
            for eps in [.02,.04]:
                approx=cv2.approxPolyDP(contour,eps*perimeter,True)
                if len(approx)!=4 or not cv2.isContourConvex(approx):continue
                q=order_quad(approx.reshape(4,2)/scale+lo)
                area=abs(cv2.contourArea(q));ratio=area/(w*h)
                bounds=np.r_[q.min(0),q.max(0)]
                overlap=iou(bounds,np.asarray(box))
                if not .4<=ratio<=1.25 or overlap<.55:continue
                lengths=np.linalg.norm(q-np.roll(q,1,axis=0),axis=1)
                if lengths.min()<5 or max(lengths[0],lengths[2])/min(lengths[0],lengths[2])>2.5:continue
                if any(np.mean(np.linalg.norm(q-old[1],axis=1))<max(1.5,min(w,h)*.04) for old in proposals):continue
                proposals.append((overlap-.2*abs(1-ratio),q))
    return [q for _,q in sorted(proposals,key=lambda x:-x[0])[:max_candidates]]
