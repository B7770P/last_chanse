"""Bounded OCR alternatives preserving the full height of tight type1 boxes."""
import cv2
import numpy as np

def context_crops(image,quad):
    q=np.asarray(quad,np.float32).reshape(4,2)
    if not cv2.isContourConvex(q) or abs(cv2.contourArea(q))<4:return []
    dest=np.float32([[0,0],[519,0],[519,111],[0,111]])
    plate=cv2.warpPerspective(image,cv2.getPerspectiveTransform(q,dest),(520,112),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    gray=cv2.cvtColor(plate,cv2.COLOR_RGB2GRAY)
    return [(cv2.resize(gray[top:bottom,8:510],(320,48),interpolation=cv2.INTER_AREA),q.copy(),'type1_fullheight_'+str(bottom)) for top,bottom in [(3,105),(6,96)]]
