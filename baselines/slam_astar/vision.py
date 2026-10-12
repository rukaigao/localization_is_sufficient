from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
import math
import cv2
import numpy as np

@lru_cache(maxsize=8)
def load_gray(path):
    im=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError(f"Could not read image: {path}")
    if im.ndim==3:
        if im.shape[2]>=3:
            im=cv2.cvtColor(im[:,:,:3],cv2.COLOR_BGR2GRAY)
        else:
            im=im[:,:,0]
    if im.dtype!=np.uint8:
        x=im.astype(np.float32)
        lo,hi=np.percentile(x,[1,99])
        if hi<=lo: hi=lo+1
        x=np.clip((x-lo)/(hi-lo),0,1)
        im=(255*x).astype(np.uint8)
    return im

def crop_center(image,center,size=512):
    """
    Extract a north-aligned square crop centered at continuous map/image xy.
    Reflect padding allows the simulator to retain [0,10000]^2 boundary
    semantics used by the RL baselines.
    """
    x,y=float(center[0]),float(center[1])
    half=size//2
    cx=int(round(x)); cy=int(round(y))
    x0=cx-half; y0=cy-half; x1=x0+size; y1=y0+size

    pl=max(0,-x0); pt=max(0,-y0)
    pr=max(0,x1-image.shape[1]); pb=max(0,y1-image.shape[0])
    if pl or pt or pr or pb:
        padded=cv2.copyMakeBorder(image,pt,pb,pl,pr,cv2.BORDER_REFLECT_101)
        x0+=pl; x1+=pl; y0+=pt; y1+=pt
        return padded[y0:y1,x0:x1].copy()
    return image[y0:y1,x0:x1].copy()

class SIFTBackend:
    def __init__(self,nfeatures=1800,ratio=0.75,ransac_px=4.0,cache_size=256):
        self.sift=cv2.SIFT_create(nfeatures=int(nfeatures))
        self.matcher=cv2.BFMatcher(cv2.NORM_L2)
        self.ratio=float(ratio)
        self.ransac_px=float(ransac_px)
        self.cache=OrderedDict()
        self.cache_size=int(cache_size)

    def features(self,img,key=None):
        if key is not None and key in self.cache:
            val=self.cache.pop(key); self.cache[key]=val
            return val
        kp,des=self.sift.detectAndCompute(img,None)
        val=(kp,des)
        if key is not None:
            self.cache[key]=val
            while len(self.cache)>self.cache_size:
                self.cache.popitem(last=False)
        return val

    def match_affine(self,img_a,img_b,key_a=None,key_b=None):
        """
        Estimate affine partial transform that maps image-A coordinates -> image-B.
        Returns None on geometric failure.
        """
        ka,da=self.features(img_a,key_a)
        kb,db=self.features(img_b,key_b)
        if da is None or db is None or len(ka)<2 or len(kb)<2:
            return None
        knn=self.matcher.knnMatch(da,db,k=2)
        good=[]
        for pair in knn:
            if len(pair)<2: continue
            m,n=pair
            if m.distance < self.ratio*n.distance:
                good.append(m)
        if len(good)<3:
            return {"matches":len(good),"inliers":0,"inlier_ratio":0.0}

        pa=np.float32([ka[m.queryIdx].pt for m in good])
        pb=np.float32([kb[m.trainIdx].pt for m in good])
        A,mask=cv2.estimateAffinePartial2D(
            pa,pb,method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_px,
            maxIters=2000,confidence=0.995,refineIters=10
        )
        if A is None or mask is None:
            return {"matches":len(good),"inliers":0,"inlier_ratio":0.0}
        inl=int(mask.ravel().sum())
        a,b=float(A[0,0]),float(A[1,0])
        scale=math.sqrt(max(a*a+b*b,1e-12))
        rot=math.degrees(math.atan2(b,a))
        return {
            "A":A.astype(np.float64),
            "matches":len(good),
            "inliers":inl,
            "inlier_ratio":inl/max(len(good),1),
            "scale":scale,
            "rotation_deg":rot,
            "translation":A[:,2].astype(np.float64),
        }
