from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import torch

class AnyLocQueryEncoder:
    """
    Compute ONE AnyLoc-VLAD descriptor from an actual camera crop.

    This follows the official AnyLoc demo API:
      DinoV2ExtractFeatures("dinov2_vitg14", 31, "value")
      + VLAD with c32 aerial vocabulary.

    The encoder is initialized lazily because model/vocabulary loading is much
    more expensive than the subsequent single-query embedding.
    """
    def __init__(self, anyloc_root="/workspace/AnyLoc", device="cuda",
                 domain="aerial", num_clusters=32):
        self.anyloc_root=Path(anyloc_root)
        self.device=torch.device(device if torch.cuda.is_available() else "cpu")
        self.domain=str(domain)
        self.num_clusters=int(num_clusters)
        self._ready=False

    def _find_demo_dir(self):
        candidates=[
            self.anyloc_root/"demo",
            self.anyloc_root,
        ]
        for d in candidates:
            if (d/"utilities.py").is_file():
                return d
        raise RuntimeError(
            f"Could not find AnyLoc utilities.py under {self.anyloc_root}. "
            "Expected /workspace/AnyLoc/demo/utilities.py or equivalent."
        )

    def _find_centers(self, demo_dir):
        rel=Path("vocabulary")/"dinov2_vitg14"/f"l31_value_c{self.num_clusters}"/self.domain/"c_centers.pt"
        candidates=[
            demo_dir/"cache"/rel,
            self.anyloc_root/"cache"/rel,
        ]
        for p in candidates:
            if p.is_file():
                return p

        # Last-resort recursive search, constrained to the exact filename/tree.
        hits=list(self.anyloc_root.glob(
            f"**/vocabulary/dinov2_vitg14/l31_value_c{self.num_clusters}/{self.domain}/c_centers.pt"
        ))
        if hits:
            return hits[0]

        raise RuntimeError(
            "Could not find AnyLoc aerial VLAD vocabulary c_centers.pt. "
            f"Searched under {self.anyloc_root}."
        )

    def _setup(self):
        if self._ready:
            return
        demo_dir=self._find_demo_dir()
        if str(demo_dir) not in sys.path:
            sys.path.insert(0,str(demo_dir))

        try:
            from utilities import DinoV2ExtractFeatures, VLAD
        except Exception as e:
            raise RuntimeError(
                f"Failed to import AnyLoc demo utilities from {demo_dir}: {e}"
            ) from e

        from torchvision import transforms as tvf

        self.tvf=tvf
        self.extractor=DinoV2ExtractFeatures(
            "dinov2_vitg14",31,"value",device=self.device
        )

        centers=self._find_centers(demo_dir)
        self.vlad=VLAD(
            self.num_clusters,
            desc_dim=None,
            cache_dir=str(centers.parent),
        )
        self.vlad.fit(None)

        self.base_tf=tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225],
            ),
        ])
        self._ready=True

    @torch.no_grad()
    def encode(self, gray_or_rgb):
        self._setup()
        from PIL import Image

        arr=np.asarray(gray_or_rgb)
        if arr.ndim==2:
            arr=np.repeat(arr[...,None],3,axis=2)
        elif arr.ndim==3 and arr.shape[2]>3:
            arr=arr[:,:,:3]
        if arr.dtype!=np.uint8:
            x=arr.astype(np.float32)
            lo,hi=np.percentile(x,[1,99])
            if hi<=lo:
                hi=lo+1.0
            arr=(255*np.clip((x-lo)/(hi-lo),0,1)).astype(np.uint8)

        pil=Image.fromarray(arr).convert("RGB")
        img=self.base_tf(pil).to(self.device)

        # DINOv2 ViT-G/14 requires dimensions divisible by 14.
        _,h,w=img.shape
        hn=(h//14)*14
        wn=(w//14)*14
        img=self.tvf.CenterCrop((hn,wn))(img)[None,...]

        local=self.extractor(img)              # [1, patches, dim]
        gd=self.vlad.generate(local.cpu().squeeze(0))
        q=gd.detach().cpu().numpy().astype(np.float32)
        q/=max(float(np.linalg.norm(q)),1e-12)
        return q
