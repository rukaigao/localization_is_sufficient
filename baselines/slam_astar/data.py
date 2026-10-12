from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
import itertools, random, json
import numpy as np
import torch

IMAGE_EXTS={".tif",".tiff",".png",".jpg",".jpeg"}

@dataclass(frozen=True)
class PairSpec:
    scene: str
    ref_cache: str
    obs_cache: str
    ref_image: str
    obs_image: str

@lru_cache(maxsize=None)
def load_cache(path):
    try:
        d=torch.load(path,map_location="cpu",weights_only=False)
    except TypeError:
        d=torch.load(path,map_location="cpu")
    pos=np.asarray(d["positions"],dtype=np.float64)
    desc=d["descriptors"]
    if hasattr(desc,"float"):
        desc=desc.float().cpu().numpy()
    desc=np.asarray(desc,dtype=np.float32)
    desc=desc/np.maximum(np.linalg.norm(desc,axis=1,keepdims=True),1e-12)
    pos.setflags(write=False); desc.setflags(write=False)
    return pos,desc

def _images(scene_dir):
    return [p for p in sorted(Path(scene_dir).iterdir()) if p.suffix.lower() in IMAGE_EXTS]

def resolve_image_for_cache(cache_path, scene_dir):
    cp=Path(cache_path); imgs=_images(scene_dir)
    if not imgs: raise RuntimeError(f"No images found in {scene_dir}")
    hits=[im for im in imgs if im.stem in cp.name]
    if hits: return str(max(hits,key=lambda p:len(p.stem)))
    hits=[im for im in imgs if cp.stem in im.stem or im.stem in cp.stem]
    if hits: return str(max(hits,key=lambda p:len(p.stem)))
    raise RuntimeError(f"Could not associate cache {cache_path} with a raw image in {scene_dir}")


def pair_from_cache_paths(ref_cache,obs_cache,dataset_root,scene=None):
    ref_cache=Path(ref_cache); obs_cache=Path(obs_cache)
    if scene is None:
        scene=ref_cache.parent.name
    sd=Path(dataset_root)/scene
    return PairSpec(
        scene=scene,
        ref_cache=str(ref_cache),
        obs_cache=str(obs_cache),
        ref_image=resolve_image_for_cache(ref_cache,sd),
        obs_image=resolve_image_for_cache(obs_cache,sd),
    )

def _swissview_scene_pool(root):
    root=Path(root); excluded={"swisstopo_00","swisstopo_32"}
    scenes=[f"swisstopo_{i:02d}" for i in range(50) if f"swisstopo_{i:02d}" not in excluded]
    missing=[s for s in scenes if not (root/s).is_dir()]
    if missing: raise RuntimeError("Missing scenes: "+", ".join(missing))
    return scenes

def _xbd_scene_pool(root, cache_root=None):
    # The frozen 100-pair xBD selection is inferred from the cached scene
    # folders under cache_root, so it always matches whatever was encoded.
    root=Path(root)
    if cache_root is None:
        raise ValueError("cache_root is required to infer the frozen 100-scene xBD subset.")
    cache_root=Path(cache_root)
    scenes=[]
    for d in sorted(cache_root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        if len(list(d.glob("*.pt"))) < 2:
            continue
        if not (root/d.name).is_dir():
            raise RuntimeError(f"xBD cache scene exists but raw scene folder is missing: {d.name}")
        scenes.append(d.name)
    if len(scenes) != 100:
        raise RuntimeError(
            f"Expected exactly 100 cached xBD scene pairs, found {len(scenes)} in {cache_root}."
        )
    if len(scenes)!=len(set(scenes)):
        raise RuntimeError("Cached xBD scene folders contain duplicate names")
    missing=[s for s in scenes if not (root/s).is_dir()]
    if missing: raise RuntimeError("Selected xBD scene folder(s) missing: "+", ".join(missing[:20]))
    return scenes

def scene_split(dataset_root,ntrain,ntest,seed=2026,dataset_kind="swissview",cache_root=None):
    kind=dataset_kind.lower()
    if kind=="swissview": scenes=_swissview_scene_pool(dataset_root)
    elif kind=="xbd": scenes=_xbd_scene_pool(dataset_root,cache_root=cache_root)
    else: raise ValueError(dataset_kind)
    if ntrain+ntest!=len(scenes):
        raise ValueError(f"Dataset {kind!r} has {len(scenes)} scenes; got {ntrain}+{ntest}")
    r=random.Random(seed); r.shuffle(scenes)
    return sorted(scenes[:ntrain]), sorted(scenes[ntrain:ntrain+ntest])

def pairs_for_scene(scene,dataset_root,cache_root):
    sd=Path(dataset_root)/scene; cd=Path(cache_root)/scene
    if not sd.is_dir(): raise RuntimeError(f"scene folder not found: {sd}")
    if not cd.is_dir(): raise RuntimeError(f"cache folder not found: {cd}")
    caches=sorted(cd.glob("*.pt"))
    if len(caches)<2: raise RuntimeError(f"{scene}: need >=2 caches in {cd}")
    resolved=[]
    for c in caches:
        try: resolved.append((str(c),resolve_image_for_cache(c,sd)))
        except RuntimeError: pass
    uniq=[]; seen=set()
    for x in resolved:
        if x[0] not in seen: uniq.append(x); seen.add(x[0])
    if len(uniq)<2: raise RuntimeError(f"{scene}: could not resolve >=2 cache/image pairs")
    out=[]
    for (rc,ri),(oc,oi) in itertools.permutations(uniq,2):
        out.append(PairSpec(scene,rc,oc,ri,oi))
    return out

def all_pairs(scenes,dataset_root,cache_root):
    out=[]
    for s in scenes: out.extend(pairs_for_scene(s,dataset_root,cache_root))
    return out

def sample_start_goal(seed,map_min,map_max,min_start_goal_px):
    """Exact same sampling logic/RNG as RL VisualNavEnv._sample()."""
    rng=np.random.default_rng(int(seed))
    for _ in range(10000):
        s=rng.uniform(map_min,map_max,2)
        g=rng.uniform(map_min,map_max,2)
        if np.linalg.norm(g-s)>=min_start_goal_px:
            return s,g
    raise RuntimeError("could not sample start/goal")

def save_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2))
