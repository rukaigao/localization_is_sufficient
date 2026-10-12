from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
import itertools
import random
import json
import numpy as np
import torch


@dataclass(frozen=True)
class PairSpec:
    scene: str
    ref: str
    obs: str


@lru_cache(maxsize=None)
def load_cache(path):
    """
    Load each descriptor cache once per process and normalize descriptors once.

    NOTE: This preserves the original RL-baseline coordinate convention:
    cached `positions` are used exactly as saved. This keeps SwissView behavior
    unchanged while allowing the same code to consume xBD's 15x15 caches.
    """
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        d = torch.load(path, map_location="cpu")

    pos = np.asarray(d["positions"], dtype=np.float64)
    x = d["descriptors"]
    if hasattr(x, "float"):
        x = x.float().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    pos.setflags(write=False)
    x.setflags(write=False)
    return pos, x


def _swissview_scene_pool(root: Path) -> list[str]:
    """Preserve the original exact SwissView scene pool."""
    excluded = {"swisstopo_00", "swisstopo_32"}
    scenes = [
        f"swisstopo_{i:02d}"
        for i in range(50)
        if f"swisstopo_{i:02d}" not in excluded
    ]
    missing = [s for s in scenes if not (root / s).is_dir()]
    if missing:
        raise RuntimeError(
            "Missing required scenes from swisstopo_00..swisstopo_49: "
            + ", ".join(missing)
        )
    return scenes


def _xbd_scene_pool(root: Path, cache_root: Path | None = None) -> list[str]:
    """
    Infer the frozen 100-pair xBD selection from the cached scene folders.

    xBD may contain hundreds of valid pair folders, but only the 100 that were
    encoded/cached are trained/evaluated. Inferring the selection from
    cache_root (rather than from a separate list) guarantees that DQN, DDPG,
    SLAM+A*, and PF/MC all share the same scenes whenever they share a
    cache_root.
    """
    if cache_root is None:
        raise ValueError(
            "cache_root is required to infer the frozen 100-pair xBD subset "
            "from cached scene folders."
        )
    cache_root = Path(cache_root)
    scenes = [
        d.name for d in sorted(cache_root.iterdir())
        if d.is_dir() and not d.name.startswith("_") and len(list(d.glob("*.pt"))) >= 2
    ]
    if len(scenes) != 100:
        raise RuntimeError(
            f"Expected 100 cached xBD scene folders in {cache_root}, found {len(scenes)}"
        )
    if len(scenes) != len(set(scenes)):
        raise RuntimeError("Cached xBD scene folders contain duplicate names")
    missing = [s for s in scenes if not (root / s).is_dir()]
    if missing:
        raise RuntimeError(
            "Selected xBD scene folder(s) missing: " + ", ".join(missing[:20])
        )
    return scenes


def scene_split(root, ntrain=40, ntest=10, seed=2026, dataset_kind="swissview", cache_root=None):
    root = Path(root)
    kind = dataset_kind.lower()
    if kind == "swissview":
        scenes = _swissview_scene_pool(root)
    elif kind == "xbd":
        scenes = _xbd_scene_pool(root, cache_root)
    else:
        raise ValueError(f"Unknown dataset kind: {dataset_kind}")

    if ntrain + ntest != len(scenes):
        raise ValueError(
            f"Dataset '{kind}' has {len(scenes)} selected usable scenes; "
            f"got ntrain={ntrain}, ntest={ntest}."
        )

    r = random.Random(seed)
    r.shuffle(scenes)
    return sorted(scenes[:ntrain]), sorted(scenes[ntrain:ntrain + ntest])


def train_val_split(train_side, nval=4, seed=2026):
    xs = list(train_side)
    if nval < 0 or nval >= len(xs):
        raise ValueError(f"Invalid nval={nval} for {len(xs)} training-side scenes")
    r = random.Random(seed + 17)
    r.shuffle(xs)
    return sorted(xs[nval:]), sorted(xs[:nval])


def pairs_for_scene(scene, dataset_root, cache_root):
    sd = Path(dataset_root) / scene
    cd = Path(cache_root) / scene

    if not sd.is_dir():
        raise RuntimeError(f"scene folder not found: {sd}")
    if not cd.is_dir():
        raise RuntimeError(f"cache folder not found: {cd}")

    imgs = [
        p for p in sorted(sd.iterdir())
        if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    ]
    pts = sorted(cd.glob("*.pt"))

    if len(imgs) < 2:
        raise RuntimeError(f"{scene}: <2 images")
    if len(pts) < 2:
        raise RuntimeError(f"{scene}: <2 descriptor caches in {cd}")

    resolved = []
    for im in imgs:
        matches = [p for p in pts if im.stem in p.name]
        if matches:
            resolved.append(max(matches, key=lambda p: len(p.name)))

    if len(set(resolved)) < 2:
        print(f"[data] {scene}: stem matching insufficient; using all .pt caches")
        resolved = pts

    resolved = list(dict.fromkeys(map(str, resolved)))

    # For xBD, each selected folder should contain exactly pre/post -> 2 ordered
    # directions. SwissView may contain 2--4 time points, so retain all ordered
    # permutations exactly as before.
    return [PairSpec(scene, a, b) for a, b in itertools.permutations(resolved, 2)]


def all_pairs(scenes, dataset_root, cache_root):
    out = []
    for s in scenes:
        out += pairs_for_scene(s, dataset_root, cache_root)
    return out


def save_manifest(path, tr, va, te, trp, vap, tep, dataset_kind=None):
    Path(path).write_text(json.dumps({
        "dataset_kind": dataset_kind,
        "train_scenes": tr,
        "val_scenes": va,
        "test_scenes": te,
        "train_pairs": [p.__dict__ for p in trp],
        "val_pairs": [p.__dict__ for p in vap],
        "test_pairs": [p.__dict__ for p in tep],
    }, indent=2))
