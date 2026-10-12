"""Shared deterministic benchmark episode manifest utilities.

The manifest freezes everything that must be identical across methods:
  * test scenes
  * ordered reference/deployment image-cache pairs
  * pair indices
  * episode indices / seeds
  * true simulator start and known goal coordinates

SwissView geometry: 10000x10000, stride/action 256 px.
xBD geometry:      1024x1024, stride/action 64 px.
"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def dataset_geometry(dataset: str):
    kind = dataset.lower()
    if kind == "swissview":
        return dict(map_min=0.0, map_max=10000.0, step_px=256.0,
                    grid_h=38, grid_w=38, crop_size=512, stride=256,
                    train_side=38, test_scenes=10, val_scenes=4)
    if kind == "xbd":
        return dict(map_min=0.0, map_max=1024.0, step_px=64.0,
                    grid_h=15, grid_w=15, crop_size=128, stride=64,
                    train_side=80, test_scenes=20, val_scenes=20)
    raise ValueError(dataset)


def default_eval_min_start_goal(dataset: str) -> float:
    # Preserve the same minimum task length in nominal action steps:
    # SwissView: 2560 / 256 = 10 steps; xBD: 640 / 64 = 10 steps.
    g = dataset_geometry(dataset)
    return 10.0 * g["step_px"]


def _images(scene_dir: Path):
    return [p for p in sorted(scene_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def resolve_image_for_cache(cache_path: Path, scene_dir: Path) -> Path:
    imgs = _images(scene_dir)
    if not imgs:
        raise RuntimeError(f"No raw images found in {scene_dir}")
    hits = [im for im in imgs if im.stem in cache_path.name]
    if hits:
        return max(hits, key=lambda p: len(p.stem))
    hits = [im for im in imgs
            if cache_path.stem in im.stem or im.stem in cache_path.stem]
    if hits:
        return max(hits, key=lambda p: len(p.stem))
    raise RuntimeError(f"Could not associate cache {cache_path} with an image in {scene_dir}")


def scene_pool(dataset: str, dataset_root: Path, cache_root: Path) -> list[str]:
    kind = dataset.lower()
    dataset_root = Path(dataset_root)
    cache_root = Path(cache_root)
    if kind == "swissview":
        excluded = {"swisstopo_00", "swisstopo_32"}
        scenes = [f"swisstopo_{i:02d}" for i in range(50)
                  if f"swisstopo_{i:02d}" not in excluded]
        missing = [s for s in scenes if not (dataset_root / s).is_dir()]
        if missing:
            raise RuntimeError("Missing required SwissView scenes: " + ", ".join(missing))
        return scenes

    if kind == "xbd":
        # The 100 selected xBD pairs are exactly the scene directories that
        # were cached under cache_root.
        scenes = []
        for d in sorted(cache_root.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            if len(list(d.glob("*.pt"))) < 2:
                continue
            if not (dataset_root / d.name).is_dir():
                raise RuntimeError(
                    f"xBD cache scene exists but raw scene folder is missing: {d.name}"
                )
            scenes.append(d.name)
        if len(scenes) != 100:
            raise RuntimeError(
                f"Expected exactly 100 cached xBD scene pairs, found {len(scenes)} in {cache_root}. "
                "The benchmark must use the frozen 100-pair cache subset."
            )
        return scenes
    raise ValueError(dataset)


def split_scenes(dataset: str, dataset_root: Path, cache_root: Path, split_seed: int):
    scenes = scene_pool(dataset, dataset_root, cache_root)
    g = dataset_geometry(dataset)
    if g["train_side"] + g["test_scenes"] != len(scenes):
        raise RuntimeError("Scene-count/profile mismatch")
    r = random.Random(int(split_seed))
    r.shuffle(scenes)
    return sorted(scenes[:g["train_side"]]), sorted(scenes[g["train_side"]:])


def ordered_pairs_for_scene(scene: str, dataset_root: Path, cache_root: Path):
    sd = Path(dataset_root) / scene
    cd = Path(cache_root) / scene
    if not sd.is_dir():
        raise RuntimeError(f"Scene folder not found: {sd}")
    if not cd.is_dir():
        raise RuntimeError(f"Cache folder not found: {cd}")

    resolved = []
    for c in sorted(cd.glob("*.pt")):
        try:
            resolved.append((c, resolve_image_for_cache(c, sd)))
        except RuntimeError:
            pass

    uniq = []
    seen = set()
    for c, im in resolved:
        key = str(c.resolve())
        if key not in seen:
            uniq.append((c, im))
            seen.add(key)
    if len(uniq) < 2:
        raise RuntimeError(f"{scene}: need >=2 resolved cache/image pairs")

    return [
        dict(scene=scene,
             ref_cache=str(a[0].resolve()), deployment_cache=str(b[0].resolve()),
             ref_image=str(a[1].resolve()), obs_image=str(b[1].resolve()))
        for a, b in itertools.permutations(uniq, 2)
    ]


def sample_start_goal(seed: int, min_dist: float, map_min: float, map_max: float):
    rng = np.random.default_rng(int(seed))
    for _ in range(10000):
        start = rng.uniform(map_min, map_max, 2)
        goal = rng.uniform(map_min, map_max, 2)
        if float(np.linalg.norm(goal - start)) >= float(min_dist):
            return start.astype(float), goal.astype(float)
    raise RuntimeError("Could not sample start/goal")


def build_manifest(dataset: str, dataset_root: Path, cache_root: Path,
                   episodes_per_pair: int = 5, split_seed: int = 2026,
                   eval_seed: int = 900000, min_start_goal_distance: float | None = None):
    dataset = dataset.lower()
    g = dataset_geometry(dataset)
    if min_start_goal_distance is None:
        min_start_goal_distance = default_eval_min_start_goal(dataset)

    train_side, test_scenes = split_scenes(dataset, dataset_root, cache_root, split_seed)
    pairs = []
    for scene in test_scenes:
        pairs.extend(ordered_pairs_for_scene(scene, dataset_root, cache_root))

    episodes = []
    for pair_index, pair in enumerate(pairs):
        for episode_index in range(int(episodes_per_pair)):
            seed = int(eval_seed) + pair_index * 100_000 + episode_index
            start, goal = sample_start_goal(
                seed, min_start_goal_distance, g["map_min"], g["map_max"]
            )
            episodes.append({
                **pair,
                "pair_index": pair_index,
                "episode_index": episode_index,
                "seed": seed,
                "start": start.tolist(),
                "goal": goal.tolist(),
                "initial_goal_distance": float(np.linalg.norm(goal - start)),
            })

    return {
        "schema": "shared_navigation_benchmark_v1",
        "dataset": dataset,
        "dataset_root": str(Path(dataset_root).resolve()),
        "cache_root": str(Path(cache_root).resolve()),
        "geometry": g,
        "split_seed": int(split_seed),
        "eval_seed": int(eval_seed),
        "episodes_per_pair": int(episodes_per_pair),
        "min_start_goal_distance": float(min_start_goal_distance),
        "train_side_scenes": train_side,
        "test_scenes": test_scenes,
        "num_test_pairs": len(pairs),
        "num_episodes": len(episodes),
        "episodes": episodes,
    }


def save_manifest(path: Path, manifest: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=2))


def load_manifest(path: Path):
    return json.loads(Path(path).read_text())
