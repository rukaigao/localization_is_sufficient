#!/usr/bin/env python3
"""Evaluate the online-only PF/MC navigator on SwissView or xBD.

Preferred usage is with a frozen shared episode manifest.  The exact same JSON
can be consumed by DQN, DDPG, SLAM+A*, and this PF/MC method.

Dataset profiles:
  SwissView: 10000x10000, crop 512, stride/action 256, eval min distance 2560.
  xBD:       1024x1024, crop 128, stride/action 64, eval min distance 640.

The sampled true start is simulator/evaluator-only.  It is never supplied to
the online controller, PF update, risk detector, or recovery planner.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from config import apply_dataset_profile
from main import load_config
from online_pipeline import run_online
from reference_map import ReferenceMap, DeploymentMap, load_cache
from benchmark_manifest import (
    build_manifest,
    default_eval_min_start_goal,
    load_manifest,
)


def _validate_cache_profile(cache: dict, dataset: str, path: Path):
    expected = {
        "swissview": (512, 256),
        "xbd": (128, 64),
    }[dataset]
    got = (int(cache.get("crop_size", -1)), int(cache.get("stride", -1)))
    if got != expected:
        raise ValueError(
            f"{path}: cache geometry {got} does not match {dataset} "
            f"profile crop/stride={expected}."
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["swissview", "xbd"], default=None,
                   help="Required without --episodes-manifest; with a manifest it must match.")
    p.add_argument("--dataset-root", type=Path, default=None)
    p.add_argument("--cache-root", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes-per-pair", type=int, default=5)
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--eval-seed", type=int, default=900000)
    p.add_argument("--min-start-goal-distance", type=float, default=None)
    p.add_argument("--episodes-manifest", type=Path, default=None,
                   help="Frozen shared episode JSON; strongly recommended.")
    p.add_argument("--mode", choices=["risk_aware", "goal_only"], default="risk_aware")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--background-tif", action="store_true")
    p.add_argument("--limit-pairs", type=int, default=None)
    p.add_argument("--limit-episodes", type=int, default=None)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.episodes_manifest is not None:
        shared = load_manifest(args.episodes_manifest)
        dataset = str(shared.get("dataset", "")).lower()
        if dataset not in {"swissview", "xbd"}:
            raise ValueError(f"Unsupported dataset in manifest: {dataset!r}")
        if args.dataset is not None and args.dataset != dataset:
            raise ValueError(
                f"--dataset={args.dataset} disagrees with manifest dataset={dataset}"
            )
        episodes = list(shared["episodes"])
        if args.limit_pairs is not None:
            episodes = [e for e in episodes
                        if int(e["pair_index"]) < int(args.limit_pairs)]
        if args.limit_episodes is not None:
            episodes = [e for e in episodes
                        if int(e["episode_index"]) < int(args.limit_episodes)]
        manifest = {
            **shared,
            "mode": args.mode,
            "task_assumption": "unknown_start_known_goal",
            "source_manifest": str(args.episodes_manifest),
        }
    else:
        if args.dataset is None:
            raise ValueError("--dataset is required when --episodes-manifest is not supplied.")
        if args.dataset_root is None or args.cache_root is None:
            raise ValueError("--dataset-root and --cache-root are required without a manifest.")
        dataset = args.dataset
        min_sg = args.min_start_goal_distance
        if min_sg is None:
            min_sg = default_eval_min_start_goal(dataset)
        shared = build_manifest(
            dataset=dataset,
            dataset_root=args.dataset_root,
            cache_root=args.cache_root,
            episodes_per_pair=args.episodes_per_pair,
            split_seed=args.split_seed,
            eval_seed=args.eval_seed,
            min_start_goal_distance=min_sg,
        )
        episodes = list(shared["episodes"])
        if args.limit_pairs is not None:
            episodes = [e for e in episodes
                        if int(e["pair_index"]) < int(args.limit_pairs)]
        if args.limit_episodes is not None:
            episodes = [e for e in episodes
                        if int(e["episode_index"]) < int(args.limit_episodes)]
        manifest = {
            **shared,
            "mode": args.mode,
            "task_assumption": "unknown_start_known_goal",
        }

    cfg = apply_dataset_profile(load_config(args.config), dataset)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    rows = []
    t_all = time.perf_counter()
    cached_pair = None
    ref = dep = None

    for spec in episodes:
        pi = int(spec["pair_index"])
        ei = int(spec["episode_index"])
        seed = int(spec["seed"])
        scene = spec["scene"]
        ref_cache = Path(spec["ref_cache"])
        dep_cache = Path(spec["deployment_cache"])
        ref_img = Path(spec["ref_image"])
        obs_img = Path(spec["obs_image"])

        key = (str(ref_cache), str(dep_cache))
        if key != cached_pair:
            ref_payload = load_cache(ref_cache)
            dep_payload = load_cache(dep_cache)
            _validate_cache_profile(ref_payload, dataset, ref_cache)
            _validate_cache_profile(dep_payload, dataset, dep_cache)
            ref = ReferenceMap(ref_payload, cfg)
            dep = DeploymentMap(dep_payload)
            cached_pair = key

        start = np.asarray(spec["start"], dtype=float)
        goal = np.asarray(spec["goal"], dtype=float)

        ep = args.out / f"{scene}_p{pi:03d}_e{ei:02d}"
        ep.mkdir(parents=True, exist_ok=True)
        ep_cfg = dataclasses.replace(cfg, seed=seed)

        t0 = time.perf_counter()
        result = run_online(
            start,
            goal,
            ref,
            dep,
            ep_cfg,
            ep,
            mode=args.mode,
            verbose=False,
            background_tif=obs_img if args.background_tif else None,
        )

        row = {
            "dataset": dataset,
            "scene": scene,
            "pair_index": pi,
            "episode_index": ei,
            "seed": seed,
            "ref_cache": str(ref_cache),
            "deployment_cache": str(dep_cache),
            "ref_image": str(ref_img),
            "obs_image": str(obs_img),
            "start": start.tolist(),
            "goal": goal.tolist(),
            "initial_goal_distance": float(np.linalg.norm(goal - start)),
            "runtime_s": time.perf_counter() - t0,
            **dataclasses.asdict(result),
        }
        rows.append(row)

        # Persist after every completed episode.
        (args.out / "episodes.json").write_text(json.dumps(rows, indent=2))

    sr = float(np.mean([r["vanilla_success"] for r in rows])) if rows else float("nan")
    summary = {
        "dataset": dataset,
        "episodes": len(rows),
        "success_rate": sr,
        "mean_executed_distance": (
            float(np.mean([r["executed_distance"] for r in rows]))
            if rows else float("nan")
        ),
        "termination_counts": {},
        "wall_time_seconds": time.perf_counter() - t_all,
    }
    for r in rows:
        k = r["termination_reason"]
        summary["termination_counts"][k] = summary["termination_counts"].get(k, 0) + 1

    (args.out / "episodes.json").write_text(json.dumps(rows, indent=2))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== BENCHMARK RESULTS ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
