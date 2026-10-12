#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from benchmark_manifest import build_manifest, save_manifest, default_eval_min_start_goal


def main():
    p = argparse.ArgumentParser(description="Generate one frozen episode manifest shared by all methods.")
    p.add_argument("--dataset", choices=["swissview", "xbd"], required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes-per-pair", type=int, default=5)
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--eval-seed", type=int, default=900000)
    p.add_argument("--min-start-goal-distance", type=float, default=None)
    a = p.parse_args()

    min_sg = a.min_start_goal_distance
    if min_sg is None:
        min_sg = default_eval_min_start_goal(a.dataset)
    m = build_manifest(a.dataset, a.dataset_root, a.cache_root,
                       a.episodes_per_pair, a.split_seed, a.eval_seed, min_sg)
    save_manifest(a.out, m)
    print(f"Saved: {a.out}")
    print(f"dataset={m['dataset']} test_scenes={len(m['test_scenes'])} "
          f"test_pairs={m['num_test_pairs']} episodes={m['num_episodes']} "
          f"min_sg={m['min_start_goal_distance']}")


if __name__ == "__main__":
    main()
