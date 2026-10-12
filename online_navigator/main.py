"""CLI for the online-only unknown-start GPS-denied navigator.

The task assumption is:
  - start/current position is hidden from the navigation algorithm;
  - the simulator knows the start only to place the vehicle, generate the
    deployment observation stream, and enforce the external evaluation budget;
  - the goal coordinate and historical geo-referenced reference map are known;
  - the PF starts approximately uniform over the whole map.

Example:
  python main.py \
      --reference-cache REF.pt \
      --deployment-cache DEPLOY.pt \
      --start 8119.61 5572.17 \
      --goal 7152.26 4374.66 \
      --out online_run \
      --mode risk_aware \
      --seed 2100003

`--start` is simulator/evaluator-only.  It is never passed to the
controller, PF update, risk detector, or recovery planner.
"""
from __future__ import annotations
import argparse, dataclasses, json
from pathlib import Path
import numpy as np

from config import OnlineConfig, apply_dataset_profile
from online_pipeline import run_online
from reference_map import ReferenceMap, DeploymentMap, load_cache


def load_config(path):
    cfg=OnlineConfig()
    if path is None: return cfg
    with open(path) as f: d=json.load(f)
    return dataclasses.replace(cfg,**d)


def main():
    p=argparse.ArgumentParser(description="Online-only unknown-start GPS-denied navigator")
    p.add_argument("--dataset",choices=["swissview","xbd"],default="swissview",
                   help="dataset geometry/profile; default: swissview")
    p.add_argument("--reference-cache",required=True,type=Path)
    p.add_argument("--deployment-cache",required=True,type=Path,
                   help="deployment-time descriptor cache; GT selects observations only inside the simulator")
    p.add_argument("--start",required=True,type=float,nargs=2,metavar=("X","Y"),
                   help="SIMULATOR-ONLY start position; never exposed to the navigation policy")
    p.add_argument("--goal",required=True,type=float,nargs=2,metavar=("X","Y"),
                   help="known goal coordinate available to the navigator")
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--mode",choices=["risk_aware","goal_only"],default="risk_aware",
                   help="risk_aware enables PF/MC recovery; goal_only always steers from PF robust center toward goal")
    p.add_argument("--background-tif",type=Path,default=None,
                   help="optional deployment .tif used only for GIF rendering")
    p.add_argument("--config",type=Path,default=None)
    p.add_argument("--seed",type=int,default=None)
    args=p.parse_args()

    cfg=apply_dataset_profile(load_config(args.config), args.dataset)
    if args.seed is not None: cfg=dataclasses.replace(cfg,seed=args.seed)
    start=np.asarray(args.start,dtype=np.float64)
    goal=np.asarray(args.goal,dtype=np.float64)

    ref=ReferenceMap(load_cache(args.reference_cache),cfg)
    dep=DeploymentMap(load_cache(args.deployment_cache))
    if not bool(ref.in_bounds(start)):
        raise ValueError(f"--start is outside configured map bounds: {start.tolist()}")
    if not bool(ref.in_bounds(goal)):
        raise ValueError(f"--goal is outside configured map bounds: {goal.tolist()}")

    args.out.mkdir(parents=True,exist_ok=True)
    nominal_distance=float(np.linalg.norm(goal-start))
    with open(args.out/"run_metadata.json","w") as f:
        json.dump({
            "mode":args.mode,
            "dataset":args.dataset,
            "task_assumption":"unknown_start_known_goal",
            "start_simulator_only":start.tolist(),
            "goal_known_to_agent":goal.tolist(),
            "nominal_start_goal_distance_eval_only":nominal_distance,
            "distance_budget_eval_only":float(cfg.online_max_route_length_factor)*nominal_distance,
            "initial_pf":"uniform_full_map",
        },f,indent=2)

    result=run_online(
        start=start,
        goal=goal,
        ref_map=ref,
        deployment_map=dep,
        cfg=cfg,
        out_dir=args.out,
        mode=args.mode,
        verbose=True,
        background_tif=args.background_tif,
    )
    print("\n=== ONLINE RESULT ===")
    print(f"mode={args.mode} task=unknown_start_known_goal")
    print(json.dumps(dataclasses.asdict(result),indent=2))


if __name__=="__main__": main()
