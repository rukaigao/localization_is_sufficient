"""Evaluate DQN/DDPG on one frozen shared episode manifest.

If --episodes-manifest is supplied, scene/pair/start/goal/seed definitions are
read verbatim from that file. This is the recommended paper-evaluation path.
Without it, a manifest is generated deterministically from the CLI settings.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from benchmark_manifest import build_manifest, load_manifest, save_manifest, default_eval_min_start_goal
from config import Common, apply_dataset_profile
from data import PairSpec
from env import VisualNavEnv
from models import DQN, Actor
from rl_utils import one


def bootstrap_ci(values, n_boot=2000, seed=0, alpha=0.05):
    values=np.asarray(values,dtype=np.float64)
    rng=np.random.default_rng(seed); n=len(values); boots=np.empty(n_boot)
    for i in range(n_boot):
        idx=rng.integers(0,n,n); boots[i]=values[idx].mean()
    lo,hi=np.quantile(boots,[alpha/2,1-alpha/2])
    return float(values.mean()),float(lo),float(hi)


def spl_terms(success,executed,initial_goal_dist):
    s=np.asarray(success,float); p=np.asarray(executed,float); l=np.asarray(initial_goal_dist,float)
    return s*l/np.maximum(p,l)


def sct_terms(success,steps,initial_goal_dist,step_px):
    s=np.asarray(success,float); c=np.asarray(steps,float); t=np.asarray(initial_goal_dist,float)/step_px
    return s*t/np.maximum(c,t)


def run_episode(spec,cfg,continuous,act_fn,device):
    pair=PairSpec(spec["scene"],spec["ref_cache"],spec["deployment_cache"])
    seed=int(spec["seed"])
    env=VisualNavEnv(pair,cfg,continuous=continuous,seed=seed)
    start=np.asarray(spec["start"],dtype=float); goal=np.asarray(spec["goal"],dtype=float)
    o,_=env.reset(start=start,goal=goal)
    done=False; step_info=None
    while not done:
        x=one(o,device); a=act_fn(x); o,_,done,step_info=env.step(a)
    return {
        "scene":spec["scene"],"pair_index":int(spec["pair_index"]),
        "episode_index":int(spec["episode_index"]),"seed":seed,
        "ref_cache":spec["ref_cache"],"deployment_cache":spec["deployment_cache"],
        "start":spec["start"],"goal":spec["goal"],
        "success":bool(step_info["success"]),
        "boundary_failure":bool(step_info.get("boundary",False)),
        "steps":int(env.steps),"executed_distance":float(env.dist),
        "initial_goal_distance":float(np.linalg.norm(goal-start)),
        "final_goal_distance":step_info.get("goal_distance",None),
    }


@torch.no_grad()
def evaluate_dqn(episodes,cfg,ckpt_path,device):
    q=DQN(cfg.feature_dim).to(device); ck=torch.load(ckpt_path,map_location=device)
    q.load_state_dict(ck["online"]); q.eval()
    return [run_episode(e,cfg,False,lambda x:int(q(x).argmax(-1).item()),device) for e in episodes]


@torch.no_grad()
def evaluate_ddpg(episodes,cfg,ckpt_path,device):
    actor=Actor(cfg.feature_dim).to(device); ck=torch.load(ckpt_path,map_location=device)
    actor.load_state_dict(ck["actor"]); actor.eval()
    return [run_episode(e,cfg,True,lambda x:float(actor(x).item()),device) for e in episodes]


def summarize(name,records,step_px):
    success=[r["success"] for r in records]; executed=[r["executed_distance"] for r in records]
    initial=[r["initial_goal_distance"] for r in records]; steps=[r["steps"] for r in records]
    sr=bootstrap_ci(success); spl=bootstrap_ci(spl_terms(success,executed,initial)); sct=bootstrap_ci(sct_terms(success,steps,initial,step_px))
    result={"n":len(records),"SR":sr,"SPL":spl,"SCT":sct}
    print(f"\n=== {name} (n={len(records)}) ===")
    print(f"SR:  {sr[0]:.3f}  [{sr[1]:.3f}, {sr[2]:.3f}]")
    print(f"SPL: {spl[0]:.3f}  [{spl[1]:.3f}, {spl[2]:.3f}]")
    print(f"SCT: {sct[0]:.3f}  [{sct[1]:.3f}, {sct[2]:.3f}]")
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",choices=["swissview","xbd"],default="swissview")
    ap.add_argument("--dataset-root",default=None); ap.add_argument("--cache-root",default=None)
    ap.add_argument("--episodes-manifest",type=Path,default=None)
    ap.add_argument("--dqn-checkpoint",default="runs/double_dqn/best.pt")
    ap.add_argument("--ddpg-checkpoint",default="runs/ddpg/best.pt")
    ap.add_argument("--out-dir",default="runs/eval")
    ap.add_argument("--split-seed",type=int,default=2026); ap.add_argument("--eval-seed",type=int,default=900000)
    ap.add_argument("--episodes-per-pair",type=int,default=5)
    ap.add_argument("--min-start-goal-distance",type=float,default=None)
    ap.add_argument("--device",default="cuda")
    a=ap.parse_args()

    dataset_root=a.dataset_root or ("/workspace/swissimage" if a.dataset=="swissview" else "/workspace/xBD")
    cache_root=a.cache_root or ("/workspace/AnyLoc/outputs" if a.dataset=="swissview" else "/workspace/AnyLoc/outputs_xbd")
    min_sg=a.min_start_goal_distance if a.min_start_goal_distance is not None else default_eval_min_start_goal(a.dataset)

    if a.episodes_manifest is not None:
        manifest=load_manifest(a.episodes_manifest)
        if manifest["dataset"] != a.dataset:
            raise ValueError(f"Manifest dataset={manifest['dataset']} but --dataset={a.dataset}")
    else:
        manifest=build_manifest(a.dataset,Path(dataset_root),Path(cache_root),a.episodes_per_pair,a.split_seed,a.eval_seed,min_sg)

    c=apply_dataset_profile(Common(dataset_kind=a.dataset,dataset_root=dataset_root,cache_root=cache_root,min_start_goal_px=float(manifest["min_start_goal_distance"])),a.dataset)
    c.min_start_goal_px=float(manifest["min_start_goal_distance"])
    device=torch.device(a.device if torch.cuda.is_available() else "cpu")
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    save_manifest(out/"episodes_manifest_used.json",manifest)

    print(f"dataset={a.dataset} map={c.map_max:g} grid={c.grid_h}x{c.grid_w} step={c.step_px:g}")
    print(f"shared manifest: pairs={manifest['num_test_pairs']} episodes={manifest['num_episodes']} min_sg={manifest['min_start_goal_distance']}")

    t=time.time(); dqn=evaluate_dqn(manifest["episodes"],c,a.dqn_checkpoint,device)
    (out/"dqn_episodes.json").write_text(json.dumps(dqn,indent=2)); print(f"DQN evaluation: {time.time()-t:.1f}s")
    t=time.time(); ddpg=evaluate_ddpg(manifest["episodes"],c,a.ddpg_checkpoint,device)
    (out/"ddpg_episodes.json").write_text(json.dumps(ddpg,indent=2)); print(f"DDPG evaluation: {time.time()-t:.1f}s")

    summary={"DQN":summarize("DQN",dqn,c.step_px),"DDPG":summarize("DDPG",ddpg,c.step_px)}
    (out/"summary.json").write_text(json.dumps(summary,indent=2))

if __name__=="__main__": main()
