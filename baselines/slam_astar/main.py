import argparse, json
from benchmark_manifest import default_eval_min_start_goal
from config import SLAMAStarConfig, apply_dataset_profile
from data import pair_from_cache_paths
from runner import run_episode

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=["swissview","xbd"],default="swissview")
    p.add_argument("--dataset-root",default=None)
    p.add_argument("--reference-cache",required=True)
    p.add_argument("--deployment-cache",required=True)
    p.add_argument("--scene",default=None)
    p.add_argument("--start",nargs=2,type=float,default=None)
    p.add_argument("--goal",nargs=2,type=float,default=None)
    p.add_argument("--min-start-goal-distance",type=float,default=None)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--out",default="slam_astar_run")
    p.add_argument("--anyloc-root",default="/workspace/AnyLoc")
    p.add_argument("--anyloc-device",default="cuda")
    p.add_argument("--quiet",action="store_true")
    a=p.parse_args()
    dataset_root=a.dataset_root or ("/workspace/swissimage" if a.dataset=="swissview" else "/workspace/xBD")
    min_sg=a.min_start_goal_distance if a.min_start_goal_distance is not None else default_eval_min_start_goal(a.dataset)
    cfg=apply_dataset_profile(SLAMAStarConfig(),a.dataset)
    cfg.min_start_goal_px=min_sg; cfg.seed=a.seed; cfg.anyloc_root=a.anyloc_root; cfg.anyloc_device=a.anyloc_device
    pair=pair_from_cache_paths(a.reference_cache,a.deployment_cache,dataset_root,a.scene)
    res=run_episode(pair,cfg,start=a.start,goal=a.goal,seed=a.seed,out_dir=a.out,verbose=not a.quiet)
    print("\n=== SLAM + A* RESULT ==="); print(json.dumps(res,indent=2))

if __name__=="__main__": main()
