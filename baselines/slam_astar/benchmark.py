import argparse, json, time
from pathlib import Path
import numpy as np

from config import SLAMAStarConfig, apply_dataset_profile
from data import save_json, load_cache, PairSpec
from runner import run_episode
from anyloc_query import AnyLocQueryEncoder
from vision import SIFTBackend, load_gray
from benchmark_manifest import build_manifest, load_manifest, default_eval_min_start_goal


def spl_terms(rows):
    vals=[]
    for r in rows:
        s=float(bool(r["mission_success"]))
        l=float(r["initial_goal_distance"]); p=float(r["executed_distance"])
        vals.append(s*l/max(p,l,1e-12))
    return np.asarray(vals,dtype=np.float64)


def sct_terms(rows,step_px):
    vals=[]
    for r in rows:
        s=float(bool(r["mission_success"]))
        t=float(r["initial_goal_distance"])/float(step_px)
        c=float(r["steps"])
        vals.append(s*t/max(c,t,1e-12))
    return np.asarray(vals,dtype=np.float64)


def bootstrap_ci(values,n_boot=2000,seed=0,alpha=0.05):
    x=np.asarray(values,dtype=np.float64); rng=np.random.default_rng(seed); n=len(x)
    boots=np.empty(n_boot,dtype=np.float64)
    for i in range(n_boot):
        boots[i]=x[rng.integers(0,n,n)].mean()
    lo,hi=np.quantile(boots,[alpha/2,1-alpha/2])
    return [float(x.mean()),float(lo),float(hi)]


def _validate_shared_manifest(m,a,cfg):
    if m.get("schema") != "shared_navigation_benchmark_v1":
        raise ValueError("Unsupported --episodes-manifest schema")
    if m.get("dataset") != a.dataset:
        raise ValueError(f"Manifest dataset={m.get('dataset')!r} but --dataset={a.dataset!r}")
    g=m.get("geometry",{})
    checks={"map_max":cfg.map_max,"step_px":cfg.step_px,"grid_h":cfg.grid_h,"grid_w":cfg.grid_w,"crop_size":cfg.crop_size}
    for k,v in checks.items():
        if k in g and abs(float(g[k])-float(v))>1e-9:
            raise ValueError(f"Manifest geometry mismatch for {k}: {g[k]} vs config {v}")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=["swissview","xbd"],default="swissview")
    p.add_argument("--dataset-root",default=None)
    p.add_argument("--cache-root",default=None)
    p.add_argument("--episodes-manifest",default=None,
                   help="Frozen shared_navigation_benchmark_v1 JSON. When supplied, scene/pair/start/goal/seed settings come exactly from this file.")
    p.add_argument("--out",default="slam_astar_benchmark")
    p.add_argument("--episodes-per-pair",type=int,default=10)
    p.add_argument("--min-start-goal-distance",type=float,default=None)
    p.add_argument("--split-seed",type=int,default=2026)
    p.add_argument("--eval-seed",type=int,default=900000)
    p.add_argument("--anyloc-root",default="/workspace/AnyLoc")
    p.add_argument("--anyloc-device",default="cuda")
    p.add_argument("--reference-feature-cache-size",type=int,default=512)
    p.add_argument("--progress-every",type=int,default=0)
    p.add_argument("--save-plots",action="store_true")
    p.add_argument("--disable-relocalization-fallback",action="store_true")
    p.add_argument("--relocalization-patience",type=int,default=3)
    a=p.parse_args()

    dataset_root=a.dataset_root or ("/workspace/swissimage" if a.dataset=="swissview" else "/workspace/xBD")
    cache_root=a.cache_root or ("/workspace/AnyLoc/outputs" if a.dataset=="swissview" else "/workspace/AnyLoc/outputs_xbd")

    cfg=apply_dataset_profile(SLAMAStarConfig(),a.dataset)
    cfg.anyloc_root=a.anyloc_root
    cfg.anyloc_device=a.anyloc_device
    cfg.enable_relocalization_fallback=not a.disable_relocalization_fallback
    cfg.relocalization_patience=a.relocalization_patience
    cfg.reference_feature_cache_size=a.reference_feature_cache_size

    if a.episodes_manifest:
        manifest=load_manifest(Path(a.episodes_manifest))
        _validate_shared_manifest(manifest,a,cfg)
    else:
        # Same manifest generate_benchmark_episodes.py would produce for these
        # arguments, so the no-manifest default path is exactly the shared
        # fairness contract, not a separately maintained one.
        min_sg=a.min_start_goal_distance if a.min_start_goal_distance is not None else default_eval_min_start_goal(a.dataset)
        manifest=build_manifest(a.dataset,Path(dataset_root),Path(cache_root),a.episodes_per_pair,a.split_seed,a.eval_seed,min_sg)

    episodes=list(manifest["episodes"])
    cfg.min_start_goal_px=float(manifest["min_start_goal_distance"])

    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    save_json(out/"manifest.json",manifest)

    print(f"[benchmark] dataset={a.dataset} episodes={len(episodes)} min_sg={cfg.min_start_goal_px:g}")
    if a.episodes_manifest:
        print(f"[benchmark] shared episodes: {a.episodes_manifest}")

    query_encoder=AnyLocQueryEncoder(
        anyloc_root=cfg.anyloc_root,device=cfg.anyloc_device,
        domain=cfg.anyloc_domain,num_clusters=cfg.anyloc_num_clusters,
    )

    rows=[]; active_scene=None; image_cache={}; descriptor_cache={}; vision=None
    for run_idx,e in enumerate(episodes, start=1):
        scene=e["scene"]
        pair=PairSpec(
            scene=scene,
            ref_cache=e["ref_cache"],
            obs_cache=e.get("deployment_cache",e.get("obs_cache")),
            ref_image=e["ref_image"],
            obs_image=e["obs_image"],
        )
        if pair.obs_cache is None:
            raise ValueError(f"Episode is missing deployment_cache/obs_cache: {e}")

        if scene != active_scene:
            active_scene=scene; image_cache.clear(); descriptor_cache.clear()
            vision=SIFTBackend(
                nfeatures=cfg.sift_nfeatures,ratio=cfg.ratio_test,
                ransac_px=cfg.ransac_reproj_px,
                cache_size=max(1,int(a.reference_feature_cache_size)),
            )

        if pair.obs_image not in image_cache: image_cache[pair.obs_image]=load_gray(pair.obs_image)
        if pair.ref_image not in image_cache: image_cache[pair.ref_image]=load_gray(pair.ref_image)
        if pair.ref_cache not in descriptor_cache: descriptor_cache[pair.ref_cache]=load_cache(pair.ref_cache)

        start=np.asarray(e["start"],dtype=np.float64)
        goal=np.asarray(e["goal"],dtype=np.float64)
        seed=int(e["seed"])
        pi=int(e["pair_index"]); ei=int(e["episode_index"])
        epout=out/f"{scene}_p{pi:03d}_e{ei:02d}"

        # IMPORTANT: pass the frozen start/goal explicitly. The simulator still
        # uses seed for identical motion/wind stochasticity; no resampling occurs.
        r=run_episode(
            pair,cfg,start=start,goal=goal,seed=seed,out_dir=epout,verbose=False,
            query_encoder=query_encoder,vision=vision,save_plot=a.save_plots,
            obs_image=image_cache[pair.obs_image],ref_image=image_cache[pair.ref_image],
            ref_cache_data=descriptor_cache[pair.ref_cache],progress_every=max(0,int(a.progress_every)),
        )
        if not (np.allclose(r["start"],start,atol=1e-9) and np.allclose(r["goal"],goal,atol=1e-9)):
            raise RuntimeError(f"Frozen episode mismatch at pair={pi} ep={ei}")
        r.update({"scene":scene,"pair_index":pi,"episode_index":ei,"seed":seed})
        rows.append(r)
        save_json(out/"episodes.json",rows)

    success=np.asarray([r["mission_success"] for r in rows],dtype=np.float64)
    summary={
        "episodes":len(rows),
        "success_rate":float(success.mean()),
        "SR_95CI":bootstrap_ci(success,seed=0),
        "SPL":bootstrap_ci(spl_terms(rows),seed=1),
        "SCT":bootstrap_ci(sct_terms(rows,cfg.step_px),seed=2),
        "boundary_failure_rate":float(np.mean([r["boundary_failure"] for r in rows])),
        "mean_final_slam_error_px":float(np.mean([r["final_slam_error_px"] for r in rows])),
        "median_final_slam_error_px":float(np.median([r["final_slam_error_px"] for r in rows])),
        "mean_executed_distance":float(np.mean([r["executed_distance"] for r in rows])),
        "mean_map_closures":float(np.mean([r["map_closures"] for r in rows])),
        "mean_initialization_seconds":float(np.mean([r["initialization_seconds"] for r in rows])),
        "mean_episode_seconds":float(np.mean([r["episode_seconds"] for r in rows])),
    }
    save_json(out/"episodes.json",rows); save_json(out/"summary.json",summary)
    print("\n=== SLAM + A* BENCHMARK RESULTS ==="); print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
