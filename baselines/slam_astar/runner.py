import json, math, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from simulator import NavigationSimulator
from slam_astar import SLAMAStarNavigator

def run_episode(
    pair,cfg,start=None,goal=None,seed=0,out_dir=None,verbose=True,
    query_encoder=None,vision=None,save_plot=True,
    obs_image=None,ref_image=None,ref_cache_data=None,
    progress_every=0,
):
    episode_t0=time.perf_counter()
    sim=NavigationSimulator(pair,cfg,seed,obs_image=obs_image)
    first=sim.reset(start=start,goal=goal)
    start=sim.p.copy(); goal=sim.goal.copy()

    nav=SLAMAStarNavigator(
        pair,cfg,query_encoder=query_encoder,vision=vision,
        ref_image=ref_image,ref_cache_data=ref_cache_data,
    )
    est=nav.reset(start,goal,first)
    init_seconds=time.perf_counter()-episode_t0
    if progress_every:
        print(f"  [timing] initialization={init_seconds:.1f}s")

    true_traj=[sim.p.copy()]
    est_traj=[est.copy()]
    errors=[float(np.linalg.norm(est-sim.p))]
    boundary=False
    success=False

    if verbose:
        print(f"[SLAM+A*] start={start.tolist()} goal={goal.tolist()} "
              f"initial_distance={np.linalg.norm(goal-start):.1f}px")
        print(
            f"[SLAM+A*] initial_estimate={est.tolist()} "
            f"initial_error={errors[0]:.1f}px "
            f"verified_init={nav.initialization_verified}"
        )
        if nav.initialization_meta is not None:
            print(
                f"[SLAM+A*] init_match ref_idx={nav.initialization_meta['ref_idx']} "
                f"rank={nav.initialization_meta.get('retrieval_rank','?')} "
                f"sim={nav.initialization_meta.get('retrieval_similarity',float('nan')):.3f} "
                f"inliers={nav.initialization_meta['inliers']} "
                f"ratio={nav.initialization_meta['inlier_ratio']:.3f}"
            )

    for t in range(1,cfg.max_steps+1):
        heading=nav.action_heading()
        obs,done,info=sim.step(heading)

        if info["boundary"]:
            true_now=sim.p.copy()
        else:
            true_now=info["true_position"].copy()

        # True position is used only below for evaluation metrics.
        # It is never passed into SLAM, map registration, or A*.
        est,closure,_=nav.update(obs,heading)

        true_traj.append(true_now.copy())
        est_traj.append(est.copy())
        err=float(np.linalg.norm(est-true_now))
        errors.append(err)

        if progress_every and t % int(progress_every)==0 and not done:
            print(
                f"  [progress] step={t}/{cfg.max_steps} "
                f"elapsed={time.perf_counter()-episode_t0:.1f}s "
                f"closures={nav.closures}"
            )

        if verbose:
            print(
                f"[SLAM+A*][{t:03d}] err={err:7.1f}px "
                f"goal={info['goal_distance']:7.1f}px "
                f"dist={info['distance']:8.1f}px "
                f"VO={nav.vo_success}/{nav.vo_fail} closures={nav.closures}"
                + (" map-closure" if closure is not None else "")
            )

        if done:
            success=bool(info["success"])
            boundary=bool(info["boundary"])
            break

    result={
        "start":[float(start[0]),float(start[1])],
        "goal":[float(goal[0]),float(goal[1])],
        "mission_success":bool(success),
        "failure":bool(not success),
        "boundary_failure":bool(boundary),
        "steps":len(true_traj)-1,
        "executed_distance":float(sim.distance),
        "initial_goal_distance":float(np.linalg.norm(goal-start)),
        "final_true_goal_distance":float(np.linalg.norm(sim.p-goal)),
        "initial_slam_error_px":float(errors[0]),
        "initialization_verified":bool(nav.initialization_verified),
        "final_slam_error_px":float(errors[-1]),
        "mean_slam_error_px":float(np.mean(errors)),
        "median_slam_error_px":float(np.median(errors)),
        "map_closures":int(nav.closures),
        "global_relocalizations":int(nav.relocalizations),
        "relocalization_fallback_enabled":bool(cfg.enable_relocalization_fallback),
        "vo_successes":int(nav.vo_success),
        "vo_failures":int(nav.vo_fail),
        "initialization_seconds":float(init_seconds),
        "episode_seconds":float(time.perf_counter()-episode_t0),
    }

    if out_dir is not None:
        out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
        (out/"result.json").write_text(json.dumps(result,indent=2))
        np.save(out/"true_trajectory.npy",np.asarray(true_traj))
        np.save(out/"estimated_trajectory.npy",np.asarray(est_traj))

        if save_plot:
            tt=np.asarray(true_traj); ee=np.asarray(est_traj)
            plt.figure(figsize=(7,7))
            plt.plot(tt[:,0],tt[:,1],label="true trajectory")
            plt.plot(ee[:,0],ee[:,1],"--",label="SLAM estimate")
            plt.scatter([start[0],goal[0]],[start[1],goal[1]],s=50)
            plt.xlim(cfg.map_min,cfg.map_max)
            plt.ylim(cfg.map_min,cfg.map_max)
            plt.gca().set_aspect("equal",adjustable="box")
            plt.xlabel("x [px]"); plt.ylabel("y [px]")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out/"trajectory.png",dpi=180)
            plt.close()

    return result
