"""Online-only unknown-start navigator with belief-to-goal control and PF/MC recovery."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import csv
import numpy as np

from belief import classify_belief
from controller import normal_goal_command
from pf import (ProposalState, init_particles_uniform_full_map, sample_rollout_wind, sample_true_motion,
                apply_motion, pf_predict, pf_observe,
                proposal_update_and_resample, reinvigoration_effective_disagreement,
                pf_mean_support_mass, pf_control_trust_alpha)
from recovery import choose_recovery_candidate
from reference_map import ReferenceMap, DeploymentMap
from visualization import save_online_gif


@dataclass
class OnlineResult:
    vanilla_success: bool
    real_success: bool
    failure: bool
    steps: int
    executed_distance: float
    route_length: float
    final_belief_mass_at_goal: float
    recovery_episodes: int
    final_belief_state: str | None = None
    final_r90: float | None = None
    final_pf_mean_error_px: float | None = None
    mean_pf_error_px: float | None = None
    median_pf_error_px: float | None = None
    termination_reason: str | None = None


def target_in_observation(true_pos, goal, cfg):
    half=0.5*float(cfg.target_crop_size_px)
    d=np.abs(np.asarray(true_pos)-np.asarray(goal))
    return bool(d[0]<=half and d[1]<=half)


def belief_mass_at_goal(particles, goal, cfg):
    half=0.5*float(cfg.real_success_region_size_px)
    d=np.abs(particles.positions-np.asarray(goal)[None,:])
    mask=(d[:,0]<=half)&(d[:,1]<=half)
    return float(np.sum(particles.weights[mask]))


def _observe_current(true_pos, particles, proposal_pred, deployment_map, ref_map, cfg, rng):
    """Receive one observation and update the main PF plus optional scout proposal."""
    obs_desc=deployment_map.descriptor_at(true_pos)
    sim=ref_map.observation_similarity(obs_desc)
    u_obs=ref_map.observation_utility_from_similarity(sim)
    weighted,agreement,cell_l=pf_observe(particles,sim,u_obs,ref_map,cfg)
    ess_pre=float(weighted.effective_sample_size())
    (particles,proposal,rho,resampled,proposal_status,e_main,e_prop,proposal_ratio) = proposal_update_and_resample(
            weighted,u_obs,agreement,cell_l,ref_map,cfg,rng,proposal_pred=proposal_pred
        )
    ess_post=float(particles.effective_sample_size())
    return (particles,proposal,float(u_obs),float(agreement),float(rho),bool(resampled),
            ess_pre,ess_post,proposal_status,float(e_main),
            None if e_prop is None else float(e_prop),
            None if proposal_ratio is None else float(proposal_ratio))


def run_online(start, goal, ref_map: ReferenceMap, deployment_map: DeploymentMap,
               cfg, out_dir: Path, mode: str="risk_aware", verbose=True, background_tif=None):
    if mode not in {"risk_aware","goal_only"}:
        raise ValueError("mode must be 'risk_aware' or 'goal_only'")
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    # IMPORTANT: start is simulator/evaluator state only.  It is used to
    # place the physical vehicle and compute the external evaluation budget, but
    # is never passed to the controller, belief update, risk detector, or
    # recovery planner.
    start=np.asarray(start,dtype=np.float64).copy()
    goal=np.asarray(goal,dtype=np.float64).copy()
    nominal_start_goal_distance=float(np.linalg.norm(goal-start))
    if nominal_start_goal_distance <= 0.0:
        raise ValueError("start and goal must be distinct")
    rng=np.random.default_rng(int(cfg.seed))
    # Separate RNG for diagnostics that must not perturb the navigation/random
    # stream, especially in goal_only ablation mode.
    diag_rng=np.random.default_rng(int(cfg.seed)+1000003)
    particles=init_particles_uniform_full_map(ref_map,cfg,rng)
    proposal=None
    true_pos=start.copy(); rollout_wind=sample_rollout_wind(cfg,rng)
    executed=0.0
    # External benchmark termination only.  This value is never used to select
    # an action and is not exposed to the navigator.
    max_dist=float(cfg.online_max_route_length_factor)*nominal_start_goal_distance
    bad_count=0; risk_active=False; recovery_episodes=0
    records=[]
    snapshots=[]

    # t=0 observation: the navigator begins with a real deployment observation.
    (particles,proposal,u_obs,agreement,rho,resampled,ess_pre,ess_post,
     proposal_status,proposal_e_main,proposal_e_prop,proposal_ratio)=_observe_current(
        true_pos,particles,None,deployment_map,ref_map,cfg,rng
    )
    cls=classify_belief(particles,cfg,rng) if mode=="risk_aware" else None
    diag_cls=cls if cls is not None else classify_belief(particles,cfg,diag_rng)
    if cls is not None:
        bad_count=1 if cls.state in {"BROAD","MULTIMODAL"} else 0

    # If the start crop already contains the target, evaluate immediately.
    vanilla=target_in_observation(true_pos,goal,cfg)
    goal_mass=belief_mass_at_goal(particles,goal,cfg)
    if vanilla:
        pf_err=float(np.linalg.norm(particles.robust_center()-true_pos))
        result=OnlineResult(True,goal_mass>=float(cfg.real_success_belief_mass),False,0,0.0,nominal_start_goal_distance,goal_mass,0,diag_cls.state,float(diag_cls.r90),pf_err,pf_err,pf_err,"goal")
        with open(out_dir/"summary.json","w") as f: json.dump(asdict(result),f,indent=2)
        (out_dir/"steps.jsonl").write_text("")
        np.save(out_dir/"true_trajectory.npy",np.asarray([true_pos],dtype=np.float64))
        return result

    trajectory=[true_pos.copy()]
    snapshots.append({
        "step":0,"true_pos":true_pos.copy(),"trajectory":np.asarray(trajectory).copy(),
        "particles":particles.positions.copy(),"pf_mean":particles.robust_center().copy(),
        "state":diag_cls.state,"risk":False,"action":"initial","r90":float(diag_cls.r90),
        "ess_pre":ess_pre,"u_obs":u_obs,"rho":rho,
        "proposal_status":proposal_status,"proposal_e_main":proposal_e_main,
        "proposal_e_prop":proposal_e_prop,"proposal_ratio":proposal_ratio,
        "effective_disagreement":reinvigoration_effective_disagreement(agreement,cfg),
        "pf_mean_error":float(np.linalg.norm(particles.robust_center()-true_pos)),"goal_mass":goal_mass,
    })
    for step in range(1,int(cfg.max_steps)+1):
        # Risk state entering/exiting is based on the CURRENT posterior, which was
        # produced by the latest real observation.
        if mode=="risk_aware":
            is_bad=cls.state in {"BROAD","MULTIMODAL"}
            if not risk_active:
                if step>1:  # t=0 was already counted above
                    bad_count = bad_count+1 if is_bad else 0
                if bad_count>=int(cfg.risk_persistence_n):
                    risk_active=True; recovery_episodes+=1
                    if verbose: print(f"[online][{step-1:04d}] RISK OPEN state={cls.state} R90={cls.r90:.1f}px")
            elif cls.state=="GOOD":
                # Asymmetric rule: one strict GOOD posterior is enough to stop
                # information-seeking and return to direct belief-to-goal control.
                risk_active=False; bad_count=0
                if verbose: print(f"[online][{step-1:04d}] RISK CLOSED strict GOOD R90={cls.r90:.1f}px")

        candidate_meta=None
        # Mission forward direction is defined entirely by the current belief:
        # robust PF center -> known goal.  No Phase-A route or true start is used.
        base_heading,belief_center,c_r,control_alpha=normal_goal_command(particles,goal,cfg)
        if mode=="risk_aware" and risk_active:
            cand,candidate_meta=choose_recovery_candidate(
                particles,cls,ref_map,cfg,rng,base_heading,verbose=verbose,proposal=proposal
            )
            if cand is not None and len(cand.points)>=2:
                headings,mags=cand.actions(base_heading, cfg.action_magnitude)
                nominal_heading=float(headings[0]); nominal_mag=float(mags[0])
                action_source="recovery"
            else:
                nominal_heading=float(base_heading)
                nominal_mag=float(cfg.action_magnitude)
                action_source="goal_fallback"
        else:
            nominal_heading=float(base_heading)
            nominal_mag=float(cfg.action_magnitude)
            action_source="goal"

        # Execute ONE action only.  The controller uses only belief + known goal.
        # OOB is checked on realized simulator motion after action/wind noise.
        route_forward_progress=0.0  # legacy diagnostic field; no route exists.
        ah,am,aw=sample_true_motion(nominal_heading,nominal_mag,cfg,rng,rollout_wind)
        old=true_pos.copy()
        proposed_true_pos=apply_motion(true_pos,ah,am,aw)
        if not bool(ref_map.in_bounds(proposed_true_pos)):
            # The attempted OOB move is rejected physically and terminates the
            # episode immediately. No outside-map observation/PF update occurs.
            goal_mass=belief_mass_at_goal(particles,goal,cfg)
            pf_mean_error=float(np.linalg.norm(particles.robust_center()-true_pos))
            error_series=np.asarray([r["pf_mean_error_px"] for r in records],dtype=np.float64)
            mean_pf_error=float(np.mean(error_series)) if error_series.size else pf_mean_error
            median_pf_error=float(np.median(error_series)) if error_series.size else pf_mean_error
            result=OnlineResult(
                False,False,True,step,executed,nominal_start_goal_distance,goal_mass,
                recovery_episodes,diag_cls.state,float(diag_cls.r90),pf_mean_error,
                mean_pf_error,median_pf_error,"out_of_bounds"
            )
            if verbose:
                print(
                    f"[online][{step:04d}] TERMINATE out_of_bounds "
                    f"from=({old[0]:.1f},{old[1]:.1f}) "
                    f"proposed=({proposed_true_pos[0]:.1f},{proposed_true_pos[1]:.1f})"
                )
            break
        true_pos=proposed_true_pos
        step_dist=float(np.linalg.norm(true_pos-old)); executed+=step_dist
        trajectory.append(true_pos.copy())

        # Predict using the issued action, then receive the NEW real observation.
        particles=pf_predict(
            particles,nominal_heading,nominal_mag,cfg,rng,rollout_wind,ref_map
        )
        proposal_pred=None
        if proposal is not None:
            proposal_pred=ProposalState(pf_predict(
                proposal.particles,nominal_heading,nominal_mag,cfg,rng,rollout_wind,ref_map
            ))
        (particles,proposal,u_obs,agreement,rho,resampled,ess_pre,ess_post,
         proposal_status,proposal_e_main,proposal_e_prop,proposal_ratio)=_observe_current(
            true_pos,particles,proposal_pred,deployment_map,ref_map,cfg,rng
        )
        cls=classify_belief(particles,cfg,rng) if mode=="risk_aware" else None
        diag_cls=cls if cls is not None else classify_belief(particles,cfg,diag_rng)

        vanilla=target_in_observation(true_pos,goal,cfg)
        goal_mass=belief_mass_at_goal(particles,goal,cfg)
        real=bool(vanilla and goal_mass>=float(cfg.real_success_belief_mass))
        pf_mean=particles.robust_center(); pf_mean_error=float(np.linalg.norm(pf_mean-true_pos))

        # IMPORTANT DIAGNOSTIC TIMING:
        #   c_r / control_alpha were computed from the PRE-ACTION belief and
        #   are exactly the quantities that selected this action.
        #   R90/ESS/G/rho below describe the POST-ACTION, POST-OBSERVATION PF.
        # Compute a matching POST pair so R90_post and C_R_post refer to the
        # exact same particle set and robust center.  This is diagnostic only
        # and does not affect control.
        control_radius = float(cfg.control_trust_radius_factor) * float(cfg.D_min)
        c_r_post = pf_mean_support_mass(particles, control_radius)
        control_alpha_post = pf_control_trust_alpha(c_r_post, cfg)

        records.append({
            "step":step,"true_x":float(true_pos[0]),"true_y":float(true_pos[1]),
            "pf_mean_x":float(pf_mean[0]),"pf_mean_y":float(pf_mean[1]),"pf_mean_error_px":pf_mean_error,
            "executed_distance":executed,"u_obs":u_obs,"agreement":agreement,
            "effective_disagreement":reinvigoration_effective_disagreement(agreement,cfg),
            "rho":rho,"ess_pre_resample":ess_pre,"ess_post_resample":ess_post,"resampled":resampled,
            "proposal_status":proposal_status,"proposal_active":bool(proposal is not None),
            "proposal_e_main":proposal_e_main,"proposal_e_prop":proposal_e_prop,
            "proposal_ratio":proposal_ratio,
            "belief_state":None if cls is None else cls.state,"diagnostic_belief_state":diag_cls.state,
            "r90":float(diag_cls.r90),"r90_post":float(diag_cls.r90),"selected_k":int(diag_cls.selected_k),
            "best_icl_k":int(diag_cls.best_icl_k),"icl_margin_vs_1":float(diag_cls.icl_margin_vs_1),
            "gmm_component_weights":list(diag_cls.best_component_weights),
            "gmm_core_masses":list(diag_cls.best_core_masses),
            "gmm_min_pairwise_separation":diag_cls.min_pairwise_separation,
            "belief_reason":diag_cls.reason,
            # Backward-compatible aliases: these historical fields are PRE-action.
            "control_support_mass_cr":float(c_r),"control_alpha":float(control_alpha),
            "control_support_mass_cr_pre":float(c_r),"control_alpha_pre":float(control_alpha),
            "control_support_mass_cr_post":float(c_r_post),"control_alpha_post":float(control_alpha_post),
            "route_arc":None,"route_forward_progress":float(route_forward_progress),
            "belief_center_x_pre":float(belief_center[0]),"belief_center_y_pre":float(belief_center[1]),
            "goal_heading_pre":float(base_heading),
            "risk_active":bool(risk_active),"bad_persistence":int(bad_count),"action_source":action_source,
            "vanilla_success":vanilla,"goal_belief_mass":goal_mass,"real_success":real,
            "candidate_confirmed":None if candidate_meta is None else candidate_meta.get("confirmed"),
            "candidate_success_fraction":None if candidate_meta is None else candidate_meta.get("success_fraction"),
            "candidate_rank":None if candidate_meta is None else candidate_meta.get("rank"),
            "candidate_expansion":None if candidate_meta is None else candidate_meta.get("expansion"),
            "candidate_first_action_good_fraction":None if candidate_meta is None else candidate_meta.get("first_action_good_fraction"),
            "candidate_first_action_median_r90":None if candidate_meta is None else candidate_meta.get("first_action_median_r90"),
            "candidate_final_median_r90":None if candidate_meta is None else candidate_meta.get("final_median_r90"),
        })
        snapshots.append({
            "step":step,"true_pos":true_pos.copy(),"trajectory":np.asarray(trajectory).copy(),
            "particles":particles.positions.copy(),"pf_mean":pf_mean.copy(),
            "state":diag_cls.state,"risk":bool(risk_active),"action":action_source,"r90":float(diag_cls.r90),
            "ess_pre":ess_pre,"u_obs":u_obs,"rho":rho,
            "proposal_status":proposal_status,"proposal_e_main":proposal_e_main,
            "proposal_e_prop":proposal_e_prop,"proposal_ratio":proposal_ratio,
            "effective_disagreement":reinvigoration_effective_disagreement(agreement,cfg),
            "pf_mean_error":pf_mean_error,"goal_mass":goal_mass,
        })
        if verbose:
            state="SKIP" if cls is None else cls.state
            print(
                f"[online][{step:04d}] state={state:10s} risk={int(risk_active)} action={action_source:14s} "
                f"Uobs={u_obs:.3f} R90_post={diag_cls.r90:.1f}px K={diag_cls.best_icl_k} "
                f"dICL={diag_cls.icl_margin_vs_1:.1f} ESS={ess_pre:.1f}/{len(particles.weights)} "
                f"G={agreement:.3f} D_eff={reinvigoration_effective_disagreement(agreement,cfg):.3f} "
                f"rho={rho:.3f} prop={proposal_status} "
                f"pratio={proposal_ratio if proposal_ratio is not None else float('nan'):.2f} "
                f"C_R_pre={c_r:.3f} alpha_pre={control_alpha:.3f} "
                f"C_R_post={c_r_post:.3f} alpha_post={control_alpha_post:.3f} "
                f"PFerr={pf_mean_error:.1f}px dist={executed:.1f}/{max_dist:.1f} goal_mass={goal_mass:.3f}"
            )

        if vanilla:
            error_series=np.asarray([r["pf_mean_error_px"] for r in records],dtype=np.float64)
            result=OnlineResult(True,real,False,step,executed,nominal_start_goal_distance,goal_mass,recovery_episodes,diag_cls.state,float(diag_cls.r90),pf_mean_error,float(np.mean(error_series)),float(np.median(error_series)),"goal")
            break
        if executed>=max_dist:
            error_series=np.asarray([r["pf_mean_error_px"] for r in records],dtype=np.float64)
            result=OnlineResult(False,False,True,step,executed,nominal_start_goal_distance,goal_mass,recovery_episodes,diag_cls.state,float(diag_cls.r90),pf_mean_error,float(np.mean(error_series)),float(np.median(error_series)),"distance_budget")
            break
    else:
        goal_mass=belief_mass_at_goal(particles,goal,cfg)
        pf_mean_error=float(np.linalg.norm(particles.robust_center()-true_pos))
        error_series=np.asarray([r["pf_mean_error_px"] for r in records],dtype=np.float64)
        mean_pf_error=float(np.mean(error_series)) if error_series.size else pf_mean_error
        median_pf_error=float(np.median(error_series)) if error_series.size else pf_mean_error
        result=OnlineResult(False,False,True,int(cfg.max_steps),executed,nominal_start_goal_distance,goal_mass,recovery_episodes,diag_cls.state,float(diag_cls.r90),pf_mean_error,mean_pf_error,median_pf_error,"max_steps")

    with open(out_dir/"summary.json","w") as f: json.dump(asdict(result),f,indent=2)
    with open(out_dir/"steps.jsonl","w") as f:
        for r in records: f.write(json.dumps(r)+"\n")
    # Flat CSV makes the core numeric diagnostics convenient to inspect/plot.
    if records:
        csv_fields=[k for k,v in records[0].items() if not isinstance(v,(list,dict))]
        with open(out_dir/"diagnostics.csv","w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=csv_fields); w.writeheader()
            for r in records: w.writerow({k:r.get(k) for k in csv_fields})
    np.save(out_dir/"true_trajectory.npy",np.asarray(trajectory,dtype=np.float64))
    if bool(cfg.save_gif):
        try:
            save_online_gif(out_dir/"online_pf.gif",snapshots,np.empty((0,2),dtype=np.float64),goal,ref_map,cfg,background_tif)
        except Exception as e:
            print(f"[visualization] GIF generation failed: {type(e).__name__}: {e}")
    return result
