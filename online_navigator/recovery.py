"""Relative-corridor online recovery candidate generation and MC confirmation."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from belief import BeliefClassification, classify_belief
from pf import (ParticleSet, ProposalState, systematic_sample_indices, pf_predict, pf_observe,
                proposal_update_and_resample, sample_rollout_wind, sample_true_motion,
                apply_motion)
from reference_map import ReferenceMap


@dataclass
class RelativeCandidate:
    points: np.ndarray              # relative (forward,lateral), first point=(0,0)
    information_sum: float
    path_length: float
    nominal_length: float
    added_length: float
    ratio_score: float
    expansion_index: int

    def actions(self, base_heading: float, step_magnitude: float = 256.0):
        """Convert candidate geometry into a variable-length continuous action sequence.

        The path itself may contain diagonal beam-search waypoints.  Actions are
        re-sampled by polyline arc length so ordinary actions remain approximately
        ``step_magnitude`` pixels (the last action may be shorter), matching the
        continuous Phase-A motion model rather than treating a long waypoint jump
        as one action.
        """
        pts=np.asarray(self.points,dtype=np.float64)
        seg=np.diff(pts,axis=0); lens=np.linalg.norm(seg,axis=1)
        total=float(np.sum(lens))
        if total<=1e-9:
            return np.empty(0),np.empty(0)
        cum=np.concatenate([[0.0],np.cumsum(lens)])
        step=max(float(step_magnitude),1e-9)
        arcs=list(np.arange(0.0,total,step))
        if not arcs or abs(arcs[-1]-total)>1e-9: arcs.append(total)
        def point_at(a):
            if a>=total: return pts[-1].copy()
            k=int(np.searchsorted(cum,a,side="right")-1); k=min(max(k,0),len(seg)-1)
            f=(a-cum[k])/max(float(lens[k]),1e-12)
            return pts[k]+f*seg[k]
        dense=np.asarray([point_at(a) for a in arcs],dtype=np.float64)
        d=np.diff(dense,axis=0); mags=np.linalg.norm(d,axis=1)
        rel_h=np.arctan2(d[:,1],d[:,0])
        return np.arctan2(np.sin(base_heading+rel_h),np.cos(base_heading+rel_h)),mags


def _sample_generation_starts(p: ParticleSet, cfg, rng):
    n=max(1,int(round(float(cfg.online_pf_sample_fraction)*len(p.weights))))
    idx=systematic_sample_indices(p.weights,n,rng)
    return p.positions[idx].copy(), idx


def _broad_value(rel_xy, starts, ref_map: ReferenceMap, heading):
    f=np.array([np.cos(heading),np.sin(heading)]); n=np.array([-f[1],f[0]])
    q=starts + rel_xy[0]*f[None,:] + rel_xy[1]*n[None,:]
    valid=np.asarray(ref_map.in_bounds(q),dtype=bool)
    vals=np.zeros(len(starts),dtype=np.float64)
    if np.any(valid): vals[valid]=ref_map.utility(q[valid])
    # OOB hypotheses contribute zero rather than being renormalized away.
    return float(np.mean(vals))


def _mode_sample_groups(classification: BeliefClassification, starts, source_indices, particles):
    model=classification.model
    if model is None: return None
    # Assign each generation start to nearest fitted mode in Mahalanobis distance.
    groups=[]
    for k in range(model.k): groups.append([])
    for i,p in enumerate(starts):
        ds=[]
        for k in range(model.k):
            d=p-model.means[k]; cov=model.covariances[k]+np.eye(2)*1e-6
            ds.append(float(d@np.linalg.solve(cov,d)))
        groups[int(np.argmin(ds))].append(i)
    return groups


def _multi_value(rel_xy, starts, groups, mode_weights, ref_map: ReferenceMap, heading):
    f=np.array([np.cos(heading),np.sin(heading)]); n=np.array([-f[1],f[0]])
    predicted=[]; mean_u=[]; effective_w=[]
    for k,g in enumerate(groups):
        if len(g)==0: continue
        pts=starts[np.asarray(g)] + rel_xy[0]*f[None,:] + rel_xy[1]*n[None,:]
        valid=np.asarray(ref_map.in_bounds(pts),dtype=bool)
        if not np.any(valid):
            # Retain mode weight with zero utility; no discriminative descriptor
            # can be trusted if this relative move is OOB for the whole mode.
            continue
        ids=ref_map.nearest_index(pts[valid])
        emb=np.mean(ref_map.unit_descriptors[ids],axis=0)
        emb=emb/max(float(np.linalg.norm(emb)),1e-12)
        predicted.append(emb)
        # Invalid projected starts count as utility 0.
        u=np.zeros(len(g)); u[valid]=ref_map.localization_utility[ids]
        mean_u.append(float(np.mean(u)))
        effective_w.append(float(mode_weights[k]))
    if len(predicted)<2: return 0.0
    ew=np.asarray(effective_w,dtype=np.float64); ew/=max(float(ew.sum()),1e-300)
    dscore=0.0
    for a in range(len(predicted)):
        for b in range(a+1,len(predicted)):
            dscore += ew[a]*ew[b]*(1.0-float(np.dot(predicted[a],predicted[b])))
    ubar=float(np.sum(ew*np.asarray(mean_u)))
    return max(0.0,dscore)*max(0.0,ubar)


def build_relative_value_field(particles, classification, ref_map, cfg, rng, heading, expansion_index):
    starts,src=_sample_generation_starts(particles,cfg,rng)
    base_L=float(cfg.search_length_persistence_factor)*float(cfg.risk_persistence_n)*float(cfg.action_magnitude)
    scale=float(cfg.search_expand_factor)**int(expansion_index)
    L=base_L*scale; W=float(cfg.search_width_px)*scale
    back=float(cfg.corridor_backward_fraction)*L
    forward=L-back
    dx=max(float(cfg.candidate_longitudinal_step_px),1.0)
    dy=max(float(cfg.candidate_lateral_step_px),1.0)
    xs=np.arange(-back,forward+0.5*dx,dx,dtype=np.float64)
    ys=np.arange(-W/2,W/2+0.5*dy,dy,dtype=np.float64)
    # Include exact start/end axes even when expansion creates non-multiple extents.
    xs=np.unique(np.append(xs,[0.0,forward])); xs.sort()
    ys=np.unique(np.append(ys,0.0)); ys.sort()
    values={}
    if classification.state=="MULTIMODAL":
        groups=_mode_sample_groups(classification,starts,src,particles)
        mw=classification.model.weights if classification.model is not None else None
        for x in xs:
            for y in ys:
                values[(float(x),float(y))]=_multi_value(np.array([x,y]),starts,groups,mw,ref_map,heading)
    else:
        for x in xs:
            for y in ys:
                values[(float(x),float(y))]=_broad_value(np.array([x,y]),starts,ref_map,heading)
    return xs,ys,values,L,W,back,forward


@dataclass
class _State:
    x: float
    y: float
    points: tuple
    visited: frozenset
    info: float
    length: float


def generate_candidates(particles, classification, ref_map, cfg, rng, heading, expansion_index):
    """Generate variable-length candidates in a relative corridor.

    Each path starts at (0,0), may first go backward to one of several depths,
    then progresses monotonically through longitudinal layers to the forward
    edge, and is forced to end at lateral offset 0.  This exposes the
    configured 1/3-behind search area (``corridor_backward_fraction``) while
    keeping the search finite and smooth.
    """
    xs,ys,val,L,W,back,forward=build_relative_value_field(
        particles,classification,ref_map,cfg,rng,heading,expansion_index)
    dx=max(float(cfg.candidate_longitudinal_step_px),1.0)
    max_dy=max(float(cfg.candidate_max_lateral_change_px),float(cfg.candidate_lateral_step_px))
    eps=float(cfg.route_score_distance_epsilon)
    beam=max(int(cfg.candidate_beam_width),int(cfg.max_candidates_to_try))
    # Candidate backward turning depths include 0 and every available negative x.
    depths=np.unique(np.maximum(0.0,-xs[xs<=1e-9]))
    out=[]
    for depth in depths:
        x0=-float(depth)
        prefix=[np.array([0.0,0.0])]
        plen=0.0; visited=set(); info=0.0
        if depth>1e-9:
            p=np.array([x0,0.0]); plen=float(depth); prefix.append(p)
        # Start info is counted once for each nearest relative grid key.
        def key_at(x,y):
            xi=float(xs[np.argmin(np.abs(xs-x))]); yi=float(ys[np.argmin(np.abs(ys-y))]); return (xi,yi)
        k0=key_at(prefix[-1][0],prefix[-1][1]); visited.add(k0); info+=float(val.get(k0,0.0))
        states=[_State(x0,0.0,tuple(prefix),frozenset(visited),info,plen)]
        layer_x=xs[(xs>x0+1e-9) & (xs<forward-1e-9)]
        for x in layer_x:
            new=[]
            for st in states:
                allowed=ys[np.abs(ys-st.y)<=max_dy+1e-9]
                for y in allowed:
                    p=np.array([float(x),float(y)])
                    k=(float(x),float(y)); vis=st.visited; inf=st.info
                    if k not in vis:
                        vis=vis|frozenset([k]); inf+=float(val.get(k,0.0))
                    length=st.length+float(np.linalg.norm(p-st.points[-1]))
                    # Partial-path detour cost is measured from the ORIGINAL
                    # recovery start x=0.  Returning from x<0 back to x=0 is
                    # therefore still detour travel rather than nominal progress.
                    nominal_progress=max(float(x),0.0)
                    added=max(length-nominal_progress,0.0)
                    backward_penalty=float(cfg.candidate_backward_penalty)*float(depth)
                    effective_added=added+backward_penalty
                    approx=inf/(effective_added+eps)
                    new.append((approx,_State(float(x),float(y),st.points+(p,),vis,inf,length)))
            if not new: break
            new.sort(key=lambda z:(z[0],z[1].info,-z[1].length),reverse=True)
            states=[z[1] for z in new[:beam]]
        # Endpoint: forward edge, zero lateral displacement. This is the
        # required zero-net-lateral constraint for the complete planned maneuver.
        end=np.array([forward,0.0])
        for st in states:
            if abs(st.y)>max_dy+1e-9: continue
            pts=np.asarray(st.points+(end,),dtype=np.float64)
            length=st.length+float(np.linalg.norm(end-st.points[-1]))
            k=key_at(forward,0.0); inf=st.info
            if k not in st.visited: inf+=float(val.get(k,0.0))
            nominal=forward  # direct start->endpoint path; backward motion is detour cost
            added=max(length-nominal,0.0)
            backward_penalty=float(cfg.candidate_backward_penalty)*float(depth)
            effective_added=added+backward_penalty
            # Straight centerline carries no actual recovery detour and is skipped.
            if np.max(np.abs(pts[:,1]))<0.5*float(cfg.candidate_lateral_step_px) and depth<1e-9:
                continue
            out.append(RelativeCandidate(pts,float(inf),float(length),float(nominal),float(added),float(inf/(effective_added+eps)),int(expansion_index)))
    out.sort(key=lambda c:(c.ratio_score,c.information_sum,-c.added_length),reverse=True)
    # Deduplicate rounded relative paths.
    unique=[]; seen=set()
    for c in out:
        key=tuple(map(tuple,np.round(c.points,3)))
        if key in seen: continue
        seen.add(key); unique.append(c)
        if len(unique)>=int(cfg.candidate_pool_size): break
    return unique


def _reference_future_observation(true_pos, ref_map, cfg):
    idx=int(ref_map.nearest_index(true_pos)[0])
    sim=ref_map.pairwise_cosine_similarity[idx].astype(np.float64)
    u=float(ref_map.localization_utility[idx])
    return sim,u


def simulate_candidate_once(candidate, base_heading, start_true, start_particles, ref_map, cfg, rng, start_proposal=None):
    true=np.asarray(start_true,dtype=np.float64).copy(); pf=start_particles.copy()
    proposal=None if start_proposal is None else start_proposal.copy()
    wind=sample_rollout_wind(cfg,rng)
    headings,mags=candidate.actions(base_heading, cfg.action_magnitude)
    traveled=0.0; first_cls=None
    for action_i,(h,m) in enumerate(zip(headings,mags)):
        # Candidate simulation uses the same physical boundary rule as online:
        # no true-position command projection; realized OOB motion is failure.
        h_cmd,m_cmd=float(h),float(m)
        ah,am,aw=sample_true_motion(h_cmd,m_cmd,cfg,rng,wind)
        old=true.copy(); proposed=apply_motion(true,ah,am,aw)
        if not bool(ref_map.in_bounds(proposed)):
            return None,pf,true,traveled,first_cls,True
        true=proposed; traveled+=float(np.linalg.norm(true-old))
        predicted=pf_predict(pf,h_cmd,m_cmd,cfg,rng,wind,ref_map)
        proposal_pred=None
        if proposal is not None:
            proposal_pred=ProposalState(pf_predict(
                proposal.particles,h_cmd,m_cmd,cfg,rng,wind,ref_map
            ))
        sim,u=_reference_future_observation(true,ref_map,cfg)
        weighted,agreement,cell_l=pf_observe(predicted,sim,u,ref_map,cfg)
        pf,proposal,_,_,_,_,_,_=proposal_update_and_resample(
            weighted,u,agreement,cell_l,ref_map,cfg,rng,proposal_pred=proposal_pred
        )
        if action_i==0:
            first_cls=classify_belief(pf,cfg,rng)
    cls=classify_belief(pf,cfg,rng)
    return cls,pf,true,traveled,first_cls,False


def confirm_candidate(candidate, base_heading, current_pf, ref_map, cfg, rng, proposal=None):
    n=int(cfg.confirm_n_rollouts)
    start_idx=systematic_sample_indices(current_pf.weights,n,rng)
    success=0; states=[]; first_states=[]; first_r90=[]; final_r90=[]
    for idx in start_idx:
        cls,_,_,_,first_cls,oob=simulate_candidate_once(
            candidate,base_heading,current_pf.positions[int(idx)],current_pf,ref_map,cfg,rng,
            start_proposal=proposal
        )
        if oob:
            states.append("OUT_OF_BOUNDS")
            if first_cls is not None:
                first_states.append(first_cls.state); first_r90.append(float(first_cls.r90))
            continue
        states.append(cls.state); final_r90.append(float(cls.r90))
        if first_cls is not None:
            first_states.append(first_cls.state); first_r90.append(float(first_cls.r90))
        success += int(cls.state=="GOOD")
    frac=float(success/n)
    first_good=float(np.mean([s=="GOOD" for s in first_states])) if first_states else 0.0

    # Final-state breakdown makes the MC confirmation fraction interpretable.
    # OUT_OF_BOUNDS is displayed as OOB in the compact candidate log.
    final_state_counts={
        "GOOD": int(sum(s=="GOOD" for s in states)),
        "MULTIMODAL": int(sum(s=="MULTIMODAL" for s in states)),
        "BROAD": int(sum(s=="BROAD" for s in states)),
        "BUFFER": int(sum(s=="BUFFER" for s in states)),
        "OOB": int(sum(s=="OUT_OF_BOUNDS" for s in states)),
    }

    diag={
        "states":states,
        "final_state_counts":final_state_counts,
        "first_action_states":first_states,
        "first_action_good_fraction":first_good,
        "first_action_median_r90":float(np.median(first_r90)) if first_r90 else None,
        "final_median_r90":float(np.median(final_r90)) if final_r90 else None,
        "final_p90_r90":float(np.percentile(final_r90,90)) if final_r90 else None,
    }
    return frac>=float(cfg.confirm_success_fraction),frac,diag


def choose_recovery_candidate(particles, classification, ref_map, cfg, rng, heading, verbose=True, proposal=None):
    for ex in range(int(cfg.max_search_expansions)+1):
        ranked=generate_candidates(particles,classification,ref_map,cfg,rng,heading,ex)
        if verbose:
            print(f"[recovery] expansion={ex} generated {len(ranked)} candidate(s)")
        if not ranked: continue
        for rank,c in enumerate(ranked[:int(cfg.max_candidates_to_try)],1):
            ok,frac,diag=confirm_candidate(c,heading,particles,ref_map,cfg,rng,proposal=proposal)
            if verbose:
                counts=diag["final_state_counts"]
                print(
                    f"[recovery] candidate {rank}/{min(len(ranked),cfg.max_candidates_to_try)} "
                    f"score={c.ratio_score:.6g} info={c.information_sum:.3f} added={c.added_length:.1f}px "
                    f"back={max(0.0,-float(np.min(c.points[:,0]))):.1f}px "
                    f"confirm={frac:.3f} first_good={diag['first_action_good_fraction']:.3f} "
                    f"final=[GOOD:{counts['GOOD']} MULTIMODAL:{counts['MULTIMODAL']} "
                    f"BROAD:{counts['BROAD']} BUFFER:{counts['BUFFER']} OOB:{counts['OOB']}] "
                    f"R90_first_med={diag['first_action_median_r90'] if diag['first_action_median_r90'] is not None else float('nan'):.1f} "
                    f"R90_final_med={diag['final_median_r90'] if diag['final_median_r90'] is not None else float('nan'):.1f} "
                    f"R90_final_p90={diag['final_p90_r90'] if diag['final_p90_r90'] is not None else float('nan'):.1f}"
                )
            if ok:
                return c,{"confirmed":True,"success_fraction":frac,"rank":rank,"expansion":ex,**diag}
    # No candidate met the confirmation threshold after all expansions.
    # Return no recovery maneuver; the caller falls back to the ordinary route
    # controller for this one step and will replan after the next observation.
    return None,{"confirmed":False,"success_fraction":0.0,"rank":None,"expansion":None,"states":[],
                "final_state_counts":{"GOOD":0,"MULTIMODAL":0,"BROAD":0,"BUFFER":0,"OOB":0},
                "first_action_states":[],"first_action_good_fraction":0.0,
                "first_action_median_r90":None,"final_median_r90":None,"final_p90_r90":None}
