import math
import numpy as np

from data import load_cache
from grid import RegularGrid
from vision import load_gray,crop_center,SIFTBackend
from pose_graph import PlanarPoseGraph
from anyloc_query import AnyLocQueryEncoder

class SLAMAStarNavigator:
    def __init__(
        self,pair,cfg,query_encoder=None,vision=None,
        ref_image=None,ref_cache_data=None,
    ):
        self.pair=pair
        self.cfg=cfg

        self.rp,self.rd=(
            ref_cache_data if ref_cache_data is not None else load_cache(pair.ref_cache)
        )
        if len(self.rp)!=cfg.grid_h*cfg.grid_w:
            raise ValueError(f"Expected {cfg.grid_h}x{cfg.grid_w} reference cache")

        self.grid=RegularGrid(self.rp,cfg.grid_h,cfg.grid_w)
        self.ref_image=ref_image if ref_image is not None else load_gray(pair.ref_image)
        # Expensive, stateless feature backends may be shared across episodes.
        # This changes only recomputation/caching, not any navigation inputs.
        self.vision=vision if vision is not None else SIFTBackend(
            nfeatures=cfg.sift_nfeatures,
            ratio=cfg.ratio_test,
            ransac_px=cfg.ransac_reproj_px,
            cache_size=cfg.reference_feature_cache_size,
        )
        # The AnyLoc query encoder is no longer used online: both reference
        # and deployment sides are pre-cached, so _global_initialize() looks
        # up a descriptor instead of encoding one. Kept constructible (not
        # eagerly built) only so callers that still pass one explicitly, or
        # any offline tooling reusing this class, keep working unchanged.
        self.query_encoder=query_encoder

    def reset(self,start,goal,first_observation):
        """
        Initialize global pose from the first camera observation.

        `start` is accepted only so the caller can keep a uniform episode API;
        it is NOT used to initialize SLAM. The navigator must earn its initial
        map pose from the observation itself.
        """
        self.goal=np.asarray(goal,dtype=float).copy()
        self.prev_obs=first_observation
        self.prev_heading=None
        self.path=[]
        self.closures=0
        self.relocalizations=0
        self.local_closure_failure_streak=0
        self.vo_success=0
        self.vo_fail=0

        init=self._global_initialize(first_observation)
        if init is None:
            # Conservative fallback: start from the center of the map rather
            # than using the hidden true start.
            init_xy=np.array([
                0.5*(self.cfg.map_min+self.cfg.map_max),
                0.5*(self.cfg.map_min+self.cfg.map_max),
            ],dtype=float)
            self.initialization_verified=False
            self.initialization_meta=None
        else:
            init_xy,meta=init
            self.initialization_verified=True
            self.initialization_meta=meta

        self.graph=PlanarPoseGraph(init_xy,self.cfg)
        self.estimate=init_xy.copy()
        self.replan()
        return self.estimate.copy()

    def _global_initialize(self,obs):
        """
        Fair and efficient global initialization:

          current observation (cached descriptor + raw crop, both keyed to
          the nearest deployment grid node -- see NavigationSimulator)
              -> cosine similarity to all cached reference descriptors
              -> top-K reference candidates
              -> SIFT + RANSAC geometric verification
              -> initial (x,y) SLAM pose

        No true start coordinate is used. The query descriptor is a cache
        lookup (deployment side is pre-cached exactly like the reference
        side), not a fresh AnyLoc encode -- this is the same information a
        live encode would have produced, just without repeating an
        expensive forward pass every time this runs.
        """
        q=obs["descriptor"]
        obs_img=obs["image"]
        if q.shape[0]!=self.rd.shape[1]:
            raise RuntimeError(
                f"AnyLoc query descriptor dimension {q.shape[0]} does not match "
                f"reference cache dimension {self.rd.shape[1]}. "
                "The query encoder must use the same AnyLoc/DINO/VLAD setup "
                "that generated the reference cache."
            )

        sims=self.rd@q
        k=min(max(1,int(self.cfg.init_retrieval_top_k)),len(sims))
        ids=np.argpartition(-sims,k-1)[:k]
        ids=ids[np.argsort(-sims[ids])]

        best=None
        for rank,idx in enumerate(ids):
            idx=int(idx)
            center=self.rp[idx]
            ref_crop=crop_center(self.ref_image,center,self.cfg.crop_size)
            res=self.vision.match_affine(
                ref_crop,obs_img,
                key_a=("ref",self.pair.ref_image,idx),
                key_b=None,
            )
            if not res or "A" not in res:
                continue
            if res["matches"]<self.cfg.init_min_matches:
                continue
            if res["inliers"]<self.cfg.init_min_inliers:
                continue
            if res["inlier_ratio"]<self.cfg.init_min_inlier_ratio:
                continue
            if abs(res["rotation_deg"])>self.cfg.max_verified_rotation_deg:
                continue
            if not (self.cfg.min_verified_scale<=res["scale"]<=self.cfg.max_verified_scale):
                continue

            # ref -> obs translation t ~= ref_center - current_center.
            z=center-res["translation"]

            # Geometric verification is primary; retrieval similarity only
            # breaks otherwise similar candidates.
            geom=res["inliers"]*res["inlier_ratio"]
            score=geom*(1.0+max(float(sims[idx]),0.0))
            cand=(score,z,res,idx,float(sims[idx]),rank)
            if best is None or cand[0]>best[0]:
                best=cand

        if best is None:
            return None

        _,z,res,idx,sim,rank=best
        meta={
            "ref_idx":int(idx),
            "retrieval_rank":int(rank),
            "retrieval_similarity":float(sim),
            "inliers":int(res["inliers"]),
            "matches":int(res["matches"]),
            "inlier_ratio":float(res["inlier_ratio"]),
            "map_position":z.copy(),
        }
        return z,meta

    def _nearby_reference_candidates(self,estimate_xy):
        """
        Select map-registration candidates only from the current SLAM estimate.
        True simulator position and deployment-cache descriptors are not used.
        """
        r0,c0=self.grid.rc_from_xy(estimate_xy)
        radius=1
        ids=[]
        for dr in range(-radius,radius+1):
            for dc in range(-radius,radius+1):
                r=r0+dr
                c=c0+dc
                if 0<=r<self.cfg.grid_h and 0<=c<self.cfg.grid_w:
                    ids.append(int(self.grid.table[r,c]))
        ids=sorted(
            set(ids),
            key=lambda i: float(np.linalg.norm(self.rp[i]-np.asarray(estimate_xy)))
        )
        return ids

    def _verify_local_map_candidates(self,obs,estimate_xy):
        obs_img=obs["image"]
        best=None
        for idx in self._nearby_reference_candidates(estimate_xy):
            center=self.rp[idx]
            ref_crop=crop_center(self.ref_image,center,self.cfg.crop_size)
            res=self.vision.match_affine(
                ref_crop,obs_img,
                key_a=("ref",self.pair.ref_image,idx),
                key_b=None,
            )
            if not res or "A" not in res:
                continue
            if res["matches"]<self.cfg.min_map_matches:
                continue
            if res["inliers"]<self.cfg.min_map_inliers:
                continue
            if res["inlier_ratio"]<self.cfg.min_map_inlier_ratio:
                continue
            if abs(res["rotation_deg"])>self.cfg.max_verified_rotation_deg:
                continue
            if not (self.cfg.min_verified_scale<=res["scale"]<=self.cfg.max_verified_scale):
                continue

            # ref -> obs translation t ~= ref_center - current_center,
            # hence current_center ~= ref_center - t.
            z=center-res["translation"]

            geom=res["inliers"]*res["inlier_ratio"]
            consistency=1.0/(1.0+float(np.linalg.norm(z-estimate_xy))/max(float(self.cfg.crop_size),1.0))
            score=geom*consistency
            cand=(score,z,res,idx)
            if best is None or cand[0]>best[0]:
                best=cand
        return best

    def _visual_odometry(self,prev_obs,curr_obs):
        res=self.vision.match_affine(prev_obs["image"],curr_obs["image"])
        if not res or "A" not in res:
            return None,res
        if res["matches"]<self.cfg.min_vo_matches:
            return None,res
        if res["inliers"]<self.cfg.min_vo_inliers:
            return None,res
        if res["inlier_ratio"]<self.cfg.min_vo_inlier_ratio:
            return None,res
        if abs(res["rotation_deg"])>self.cfg.max_verified_rotation_deg:
            return None,res
        if not (self.cfg.min_verified_scale<=res["scale"]<=self.cfg.max_verified_scale):
            return None,res

        # prev -> curr feature translation is opposite camera/map translation.
        delta=-res["translation"]
        if np.linalg.norm(delta)>self.cfg.max_vo_translation_px:
            return None,res
        return delta,res

    def replan(self):
        self.path=self.grid.astar(
            self.estimate,self.goal,self.cfg.astar_connectivity
        )

    def action_heading(self):
        """
        Vanilla A*: shortest path on the 8-connected reference grid.
        A small pure-pursuit wrapper turns the next grid node into a heading.
        """
        self.replan()
        if not self.path:
            target=self.goal
        else:
            target=self.path[-1]
            for p in self.path[1:]:
                if np.linalg.norm(p-self.estimate)>self.cfg.waypoint_reached_px:
                    target=p
                    break
            # If estimated pose is already at the goal grid cell, finish toward
            # the exact continuous goal coordinate.
            if self.grid.idx_from_xy(self.estimate)==self.grid.idx_from_xy(self.goal):
                target=self.goal
        v=np.asarray(target)-self.estimate
        if np.linalg.norm(v)<1e-9:
            v=self.goal-self.estimate
        return math.atan2(float(v[1]),float(v[0]))

    def update(self,curr_obs,commanded_heading):
        """
        Update planar SLAM after the physical step.

        The navigator receives only the camera crop, commanded action, and its
        own prior SLAM estimate. True physical position is never used for
        localization, map correction, or A*.
        """
        nominal=self.cfg.step_px*np.array([
            math.cos(commanded_heading),math.sin(commanded_heading)
        ])

        delta,vo_meta=self._visual_odometry(self.prev_obs,curr_obs)
        if delta is None:
            self.vo_fail+=1
            delta=nominal
            sigma=self.cfg.control_fallback_sigma_px
        else:
            self.vo_success+=1
            sigma=self.cfg.vo_sigma_px

        pose_id=self.graph.add_step(delta,sigma)

        # Local map registration is centered on the SLAM-predicted pose.
        predicted=self.graph.poses[-1].copy()
        closure=self._verify_local_map_candidates(curr_obs,predicted)
        closure_meta=None
        if closure is not None:
            _,z,res,idx=closure
            self.graph.add_absolute(
                pose_id,z,self.cfg.map_sigma_px,
                meta={"ref_idx":idx,"inliers":res["inliers"]}
            )
            self.closures+=1
            self.local_closure_failure_streak=0
            closure_meta={
                "kind":"local",
                "map_position":z.copy(),
                "ref_idx":idx,
                "inliers":res["inliers"],
                "inlier_ratio":res["inlier_ratio"],
            }
        else:
            self.local_closure_failure_streak+=1
            if (
                self.cfg.enable_relocalization_fallback
                and self.local_closure_failure_streak>self.cfg.relocalization_patience
            ):
                # Locally lost for several steps in a row -- fall back to the
                # same full-map search used for initialization, rather than
                # continuing to dead-reckon with no absolute correction.
                reloc=self._global_initialize(curr_obs)
                if reloc is not None:
                    z,meta=reloc
                    self.graph.add_absolute(
                        pose_id,z,self.cfg.map_sigma_px,
                        meta={"ref_idx":meta["ref_idx"],"inliers":meta["inliers"]}
                    )
                    self.relocalizations+=1
                    self.local_closure_failure_streak=0
                    closure_meta={
                        "kind":"global_relocalization",
                        "map_position":z.copy(),
                        "ref_idx":meta["ref_idx"],
                        "inliers":meta["inliers"],
                        "inlier_ratio":meta["inlier_ratio"],
                    }

        self.estimate=self.graph.optimize()
        self.prev_obs=curr_obs
        return self.estimate.copy(),closure_meta,vo_meta
