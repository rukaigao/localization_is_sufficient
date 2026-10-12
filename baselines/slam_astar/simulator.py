import math
import numpy as np
from vision import load_gray,crop_center
from data import load_cache
from grid import RegularGrid

class NavigationSimulator:
    """
    The observation descriptor is a cache lookup at the nearest deployment
    grid node rather than a live AnyLoc/DINOv2 encode, which avoids running
    the encoder every step. The image crop is still taken at the true
    continuous position, so SIFT/RANSAC geometric verification (VO and map
    matching) still faces a realistic offset between the true position and
    the nearest reference-grid candidate, preserving the actual localization
    difficulty. This mirrors how the RL baselines' env.py models observations
    (a precomputed [N_obs,N_ref] similarity table) for the descriptor side.
    """
    def __init__(self,pair,cfg,seed=0,obs_image=None):
        self.pair=pair
        self.cfg=cfg
        self.rng=np.random.default_rng(seed)
        self.obs_image=obs_image if obs_image is not None else load_gray(pair.obs_image)
        self.op,self.od=load_cache(pair.obs_cache)
        if len(self.op)!=cfg.grid_h*cfg.grid_w:
            raise ValueError(f"need {cfg.grid_h*cfg.grid_w} obs cache nodes, got {len(self.op)}")
        self.obs_grid=RegularGrid(self.op,cfg.grid_h,cfg.grid_w)

    def _sample(self):
        for _ in range(10000):
            s=self.rng.uniform(self.cfg.map_min,self.cfg.map_max,2)
            g=self.rng.uniform(self.cfg.map_min,self.cfg.map_max,2)
            if np.linalg.norm(g-s)>=self.cfg.min_start_goal_px:
                return s,g
        raise RuntimeError("Could not sample start/goal")

    def reset(self,start=None,goal=None):
        if start is None:
            start,goal=self._sample()
        self.p=np.asarray(start,dtype=float).copy()
        self.goal=np.asarray(goal,dtype=float).copy()
        d=float(np.linalg.norm(self.goal-self.p))
        self.distance=0.0
        self.steps=0
        self.budget=max(self.cfg.step_px,self.cfg.max_distance_factor*d)
        self.wind=self.rng.normal(
            self.cfg.wind_mean,self.cfg.wind_std,2
        )*self.rng.choice([-1.0,1.0],2)
        return self.observe()

    def observe(self):
        # Descriptor: cache lookup at the nearest deployment node, used only
        # to rank top-K candidates during global retrieval.
        # Image: taken at the true continuous position, so geometric
        # verification still sees a realistic offset from the grid node
        # rather than an exact match.
        idx=self.obs_grid.idx_from_xy(self.p)
        return {
            "image":crop_center(self.obs_image,self.p,self.cfg.crop_size),
            "descriptor":self.od[idx],
        }

    def step(self,heading):
        old=self.p.copy()
        h=float(heading)+self.rng.normal(0,self.cfg.heading_noise_std)
        m=max(0.0,self.cfg.step_px+self.rng.normal(0,self.cfg.magnitude_noise_std))
        gust=self.rng.normal(0,self.cfg.gust_std,2)
        q=old+m*np.array([math.cos(h),math.sin(h)])+self.wind+gust
        self.steps+=1

        if np.any(q<self.cfg.map_min) or np.any(q>self.cfg.map_max):
            return self.observe(),True,{
                "success":False,"boundary":True,
                "true_position":old.copy(),
                "distance":self.distance,
                "goal_distance":float(np.linalg.norm(old-self.goal)),
            }

        self.p=q
        self.distance+=float(np.linalg.norm(q-old))
        gd=float(np.linalg.norm(q-self.goal))
        # Match the proposed pipeline exactly: success means the true UAV
        # position lies inside a goal-centered square. ``goal_tol_px`` is
        # the square half-width (256 px on SwissView, 64 px on xBD).
        goal_delta=np.abs(q-self.goal)
        success=bool(
            goal_delta[0]<=self.cfg.goal_tol_px
            and goal_delta[1]<=self.cfg.goal_tol_px
        )
        done=success or self.distance>=self.budget or self.steps>=self.cfg.max_steps
        return self.observe(),bool(done),{
            "success":bool(success),"boundary":False,
            "true_position":self.p.copy(),
            "distance":self.distance,"goal_distance":gd,
        }
