import math
from collections import OrderedDict
import numpy as np
from data import load_cache

class _RegularGridIndexer:
    """
    Verify that cached positions are a complete regular HxW grid.
    If verification succeeds, nearest-node lookup uses only the sorted x/y axes
    and an index table -- no KDTree.
    """
    def __init__(self, positions, h, w, atol=1e-3):
        p=np.asarray(positions,dtype=np.float64)
        if p.ndim!=2 or p.shape[1]!=2 or len(p)!=h*w:
            raise ValueError(f"positions must be [{h*w},2], got {p.shape}")

        # Round only for axis discovery; original coordinates are retained.
        xs=np.unique(np.round(p[:,0],6))
        ys=np.unique(np.round(p[:,1],6))
        if len(xs)!=w or len(ys)!=h:
            raise ValueError(
                f"not a complete {h}x{w} grid: unique x={len(xs)}, y={len(ys)}"
            )

        # Verify approximately uniform spacing.
        dx=np.diff(xs); dy=np.diff(ys)
        if len(dx) and not np.allclose(dx,dx[0],rtol=1e-4,atol=atol):
            raise ValueError("x coordinates are not regularly spaced")
        if len(dy) and not np.allclose(dy,dy[0],rtol=1e-4,atol=atol):
            raise ValueError("y coordinates are not regularly spaced")

        table=np.full((h,w),-1,dtype=np.int32)
        for idx,(x,y) in enumerate(p):
            ix=int(np.argmin(np.abs(xs-x)))
            iy=int(np.argmin(np.abs(ys-y)))
            if abs(xs[ix]-x)>atol or abs(ys[iy]-y)>atol:
                raise ValueError("position does not align with inferred regular grid")
            if table[iy,ix] != -1:
                raise ValueError("duplicate regular-grid cell")
            table[iy,ix]=idx

        if np.any(table<0):
            raise ValueError("regular grid has missing cells")

        self.xs=xs.astype(np.float64)
        self.ys=ys.astype(np.float64)
        self.table=table

    @staticmethod
    def _nearest_axis(axis, value):
        j=int(np.searchsorted(axis,value))
        if j<=0: return 0
        if j>=len(axis): return len(axis)-1
        return j-1 if abs(value-axis[j-1]) <= abs(value-axis[j]) else j

    def query_index(self, xy):
        ix=self._nearest_axis(self.xs,float(xy[0]))
        iy=self._nearest_axis(self.ys,float(xy[1]))
        return int(self.table[iy,ix])


class _PairSimilarityLRU:
    """
    Each table is [n_observation_nodes, n_reference_nodes].
    Row i is exactly the reference-grid similarity field that used to be recomputed
    with ref_desc @ obs_desc[i] at every environment step.
    """
    def __init__(self):
        self.cache=OrderedDict()

    def get(self, ref_path, obs_path, max_pairs):
        key=(ref_path,obs_path)
        if key in self.cache:
            value=self.cache.pop(key)
            self.cache[key]=value
            return value

        _,rd=load_cache(ref_path)
        _,od=load_cache(obs_path)
        table=np.ascontiguousarray(od @ rd.T, dtype=np.float32)
        table.setflags(write=False)
        self.cache[key]=table

        while len(self.cache)>int(max_pairs):
            self.cache.popitem(last=False)
        return table

_SIM_LRU=_PairSimilarityLRU()
_INDEXERS={}

def _get_indexer(path, positions, h, w):
    key=(path,h,w)
    if key not in _INDEXERS:
        _INDEXERS[key]=_RegularGridIndexer(positions,h,w)
    return _INDEXERS[key]


class VisualNavEnv:
    def __init__(self,pair,cfg,continuous,seed=0):
        self.cfg=cfg
        self.continuous=continuous
        self.rng=np.random.default_rng(seed)

        rp,_=load_cache(pair.ref)
        op,_=load_cache(pair.obs)
        if len(rp)!=cfg.grid_h*cfg.grid_w:
            raise ValueError(f"need {cfg.grid_h*cfg.grid_w} ref nodes, got {len(rp)}")
        if len(op)!=cfg.grid_h*cfg.grid_w:
            raise ValueError(f"need {cfg.grid_h*cfg.grid_w} obs nodes, got {len(op)}")

        self.rp=rp
        self.op=op
        self.obs_indexer=_get_indexer(pair.obs,op,cfg.grid_h,cfg.grid_w)

        # Precomputed / LRU-cached [n_obs_nodes, n_ref_nodes] table.
        self.sim_table=_SIM_LRU.get(
            pair.ref,pair.obs,cfg.similarity_lru_pairs
        )

    def _sample(self):
        for _ in range(10000):
            s=self.rng.uniform(self.cfg.map_min,self.cfg.map_max,2)
            g=self.rng.uniform(self.cfg.map_min,self.cfg.map_max,2)
            d=np.linalg.norm(g-s)
            if d>=self.cfg.min_start_goal_px:
                return s,g,float(d)
        raise RuntimeError("could not sample start/goal")

    def reset(self,start=None,goal=None):
        if start is None:
            start,goal,d=self._sample()
        else:
            start=np.asarray(start,float)
            goal=np.asarray(goal,float)
            d=float(np.linalg.norm(goal-start))

        self.p=start.copy()
        self.g=goal.copy()
        self.dist=0.
        self.steps=0
        self.budget=max(self.cfg.step_px,self.cfg.max_distance_factor*d)
        self.wind=self.rng.normal(
            self.cfg.wind_mean,self.cfg.wind_std,2
        )*self.rng.choice([-1.,1.],2)

        return self.obs(),{"start":start,"goal":goal}

    def _goal_and_covariance_features(self,s):
        # ------------------------------------------------------------
        # 1) Easy-to-learn direction-to-goal feature from the visual MAP
        #    estimate (argmax of the current similarity field).
        # ------------------------------------------------------------
        map_idx=int(np.argmax(s))
        visual_pos=self.rp[map_idx]
        span=max(float(self.cfg.map_max-self.cfg.map_min),1e-12)
        goal_vec=np.clip((self.g-visual_pos)/span,-1.0,1.0)

        # ------------------------------------------------------------
        # 2) Spatial covariance of the FULL similarity field.
        #    This tells the policy how concentrated / spread out the
        #    current visual evidence is, without manually deciding how
        #    much to trust the argmax-derived goal vector.
        # ------------------------------------------------------------
        tau=max(float(self.cfg.covariance_softmax_temperature),1e-6)
        logits=(np.asarray(s,dtype=np.float64)-float(np.max(s)))/tau
        logits=np.clip(logits,-80.0,0.0)
        w=np.exp(logits)
        w/=max(float(w.sum()),1e-12)

        mu=np.sum(w[:,None]*self.rp,axis=0)
        d=self.rp-mu[None,:]
        cov=(d*w[:,None]).T@d

        # Normalize covariance by map_size^2 so all three values are O(1).
        cov/=span*span
        cov_features=np.array([
            cov[0,0],
            cov[1,1],
            cov[0,1],
        ],dtype=np.float32)

        return np.concatenate([
            goal_vec.astype(np.float32),
            cov_features,
        ])

    def obs(self):
        obs_idx=self.obs_indexer.query_index(self.p)
        s=self.sim_table[obs_idx]

        # The similarity field is used internally only to derive the compact
        # memoryless observation. The RL policy itself receives exactly 5 values:
        #
        # [goal_dx, goal_dy, var_x, var_y, cov_xy]
        #
        # This keeps the baseline deliberately simple while still allowing the
        # policy to learn how much to trust the argmax-derived goal direction.
        return {
            "state":self._goal_and_covariance_features(s),
        }

    def heading(self,a):
        if self.continuous:
            return math.pi*float(np.clip(np.asarray(a).reshape(-1)[0],-1,1))
        a=int(a)
        if a<0 or a>7:
            raise ValueError(a)
        return a*math.pi/4.

    def step(self,a):
        old=self.p.copy()
        oldd=float(np.linalg.norm(old-self.g))

        h=self.heading(a)+self.rng.normal(0,self.cfg.heading_noise_std)
        m=max(0.,self.cfg.step_px+self.rng.normal(0,self.cfg.magnitude_noise_std))
        gust=self.rng.normal(0,self.cfg.gust_std,2)
        q=old+m*np.array([math.cos(h),math.sin(h)])+self.wind+gust
        self.steps+=1

        # Agreed behavior: terminal failure, never clip.
        if np.any(q<self.cfg.map_min) or np.any(q>self.cfg.map_max):
            return self.obs(),float(self.cfg.boundary_penalty),True,{
                "success":False,
                "boundary":True,
                "distance":self.dist
            }

        self.p=q
        self.dist+=float(np.linalg.norm(q-old))
        newd=float(np.linalg.norm(q-self.g))
        r=self.cfg.progress_scale*(oldd-newd)+self.cfg.step_penalty

        # Match the proposed pipeline exactly: success means the true UAV
        # position lies inside a goal-centered square whose half-width is
        # half one deployment observation crop (256 px on SwissView,
        # 64 px on xBD).  ``goal_tol_px`` therefore denotes this half-width.
        goal_delta=np.abs(q-self.g)
        success=bool(
            goal_delta[0]<=self.cfg.goal_tol_px
            and goal_delta[1]<=self.cfg.goal_tol_px
        )
        done=success or self.dist>=self.budget or self.steps>=self.cfg.max_steps
        if success:
            r+=self.cfg.goal_reward

        return self.obs(),float(r),bool(done),{
            "success":bool(success),
            "boundary":False,
            "distance":self.dist,
            "goal_distance":newd
        }
