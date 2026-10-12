import numpy as np
from scipy.optimize import least_squares

class PlanarPoseGraph:
    """
    Translation-only pose graph.

    This is the explicit orthogonal-to-map-plane constraint:
    altitude, roll, and pitch are fixed; the project imagery is north-aligned,
    so the optimized navigation state is only (x,y).

    Factors:
      relative: x_j - x_i ~= delta_ij   (visual odometry / control fallback)
      absolute: x_i ~= map_position      (geometrically verified map match)
    """
    def __init__(self,start_xy,cfg):
        self.cfg=cfg
        self.poses=[np.asarray(start_xy,dtype=float).copy()]
        self.rel=[]   # (i,j,delta,sigma)
        self.abs=[]   # (i,xy,sigma,meta)

    def add_step(self,delta,sigma):
        i=len(self.poses)-1
        guess=self.poses[-1]+np.asarray(delta,dtype=float)
        self.poses.append(guess)
        j=len(self.poses)-1
        self.rel.append((i,j,np.asarray(delta,dtype=float),float(sigma)))
        return j

    def add_absolute(self,i,xy,sigma,meta=None):
        self.abs.append((int(i),np.asarray(xy,dtype=float),float(sigma),meta or {}))

    def optimize(self):
        n=len(self.poses)
        if n<=1:
            return self.poses[-1].copy()

        start=max(1,n-int(self.cfg.graph_window))
        ids=list(range(start,n))
        index={pid:k for k,pid in enumerate(ids)}
        if not ids:
            return self.poses[-1].copy()
        x0=np.concatenate([self.poses[i] for i in ids])

        def pose_of(pid,x):
            if pid==0 or pid<start:
                return self.poses[pid]
            k=index[pid]
            return x[2*k:2*k+2]

        def residual(x):
            rr=[]
            for i,j,d,sig in self.rel:
                if j<start:
                    continue
                pi=pose_of(i,x); pj=pose_of(j,x)
                rr.extend(((pj-pi-d)/max(sig,1e-6)).tolist())
            for i,z,sig,_ in self.abs:
                if i<start:
                    continue
                pi=pose_of(i,x)
                rr.extend(((pi-z)/max(sig,1e-6)).tolist())
            return np.asarray(rr,dtype=float)

        r0=residual(x0)
        if r0.size==0:
            return self.poses[-1].copy()
        sol=least_squares(
            residual,x0,
            loss=self.cfg.robust_loss,
            f_scale=float(self.cfg.robust_f_scale),
            max_nfev=60,
        )
        for pid in ids:
            k=index[pid]
            self.poses[pid]=sol.x[2*k:2*k+2].copy()
        return self.poses[-1].copy()
