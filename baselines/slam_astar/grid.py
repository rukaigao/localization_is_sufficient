import heapq, math
import numpy as np

class RegularGrid:
    def __init__(self,positions,h=38,w=38,atol=1e-3):
        p=np.asarray(positions,dtype=float)
        if p.shape!=(h*w,2):
            raise ValueError(f"Expected {(h*w,2)}, got {p.shape}")
        xs=np.unique(np.round(p[:,0],6))
        ys=np.unique(np.round(p[:,1],6))
        if len(xs)!=w or len(ys)!=h:
            raise ValueError(f"Expected {w} unique x and {h} y, got {len(xs)}/{len(ys)}")
        table=np.full((h,w),-1,dtype=int)
        for idx,(x,y) in enumerate(p):
            ix=int(np.argmin(np.abs(xs-x))); iy=int(np.argmin(np.abs(ys-y)))
            if abs(xs[ix]-x)>atol or abs(ys[iy]-y)>atol:
                raise ValueError("Position does not lie on inferred grid")
            table[iy,ix]=idx
        if np.any(table<0):
            raise ValueError("Incomplete reference grid")
        self.positions=p
        self.xs=xs.astype(float); self.ys=ys.astype(float)
        self.table=table; self.h=h; self.w=w

    @staticmethod
    def _nearest(axis,v):
        j=int(np.searchsorted(axis,v))
        if j<=0:return 0
        if j>=len(axis):return len(axis)-1
        return j-1 if abs(v-axis[j-1])<=abs(v-axis[j]) else j

    def rc_from_xy(self,xy):
        c=self._nearest(self.xs,float(xy[0]))
        r=self._nearest(self.ys,float(xy[1]))
        return r,c

    def idx_from_xy(self,xy):
        r,c=self.rc_from_xy(xy)
        return int(self.table[r,c])

    def xy_from_rc(self,rc):
        r,c=rc
        return self.positions[int(self.table[r,c])].copy()

    def astar(self,start_xy,goal_xy,connectivity=8):
        sr,sc=self.rc_from_xy(start_xy)
        gr,gc=self.rc_from_xy(goal_xy)
        start=(sr,sc); goal=(gr,gc)
        if start==goal:
            return [self.xy_from_rc(start)]

        if connectivity==4:
            moves=[(-1,0),(1,0),(0,-1),(0,1)]
        else:
            moves=[
                (-1,0),(1,0),(0,-1),(0,1),
                (-1,-1),(-1,1),(1,-1),(1,1),
            ]

        def hfun(rc):
            return float(np.linalg.norm(self.xy_from_rc(rc)-self.xy_from_rc(goal)))

        gscore={start:0.0}
        parent={}
        pq=[(hfun(start),0.0,start)]
        closed=set()

        while pq:
            _,gcur,u=heapq.heappop(pq)
            if u in closed:
                continue
            if u==goal:
                break
            closed.add(u)
            ux=self.xy_from_rc(u)
            for dr,dc in moves:
                v=(u[0]+dr,u[1]+dc)
                if not (0<=v[0]<self.h and 0<=v[1]<self.w):
                    continue
                vx=self.xy_from_rc(v)
                c=float(np.linalg.norm(vx-ux))
                ng=gcur+c
                if ng<gscore.get(v,float("inf")):
                    gscore[v]=ng
                    parent[v]=u
                    heapq.heappush(pq,(ng+hfun(v),ng,v))

        if goal not in parent:
            return [self.xy_from_rc(start),self.xy_from_rc(goal)]

        path=[goal]
        while path[-1]!=start:
            path.append(parent[path[-1]])
        path.reverse()
        return [self.xy_from_rc(rc) for rc in path]
