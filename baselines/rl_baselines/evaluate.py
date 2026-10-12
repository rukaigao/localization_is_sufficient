import numpy as np
from env import VisualNavEnv

def evaluate(pairs,cfg,continuous,act,n_eps,seed=900000):
    """
    Deterministic evaluation suite.

    Every (pair, episode_index) gets its own independent seed. Therefore:
      - start position is fixed across checkpoints,
      - goal position is fixed across checkpoints,
      - persistent wind is fixed across checkpoints,
      - the per-step stochastic motion/noise stream is fixed across checkpoints.

    The environment RNG is not reused across episodes: otherwise a policy that
    terminates earlier/later would consume a different amount of RNG and
    silently change all later validation episodes.

    If DQN and DDPG are evaluated with the same `seed`, they receive the same
    pair/start/goal/disturbance benchmark suite.
    """
    N=ok=bound=0
    ds=[]
    rs=[]

    n_eps=int(n_eps)

    for pair_idx,p in enumerate(pairs):
        for ep_idx in range(n_eps):
            # Large deterministic stride avoids overlap among pair/episode seeds.
            episode_seed = int(seed) + pair_idx * 100_000 + ep_idx

            e=VisualNavEnv(
                p,
                cfg,
                continuous,
                episode_seed,
            )
            o,_=e.reset()
            done=False
            R=0.0

            while not done:
                o,r,done,info=e.step(act(o))
                R+=r

            N+=1
            ok+=int(info["success"])
            bound+=int(info.get("boundary",False))
            ds.append(info.get("distance",0.0))
            rs.append(R)

    return {
        "episodes":N,
        "success_rate":ok/max(N,1),
        "mean_distance":float(np.mean(ds)) if ds else 0.0,
        "mean_reward":float(np.mean(rs)) if rs else 0.0,
        "boundary_failure_rate":bound/max(N,1),
        "evaluation_seed":int(seed),
        "episodes_per_pair":n_eps,
    }
