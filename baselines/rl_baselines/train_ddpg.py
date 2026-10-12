import argparse, copy, json
from collections import deque
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from torch.optim import Adam

from config import DDPGConfig, apply_dataset_profile, default_train_min_start_goal
from data import scene_split, train_val_split, all_pairs, save_manifest
from env import VisualNavEnv
from models import Actor,Critic
from rl_utils import Replay, seed_all, one, batch, Stop, dump
from evaluate import evaluate

def nstd(step,c):
    f=min(1.,step/max(c.noise_decay,1))
    return c.noise_start+f*(c.noise_end-c.noise_start)

def soft(t,s,tau):
    with torch.no_grad():
        for a,b in zip(t.parameters(),s.parameters()):
            a.mul_(1-tau).add_(b,alpha=tau)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["swissview","xbd"], default="swissview")
    ap.add_argument("--dataset-root",default=None)
    ap.add_argument("--cache-root",default=None)
    ap.add_argument("--out",default="runs/ddpg")
    ap.add_argument("--episodes",type=int,default=20000)
    ap.add_argument("--min-start-goal-distance",type=float,default=None)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--device",default="cuda")
    a=ap.parse_args()

    dataset_root = a.dataset_root or ("/workspace/swissimage" if a.dataset == "swissview" else "/workspace/xBD")
    cache_root = a.cache_root or ("/workspace/AnyLoc/outputs" if a.dataset == "swissview" else "/workspace/AnyLoc/outputs_xbd")
    min_sg = (
        a.min_start_goal_distance
        if a.min_start_goal_distance is not None
        else default_train_min_start_goal(a.dataset)
    )

    c=DDPGConfig(
        dataset_kind=a.dataset,
        dataset_root=dataset_root,
        cache_root=cache_root,
        min_start_goal_px=min_sg,
        seed=a.seed,
        device=a.device
    )
    c=apply_dataset_profile(c,a.dataset)
    # Explicit CLI min distance wins over the profile.
    c.min_start_goal_px=min_sg
    seed_all(c.seed)
    dev=torch.device(a.device if torch.cuda.is_available() else "cpu")
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)

    tr40,te=scene_split(c.dataset_root,c.train_scenes,c.test_scenes,c.split_seed,c.dataset_kind,c.cache_root)
    tr,va=train_val_split(tr40,c.val_scenes,c.split_seed)
    trp=all_pairs(tr,c.dataset_root,c.cache_root)
    vap=all_pairs(va,c.dataset_root,c.cache_root)
    tep=all_pairs(te,c.dataset_root,c.cache_root)
    save_manifest(out/"split_manifest.json",tr,va,te,trp,vap,tep,c.dataset_kind)

    print(f"[DDPG] dataset={c.dataset_kind} map={c.map_max:g} grid={c.grid_h}x{c.grid_w} step={c.step_px:g} min_sg={c.min_start_goal_px:g}")
    print(
        f"[DDPG] scenes train={len(tr)} val={len(va)} test={len(te)} "
        f"pairs={len(trp)}/{len(vap)}/{len(tep)}"
    )
    print(
        f"[DDPG] optimized env: pair_block={c.episodes_per_pair_block} "
        f"update_chunk={c.update_chunk_steps} sim_lru_pairs={c.similarity_lru_pairs}"
    )

    actor=Actor(c.feature_dim).to(dev)
    critic=Critic(c.feature_dim).to(dev)
    ta=copy.deepcopy(actor).to(dev).eval()
    tc=copy.deepcopy(critic).to(dev).eval()

    ao=Adam(actor.parameters(),lr=c.actor_lr)
    co=Adam(critic.parameters(),lr=c.critic_lr)
    rb=Replay(c.replay)
    step=0
    pending_updates=0
    stopper=Stop(c.patience,c.min_delta)
    recent_train=deque(maxlen=c.eval_every)

    def deterministic(o):
        actor.eval()
        with torch.no_grad():
            s=one(o,dev)
            x=float(actor(s).item())
        actor.train()
        return float(np.clip(x,-1,1))

    def train_updates(n):
        if len(rb)<max(c.batch,c.warmup):
            return
        for _ in range(int(n)):
            xs=rb.sample(c.batch)
            s,aa,rr,ns,dd=batch(xs,dev,True)

            with torch.no_grad():
                y=rr+c.gamma*(1-dd)*tc(ns,ta(ns))

            qv=critic(s,aa)
            cl=F.mse_loss(qv,y)
            co.zero_grad(set_to_none=True)
            cl.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(),10.)
            co.step()

            al=-critic(s,actor(s)).mean()
            ao.zero_grad(set_to_none=True)
            al.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(),10.)
            ao.step()

            soft(ta,actor,c.tau)
            soft(tc,critic,c.tau)

    for ep in range(1,a.episodes+1):
        pair_idx=((ep-1)//c.episodes_per_pair_block)%len(trp)
        pair=trp[pair_idx]

        env=VisualNavEnv(pair,c,True,c.seed+ep)
        o,_=env.reset()
        done=False
        R=0.

        while not done:
            if step<c.warmup:
                act=float(np.random.uniform(-1,1))
            else:
                act=float(np.clip(
                    deterministic(o)+np.random.normal(0,nstd(step,c)),
                    -1,1
                ))

            no,r,done,info=env.step(act)
            rb.add(
                o["state"],act,r,
                no["state"],done
            )
            o=no
            R+=r
            step+=1

            # During replay warmup, collect experience only.
            # Do NOT accumulate optimization debt during warmup.
            if len(rb)>=max(c.batch,c.warmup):
                pending_updates+=1

                # After warmup, keep the intended ~1:1 update/data ratio,
                # but execute updates in small chunks for efficiency.
                if pending_updates>=c.update_chunk_steps:
                    train_updates(pending_updates)
                    pending_updates=0

        recent_train.append((
            float(R),
            int(info["success"]),
            int(info.get("boundary",False)),
        ))

        if ep%c.ckpt_every==0:
            torch.save({
                "actor":actor.state_dict(),
                "critic":critic.state_dict(),
                "target_actor":ta.state_dict(),
                "target_critic":tc.state_dict(),
                "actor_opt":ao.state_dict(),
                "critic_opt":co.state_dict(),
                "episode":ep,
                "step":step,
                "config":c.__dict__
            },out/f"checkpoint_ep{ep:06d}.pt")

        if ep%c.eval_every==0:
            # Report the completed training episodes since the previous eval.
            n_recent=max(len(recent_train),1)
            train_reward=sum(x[0] for x in recent_train)/n_recent
            train_success=sum(x[1] for x in recent_train)/n_recent
            train_boundary=sum(x[2] for x in recent_train)/n_recent
            print(
                f"[DDPG][train {ep}] success={train_success:.3f} "
                f"reward={train_reward:.2f} boundary={train_boundary:.3f} "
                f"noise={nstd(step,c):.3f}"
            )

            if pending_updates and len(rb)>=max(c.batch,c.warmup):
                train_updates(pending_updates)
                pending_updates=0

            m=evaluate(vap,c,True,deterministic,c.val_eps_per_pair,500000)
            print(
                f"[DDPG][val {ep}] success={m['success_rate']:.3f} "
                f"reward={m['mean_reward']:.2f} "
                f"boundary={m['boundary_failure_rate']:.3f}"
            )
            dump(out/"latest_val.json",m)

            imp,stop=stopper.step(m["success_rate"])
            if imp:
                torch.save({
                    "actor":actor.state_dict(),
                    "critic":critic.state_dict(),
                    "episode":ep,
                    "step":step,
                    "config":c.__dict__,
                    "val":m
                },out/"best.pt")
            if stop:
                print(f"[DDPG] early stop at {ep}, best={stopper.best:.3f}")
                break

    if pending_updates and len(rb)>=max(c.batch,c.warmup):
        train_updates(pending_updates)

    ck=torch.load(out/"best.pt",map_location=dev)
    actor.load_state_dict(ck["actor"])
    m=evaluate(tep,c,True,deterministic,c.test_eps_per_pair,900000)
    dump(out/"test_summary.json",m)
    print("[DDPG] TEST\n"+json.dumps(m,indent=2))

if __name__=="__main__":
    main()
