import random, json
from collections import deque, namedtuple
from pathlib import Path
import numpy as np, torch

T=namedtuple("T","s a r ns done")

class Replay:
    def __init__(self,n):
        self.b=deque(maxlen=int(n))
    def __len__(self):
        return len(self.b)
    def add(self,*x):
        self.b.append(T(*x))
    def sample(self,n):
        return random.sample(self.b,int(n))

def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def one(obs,dev):
    x=np.asarray(obs["state"],dtype=np.float32)
    return torch.as_tensor(x,dtype=torch.float32,device=dev).unsqueeze(0)

def batch(xs,dev,cont):
    s=torch.as_tensor(
        np.stack([x.s for x in xs]),
        dtype=torch.float32,
        device=dev,
    )
    ns=torch.as_tensor(
        np.stack([x.ns for x in xs]),
        dtype=torch.float32,
        device=dev,
    )
    r=torch.as_tensor(
        [x.r for x in xs],
        dtype=torch.float32,
        device=dev,
    ).view(-1,1)
    d=torch.as_tensor(
        [x.done for x in xs],
        dtype=torch.float32,
        device=dev,
    ).view(-1,1)

    if cont:
        a=torch.as_tensor(
            [x.a for x in xs],
            dtype=torch.float32,
            device=dev,
        ).view(-1,1)
    else:
        a=torch.as_tensor(
            [x.a for x in xs],
            dtype=torch.long,
            device=dev,
        ).view(-1,1)

    return s,a,r,ns,d

class Stop:
    def __init__(self,p,delta):
        self.p=p
        self.delta=delta
        self.best=-1e9
        self.bad=0
    def step(self,x):
        imp=x>self.best+self.delta
        if imp:
            self.best=x
            self.bad=0
        else:
            self.bad+=1
        return imp,self.bad>=self.p

def dump(path,x):
    Path(path).write_text(json.dumps(x,indent=2))
