import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    """
    Compact memoryless state:
      [goal_dx, goal_dy, var_x, var_y, cov_xy]

    No full 38x38 similarity map is passed to the RL policy.
    """
    def __init__(self,d=128):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(5,128),
            nn.ReLU(),
            nn.Linear(128,d),
            nn.ReLU(),
        )

    def forward(self,x):
        return self.net(x)


class DQN(nn.Module):
    def __init__(self,d=128):
        super().__init__()
        self.e=StateEncoder(d)
        self.h=nn.Sequential(
            nn.Linear(d,128),
            nn.ReLU(),
            nn.Linear(128,8),
        )

    def forward(self,x):
        return self.h(self.e(x))


class Actor(nn.Module):
    def __init__(self,d=128):
        super().__init__()
        self.e=StateEncoder(d)
        self.h=nn.Sequential(
            nn.Linear(d,128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Tanh(),
        )

    def forward(self,x):
        return self.h(self.e(x))


class Critic(nn.Module):
    def __init__(self,d=128):
        super().__init__()
        self.e=StateEncoder(d)
        self.h=nn.Sequential(
            nn.Linear(d+1,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,1),
        )

    def forward(self,x,a):
        return self.h(torch.cat([self.e(x),a],dim=-1))
