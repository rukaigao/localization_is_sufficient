"""Belief-to-goal controller for online-only GPS-denied navigation.

The agent never receives the simulator's true start/current position.  Normal
mission motion is computed only from the current particle belief and the known
goal coordinate.  The robust PF center is the position representative used by
both ordinary goal-directed control and the recovery coordinate frame.
"""
from __future__ import annotations
import numpy as np
from pf import ParticleSet, pf_mean_support_mass, pf_control_trust_alpha


def heading_to(a, b):
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    if float(np.linalg.norm(d)) <= 1e-12:
        return 0.0
    return float(np.arctan2(d[1], d[0]))


def normal_goal_command(particles: ParticleSet, goal: np.ndarray, cfg):
    """Return the navigator-available goal-directed heading.

    No route, start coordinate, dead-reckoned route arc, or ground truth is
    used.  The heading points from the weighted geometric median of the current
    PF to the known goal.  C_R and alpha are retained as diagnostics only; there
    is no privileged/open-loop heading to blend with in the unknown-start task.
    """
    est = particles.robust_center()
    heading = heading_to(est, goal)
    c_r = pf_mean_support_mass(
        particles, float(cfg.control_trust_radius_factor) * float(cfg.D_min)
    )
    alpha = pf_control_trust_alpha(c_r, cfg)
    return heading, est, c_r, alpha
