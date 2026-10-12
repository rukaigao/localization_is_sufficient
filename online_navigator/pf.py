"""Continuous particle filter, shared motion model, and Phase-A boundary rules."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from config import OnlineConfig
from reference_map import ReferenceMap


@dataclass
class ParticleSet:
    positions: np.ndarray
    weights: np.ndarray

    def copy(self):
        return ParticleSet(self.positions.copy(), self.weights.copy())

    def weighted_mean(self):
        """Arithmetic weighted mean (kept only for mathematical diagnostics)."""
        return np.sum(self.positions * self.weights[:, None], axis=0)

    def robust_center(self, tol: float = 1e-5, max_iter: int = 128):
        """Weighted geometric median (spatial median) of the PF.

        This is the robust current-position representative used by the online
        controller and belief diagnostics.  Unlike the arithmetic mean, a small
        number of remote particles cannot pull it far away from the dominant
        particle cloud.  Weiszfeld iterations are deterministic and use the PF
        weights directly.
        """
        x = np.asarray(self.positions, dtype=np.float64)
        w = np.asarray(self.weights, dtype=np.float64)
        if len(x) == 0:
            return np.zeros(2, dtype=np.float64)
        total = float(w.sum())
        if total <= 0.0 or not np.isfinite(total):
            w = np.full(len(x), 1.0 / len(x), dtype=np.float64)
        else:
            w = w / total
        # Arithmetic mean is a stable deterministic initialization only; the
        # returned point is the geometric median after Weiszfeld refinement.
        y = np.sum(x * w[:, None], axis=0)
        scale = max(1.0, float(np.ptp(x, axis=0).max()))
        eps = 1e-12 * scale
        for _ in range(int(max_iter)):
            d = np.linalg.norm(x - y[None, :], axis=1)
            hit = np.where(d <= eps)[0]
            if len(hit):
                # If an iterate lands exactly on a particle, a tiny denominator
                # regularization gives the standard stable Weiszfeld limit.
                d = np.maximum(d, eps)
            inv = w / d
            denom = float(inv.sum())
            if denom <= 0.0 or not np.isfinite(denom):
                break
            y_new = np.sum(x * inv[:, None], axis=0) / denom
            if float(np.linalg.norm(y_new - y)) <= float(tol):
                y = y_new
                break
            y = y_new
        return y.astype(np.float64)

    def weighted_covariance(self):
        m = self.weighted_mean()
        d = self.positions - m[None, :]
        return (d * self.weights[:, None]).T @ d

    def effective_sample_size(self):
        return float(1.0 / np.sum(self.weights ** 2))



@dataclass
class ProposalState:
    """Auxiliary one-step global hypothesis excluded from the operational PF."""
    particles: ParticleSet

    def copy(self):
        return ProposalState(self.particles.copy())

def init_particles_uniform_full_map(ref_map: ReferenceMap, cfg: OnlineConfig, rng: np.random.Generator):
    lo, hi = ref_map.map_bounds()
    n = int(cfg.n_particles)
    extent = np.maximum(hi - lo, 1e-9)
    aspect = float(extent[0] / extent[1])
    nx = max(1, int(np.ceil(np.sqrt(n * aspect))))
    ny = max(1, int(np.ceil(n / nx)))
    xe = np.linspace(lo[0], hi[0], nx + 1)
    ye = np.linspace(lo[1], hi[1], ny + 1)
    pts = []
    for iy in range(ny):
        for ix in range(nx):
            pts.append((rng.uniform(xe[ix], xe[ix+1]), rng.uniform(ye[iy], ye[iy+1])))
    pts = np.asarray(pts, dtype=np.float64)
    pts = pts[rng.permutation(len(pts))[:n]]
    return ParticleSet(pts, np.full(n, 1.0 / n))


def systematic_sample_indices(weights: np.ndarray, n: int, rng: np.random.Generator):
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(float(w.sum()), 1e-300)
    n = max(1, int(n))
    cdf = np.cumsum(w); cdf[-1] = 1.0
    u0 = rng.uniform(0.0, 1.0 / n)
    u = u0 + np.arange(n) / n
    return np.searchsorted(cdf, u)


def systematic_resample(p: ParticleSet, rng: np.random.Generator, n_out=None):
    n = len(p.weights) if n_out is None else int(n_out)
    idx = systematic_sample_indices(p.weights, n, rng)
    return ParticleSet(p.positions[idx].copy(), np.full(n, 1.0 / n))


def apply_motion(position, heading, magnitude, wind):
    direction = np.stack([np.cos(heading), np.sin(heading)], axis=-1)
    return position + direction * np.asarray(magnitude)[..., None] + wind


def nearest_feasible_heading(position, heading, magnitude, wind, ref_map: ReferenceMap):
    """Same Phase-A rule: preserve magnitude; minimally rotate continuous heading."""
    p = np.asarray(position, dtype=np.float64)
    w = np.asarray(wind, dtype=np.float64)
    r = max(float(magnitude), 0.0)
    lo, hi = ref_map.map_bounds()
    proposed = apply_motion(p, float(heading), r, w)
    if bool(ref_map.in_bounds(proposed)):
        return float(heading), r
    base = p + w
    cand = []
    if r > 1e-12:
        for xedge in (lo[0], hi[0]):
            c = (xedge - base[0]) / r
            if -1 <= c <= 1:
                a = float(np.arccos(np.clip(c, -1, 1)))
                for th in (a, -a):
                    end = base + r*np.array([np.cos(th), np.sin(th)])
                    if lo[1]-1e-9 <= end[1] <= hi[1]+1e-9:
                        cand.append(th)
        for yedge in (lo[1], hi[1]):
            ss = (yedge - base[1]) / r
            if -1 <= ss <= 1:
                a = float(np.arcsin(np.clip(ss, -1, 1)))
                for th in (a, np.pi-a):
                    end = base + r*np.array([np.cos(th), np.sin(th)])
                    if lo[0]-1e-9 <= end[0] <= hi[0]+1e-9:
                        cand.append(float(th))
    if cand:
        def diff(th): return abs(float(np.arctan2(np.sin(th-heading), np.cos(th-heading))))
        th = min(cand, key=diff)
        return float(np.arctan2(np.sin(th), np.cos(th))), r
    clipped = np.clip(proposed, lo, hi)
    eff = clipped - p - w
    mag = float(np.linalg.norm(eff))
    if mag <= 1e-12:
        return float(heading), 0.0
    return float(np.arctan2(eff[1], eff[0])), mag


def discard_oob(p: ParticleSet, ref_map: ReferenceMap):
    keep = np.asarray(ref_map.in_bounds(p.positions), dtype=bool)
    if not np.any(keep):
        lo, hi = ref_map.map_bounds()
        pos = np.clip(p.positions, lo[None,:], hi[None,:])
        w = p.weights.copy(); w /= max(float(w.sum()), 1e-300)
        return ParticleSet(pos, w)
    pos = p.positions[keep].copy(); w = p.weights[keep].copy()
    w /= max(float(w.sum()), 1e-300)
    return ParticleSet(pos, w)


def sample_rollout_wind(cfg: OnlineConfig, rng):
    w = rng.normal(cfg.wind_magnitude_mean, cfg.wind_magnitude_std, size=2)
    return w * rng.choice([-1, 1], size=2)


def sample_true_motion(nominal_heading, nominal_magnitude, cfg, rng, rollout_wind):
    h = nominal_heading + rng.normal(0.0, cfg.heading_noise_std)
    m = max(0.0, nominal_magnitude + rng.normal(0.0, cfg.magnitude_noise_std))
    return h, m, rollout_wind + rng.normal(0.0, cfg.wind_gust_std, size=2)


def pf_predict(p: ParticleSet, nominal_heading, nominal_magnitude, cfg, rng, rollout_wind, ref_map):
    n = len(p.weights)
    h = nominal_heading + rng.normal(0.0, cfg.heading_noise_std, size=n)
    m = np.maximum(0.0, nominal_magnitude + rng.normal(0.0, cfg.magnitude_noise_std, size=n))
    winds = rollout_wind[None,:] + rng.normal(0.0, cfg.wind_gust_std, size=(n,2))
    pos = apply_motion(p.positions, h, m, winds)
    return discard_oob(ParticleSet(pos, p.weights.copy()), ref_map)


def observation_likelihood(similarity: np.ndarray, u_obs: float, cfg: OnlineConfig):
    s = np.asarray(similarity, dtype=np.float64)
    log_l = max(float(cfg.observation_similarity_gain), 0.0) * float(u_obs) * (s - float(np.max(s)))
    return np.exp(np.clip(log_l, -700.0, 0.0))


def pf_observe(p: ParticleSet, similarity: np.ndarray, u_obs: float, ref_map: ReferenceMap, cfg: OnlineConfig):
    cell_l = observation_likelihood(similarity, u_obs, cfg)
    idx = ref_map.nearest_index(p.positions)
    l = cell_l[idx]
    agreement = float(np.sum(p.weights * (l / max(float(np.max(cell_l)), 1e-300))))
    w = p.weights * l
    total = float(w.sum())
    if total <= 1e-300 or not np.isfinite(total):
        w = p.weights.copy(); w /= max(float(w.sum()), 1e-300)
    else:
        w /= total
    return ParticleSet(p.positions.copy(), w), agreement, cell_l


def _sample_observation_proposals(cell_l, n, ref_map, cfg, rng):
    if n <= 0:
        return np.empty((0,2), dtype=np.float64)
    q = np.asarray(cell_l, dtype=np.float64); q /= max(float(q.sum()), 1e-300)
    idx = rng.choice(len(q), size=int(n), replace=True, p=q)
    pos = ref_map.positions[idx].astype(np.float64).copy()
    j = max(float(cfg.reinvigoration_jitter_px), 0.0)
    if j > 0:
        pos += rng.uniform(-j, j, size=pos.shape)
    lo, hi = ref_map.map_bounds()
    return np.clip(pos, lo[None,:], hi[None,:])


def reinvigoration_effective_disagreement(agreement, cfg):
    """Cross-time-tolerant disagreement used by online reinvigoration.

    Raw disagreement is D = 1-G.  The first delta of disagreement is treated
    as normal deployment-vs-reference appearance change.  Above that tolerance
    the remaining disagreement is linearly rescaled back to [0,1]:

        D_eff = max(0, (D-delta)/(1-delta)).

    delta=0 exactly recovers the previous behavior.
    """
    d = max(0.0, 1.0-float(agreement))
    delta = float(np.clip(getattr(cfg, "reinvigoration_disagreement_tolerance", 0.0), 0.0, 1.0-1e-12))
    return float(np.clip(max(0.0, (d-delta)/max(1.0-delta, 1e-12)), 0.0, 1.0))


def reinvigoration_trigger_signal(u_obs, agreement, cfg):
    """Legacy rho-shaped disagreement signal, now used ONLY to spawn scouts."""
    d_eff = reinvigoration_effective_disagreement(agreement, cfg)
    rho = float(cfg.reinvigoration_rho_max) * float(np.clip(u_obs, 0.0, 1.0)) * d_eff
    return float(np.clip(rho, 0.0, float(cfg.reinvigoration_rho_max)))


def _weight_with_cell_likelihood(p: ParticleSet, cell_l, ref_map: ReferenceMap):
    idx = np.asarray(ref_map.nearest_index(p.positions), dtype=np.int64)
    l = np.asarray(cell_l, dtype=np.float64)[idx]
    lmax = max(float(np.max(cell_l)), 1e-300)
    agreement = float(np.sum(p.weights * (l / lmax)))
    w = p.weights * l
    total = float(w.sum())
    if total <= 1e-300 or not np.isfinite(total):
        w = p.weights.copy()
        w /= max(float(w.sum()), 1e-300)
    else:
        w /= total
    return ParticleSet(p.positions.copy(), w), agreement


def _ordinary_resample_main(weighted: ParticleSet, cfg, rng):
    n_target = int(cfg.n_particles)
    need = (
        weighted.effective_sample_size() < float(cfg.ess_resample_frac) * len(weighted.weights)
        or len(weighted.weights) != n_target
    )
    if need:
        return systematic_resample(weighted, rng, n_target), True
    return weighted, False


def proposal_update_and_resample(
    weighted, u_obs, agreement, cell_l, ref_map, cfg, rng, proposal_pred=None
):
    """One-step probationary proposal mechanism replacing direct rho*N injection.

    If no proposal is active and rho_signal >= proposal_trigger_rho, spawn exactly
    proposal_n_particles scouts from the global observation likelihood. They do
    NOT enter the operational PF. On the NEXT observation, compare proposal and
    main prior-weighted normalized evidence. If E_prop/E_main reaches the
    confirmation ratio, promote exactly that many particles; otherwise discard.
    """
    rho_signal = reinvigoration_trigger_signal(u_obs, agreement, cfg)
    e_main = float(agreement)
    e_prop = None
    ratio = None
    n_target = int(cfg.n_particles)
    n_prop = int(min(max(int(cfg.proposal_n_particles), 1), max(n_target - 1, 1)))

    if proposal_pred is not None:
        prop_weighted, e_prop = _weight_with_cell_likelihood(
            proposal_pred.particles, cell_l, ref_map
        )
        ratio = float((e_prop + 1e-12) / (e_main + 1e-12))
        if ratio >= float(cfg.proposal_confirm_ratio):
            n_main = n_target - n_prop
            main_pos = systematic_resample(weighted, rng, n_main).positions
            prop_pos = systematic_resample(prop_weighted, rng, n_prop).positions
            pos = np.concatenate([main_pos, prop_pos], axis=0)
            out = ParticleSet(pos, np.full(n_target, 1.0 / n_target))
            return out, None, rho_signal, True, "promoted", e_main, e_prop, ratio
        out, did = _ordinary_resample_main(weighted, cfg, rng)
        return out, None, rho_signal, did, "rejected", e_main, e_prop, ratio

    out, did = _ordinary_resample_main(weighted, cfg, rng)
    if rho_signal >= float(cfg.proposal_trigger_rho):
        pos = _sample_observation_proposals(cell_l, n_prop, ref_map, cfg, rng)
        proposal = ProposalState(ParticleSet(pos, np.full(n_prop, 1.0 / n_prop)))
        return out, proposal, rho_signal, did, "spawned", e_main, None, None
    return out, None, rho_signal, did, "none", e_main, None, None


def pf_mean_support_mass(p: ParticleSet, radius: float):
    # Historical function name retained for compatibility; support is now
    # measured around the robust PF center rather than the arithmetic mean.
    m = p.robust_center(); d = np.linalg.norm(p.positions-m[None,:], axis=1)
    return float(np.sum(p.weights[d <= float(radius)]))


def pf_control_trust_alpha(c_r: float, cfg: OnlineConfig):
    c = float(np.clip(c_r, 0, 1)); c0=float(cfg.control_trust_transition_cr); a0=float(cfg.control_trust_transition_alpha)
    if c <= c0:
        k=max(float(cfg.control_trust_exp_k),1e-9); q=c/max(c0,1e-9)
        return float(a0*np.expm1(k*q)/np.expm1(k))
    a=max(float(cfg.control_trust_log_a),1e-9); q=(c-c0)/max(1-c0,1e-9)
    return float(a0+(1-a0)*np.log1p(a*q)/np.log1p(a))


def blend_headings(route_heading, pf_heading, alpha):
    d=np.arctan2(np.sin(pf_heading-route_heading),np.cos(pf_heading-route_heading))
    h=route_heading+float(np.clip(alpha,0,1))*d
    return float(np.arctan2(np.sin(h),np.cos(h)))
