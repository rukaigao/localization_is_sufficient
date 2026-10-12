"""Conservative online belief-topology classification: GOOD/BUFFER/BROAD/MULTIMODAL."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.mixture import GaussianMixture

from config import OnlineConfig
from pf import ParticleSet, systematic_sample_indices


@dataclass
class ModeModel:
    k: int
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    responsibilities: np.ndarray
    labels: np.ndarray
    icl: float
    bic: float
    sampled_positions: np.ndarray


@dataclass
class BeliefClassification:
    state: str
    r90: float
    selected_k: int
    icl_margin_vs_1: float
    model: ModeModel | None
    reason: str
    # Diagnostics from the best ICL model, even when it is conservatively
    # collapsed back to a unimodal interpretation.
    best_icl_k: int = 1
    best_icl: float = float("nan")
    icl_k1: float = float("nan")
    best_component_weights: tuple = ()
    best_core_masses: tuple = ()
    min_pairwise_separation: float | None = None


def weighted_r90(p: ParticleSet):
    # R90 is centered on the weighted geometric median so a small number of
    # remote particles cannot drag the reference center away from the dominant
    # belief cloud.
    m = p.robust_center()
    d = np.linalg.norm(p.positions-m[None,:], axis=1)
    order = np.argsort(d); cum=np.cumsum(p.weights[order])
    j = int(np.searchsorted(cum, 0.90, side="left")); j=min(j,len(order)-1)
    return float(d[order[j]])


def _fit_gmms(p: ParticleSet, cfg: OnlineConfig, rng: np.random.Generator):
    # Systematic resampling gives sklearn an unweighted cloud preserving PF mass.
    n_fit = max(len(p.weights), 300)
    idx = systematic_sample_indices(p.weights, n_fit, rng)
    x = p.positions[idx]
    models=[]
    for k in range(1, min(int(cfg.gmm_max_k), len(x))+1):
        gm=GaussianMixture(n_components=k, covariance_type="full", reg_covar=1e-4,
                           n_init=3, random_state=int(rng.integers(0,2**31-1)))
        gm.fit(x)
        resp=gm.predict_proba(x)
        # sklearn BIC uses lower-is-better -2 log L + p log n. ICL adds a
        # classification-entropy penalty so overlapping GMM components do not
        # masquerade as distinct localization modes.
        bic=float(gm.bic(x))
        entropy=float(-np.sum(resp*np.log(np.maximum(resp,1e-300))))
        icl=bic+2.0*entropy
        labels=np.argmax(resp,axis=1)
        models.append(ModeModel(k,gm.weights_.copy(),gm.means_.copy(),gm.covariances_.copy(),resp,labels,icl,bic,x.copy()))
    return models


def _model_diagnostics(model: ModeModel, cfg: OnlineConfig):
    core=[]
    for k in range(model.k):
        core_mass=float(np.mean((model.labels==k) & (model.responsibilities[:,k] >= float(cfg.multimodal_core_responsibility))))
        core.append(core_mass)
    seps=[]
    for a in range(model.k):
        for b in range(a+1,model.k):
            delta=model.means[a]-model.means[b]
            avg=0.5*(model.covariances[a]+model.covariances[b]) + np.eye(2)*1e-6
            seps.append(float(np.sqrt(max(0.0, delta @ np.linalg.solve(avg,delta)))))
    return tuple(float(x) for x in model.weights), tuple(core), (min(seps) if seps else None)


def _clear_multimodal(model: ModeModel, icl1: float, cfg: OnlineConfig):
    if model.k <= 1:
        return False, "ICL selected K=1"
    margin=float(icl1-model.icl)
    if margin <= float(cfg.multimodal_icl_margin):
        return False, f"ICL margin {margin:.2f} <= {cfg.multimodal_icl_margin}"
    if np.any(model.weights < float(cfg.multimodal_min_component_mass)):
        return False, "a component has too little total mass"
    core=[]
    for k in range(model.k):
        core_mass=float(np.mean((model.labels==k) & (model.responsibilities[:,k] >= float(cfg.multimodal_core_responsibility))))
        core.append(core_mass)
    if np.any(np.asarray(core) < float(cfg.multimodal_min_core_mass)):
        return False, "a component lacks a decisive probability core"
    for a in range(model.k):
        for b in range(a+1,model.k):
            delta=model.means[a]-model.means[b]
            avg=0.5*(model.covariances[a]+model.covariances[b]) + np.eye(2)*1e-6
            sep=float(np.sqrt(max(0.0, delta @ np.linalg.solve(avg,delta))))
            if sep <= float(cfg.multimodal_min_separation):
                return False, f"components {a}/{b} overlap (separation={sep:.2f})"
    return True, "clear separated GMM modes"


def classify_belief(p: ParticleSet, cfg: OnlineConfig, rng: np.random.Generator):
    models=_fit_gmms(p,cfg,rng)
    m1=models[0]; best=min(models,key=lambda m:m.icl)
    clear,why=_clear_multimodal(best,m1.icl,cfg)
    r90=weighted_r90(p)
    weights,core,min_sep=_model_diagnostics(best,cfg)
    common=dict(
        best_icl_k=int(best.k), best_icl=float(best.icl), icl_k1=float(m1.icl),
        best_component_weights=weights, best_core_masses=core,
        min_pairwise_separation=None if min_sep is None else float(min_sep),
    )
    if clear:
        return BeliefClassification("MULTIMODAL",r90,best.k,float(m1.icl-best.icl),best,why,**common)
    if r90 < float(cfg.belief_r90_good_px):
        state="GOOD"
    elif r90 > float(cfg.belief_r90_risk_px):
        state="BROAD"
    else:
        state="BUFFER"
    # Non-clear K>1 is deliberately collapsed to a unimodal interpretation.
    return BeliefClassification(state,r90,1,float(m1.icl-best.icl),None,why,**common)
