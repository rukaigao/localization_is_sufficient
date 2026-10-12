"""Reference and deployment descriptor-map wrappers for online navigation."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from config import OnlineConfig


def load_cache(path: Path):
    import torch
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        d = torch.load(path, map_location="cpu")
    if "positions" not in d or "descriptors" not in d:
        raise KeyError(f"{path} must contain positions and descriptors")
    pos = np.asarray(d["positions"], dtype=np.float64)
    desc = d["descriptors"]
    desc = desc.float().cpu().numpy() if hasattr(desc, "float") else np.asarray(desc)
    desc = np.asarray(desc, dtype=np.float64)
    if len(pos) != len(desc):
        raise ValueError("positions/descriptors count mismatch")
    return {
        "positions": pos,
        "descriptors": desc,
        "crop_size": int(d.get("crop_size", 512)),
        "stride": int(d.get("stride", 256)),
        "raw": d,
    }


class ReferenceMap:
    def __init__(self, cache: dict, cfg: OnlineConfig):
        self.cfg = cfg
        self.positions = np.asarray(cache["positions"], dtype=np.float64)
        self.descriptors = np.asarray(cache["descriptors"], dtype=np.float64)
        nrm = np.linalg.norm(self.descriptors, axis=1, keepdims=True)
        self.unit_descriptors = self.descriptors / np.maximum(nrm, 1e-12)
        self.crop_size = int(cache.get("crop_size", cfg.crop_size))
        self.stride = int(cache.get("stride", cfg.stride))
        self._tree = cKDTree(self.positions)

        self.pairwise_cosine_similarity = (self.unit_descriptors @ self.unit_descriptors.T).astype(np.float32)
        diff = self.positions[:, None, :] - self.positions[None, :, :]
        self.pairwise_spatial_distance = np.linalg.norm(diff, axis=2).astype(np.float32)
        self.alias_kernel = self._compute_alias_kernel()
        self.alias_ambiguity = np.sum(self.alias_kernel.astype(np.float64), axis=1)
        self.alias_ambiguity_lo = float(np.percentile(self.alias_ambiguity, 5.0))
        self.alias_ambiguity_hi = float(np.percentile(self.alias_ambiguity, 95.0))
        self.localization_utility = self._ambiguity_to_utility(self.alias_ambiguity)

    def _compute_alias_kernel(self):
        visual = np.maximum(
            self.pairwise_cosine_similarity.astype(np.float64) - float(self.cfg.alias_similarity_threshold),
            0.0,
        ) ** float(self.cfg.alias_similarity_power)
        d0 = max(float(self.cfg.D_min), 1e-9)
        d = self.pairwise_spatial_distance.astype(np.float64)
        spatial = 1.0 - np.exp(-(d * d) / (d0 * d0))
        return (visual * spatial).astype(np.float32)

    def _ambiguity_to_utility(self, ambiguity):
        a = np.asarray(ambiguity, dtype=np.float64)
        lo, hi = self.alias_ambiguity_lo, self.alias_ambiguity_hi
        if hi <= lo + 1e-12:
            return np.ones_like(a)
        return 1.0 - np.clip((a - lo) / (hi - lo), 0.0, 1.0)

    def observation_similarity(self, observation_descriptor: np.ndarray) -> np.ndarray:
        q = np.asarray(observation_descriptor, dtype=np.float64).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        if q.shape[0] != self.unit_descriptors.shape[1]:
            raise ValueError("deployment/reference descriptor dimensions differ")
        return self.unit_descriptors @ q

    def observation_utility_from_similarity(self, sim: np.ndarray) -> float:
        """Same U definition as U_ref, replacing ref->ref by obs->ref.

        Since the deployment observation has no known absolute position, the
        highest-similarity reference location anchors the spatial term.  Strong
        matches spread far from that anchor increase ambiguity; geographically
        concentrated matches do not.
        """
        s = np.asarray(sim, dtype=np.float64)
        best = int(np.argmax(s))
        d = np.linalg.norm(self.positions - self.positions[best][None, :], axis=1)
        visual = np.maximum(s - float(self.cfg.alias_similarity_threshold), 0.0) ** float(self.cfg.alias_similarity_power)
        d0 = max(float(self.cfg.D_min), 1e-9)
        spatial = 1.0 - np.exp(-(d * d) / (d0 * d0))
        ambiguity = float(np.sum(visual * spatial))
        return float(self._ambiguity_to_utility(np.array([ambiguity]))[0])

    def utility(self, xy):
        single = np.asarray(xy).ndim == 1
        idx = self.nearest_index(xy)
        out = self.localization_utility[idx]
        return float(out[0]) if single else out

    def nearest_index(self, xy):
        arr = np.asarray(xy, dtype=np.float64)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        _, idx = self._tree.query(arr, k=1)
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        return idx

    def map_bounds(self):
        return (
            np.array([self.cfg.map_min_x, self.cfg.map_min_y], dtype=np.float64),
            np.array([self.cfg.map_max_x, self.cfg.map_max_y], dtype=np.float64),
        )

    def in_bounds(self, xy):
        arr = np.asarray(xy, dtype=np.float64)
        lo, hi = self.map_bounds()
        if arr.ndim == 1:
            return bool(np.all(arr >= lo) and np.all(arr <= hi))
        return np.all((arr >= lo[None, :]) & (arr <= hi[None, :]), axis=1)


class DeploymentMap:
    """Simulation-time source of the *real* deployment observation descriptor.

    The navigator never receives ground-truth position.  The simulated
    environment uses GT only to select the nearest deployment-time crop.
    """
    def __init__(self, cache: dict):
        self.positions = np.asarray(cache["positions"], dtype=np.float64)
        self.descriptors = np.asarray(cache["descriptors"], dtype=np.float64)
        nrm = np.linalg.norm(self.descriptors, axis=1, keepdims=True)
        self.unit_descriptors = self.descriptors / np.maximum(nrm, 1e-12)
        self._tree = cKDTree(self.positions)

    def descriptor_at(self, true_xy: np.ndarray) -> np.ndarray:
        _, idx = self._tree.query(np.asarray(true_xy, dtype=np.float64), k=1)
        return self.unit_descriptors[int(idx)].copy()
