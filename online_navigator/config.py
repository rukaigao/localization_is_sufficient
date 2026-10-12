"""Configuration for the online GPS-denied navigator.

All distances are reference-map pixels.  Motion, PF states, and actions remain
continuous; the reference crop grid is only an efficient nearest-descriptor
lookup.
"""
from dataclasses import dataclass, replace


@dataclass
class OnlineConfig:
    # Dataset profile. SwissView remains the default for backward compatibility.
    dataset_kind: str = "swissview"

    # Shared map / continuous motion model
    crop_size: int = 512
    stride: int = 256
    action_magnitude: float = 256.0
    heading_noise_std: float = 0.13
    magnitude_noise_std: float = 32.0
    wind_magnitude_mean: float = 16.0
    wind_magnitude_std: float = 8.0
    wind_gust_std: float = 3.0
    lookahead_distance: float = 256.0

    # PF confidence diagnostics used by the online goal controller.
    control_trust_radius_factor: float = 1.0
    control_trust_transition_cr: float = 0.5
    control_trust_transition_alpha: float = 0.5
    control_trust_exp_k: float = 4.0
    control_trust_log_a: float = 4.0

    # Particle filter
    n_particles: int = 500
    ess_resample_frac: float = 0.5

    # Observation model.  U_obs is computed with exactly the same reference-map
    # aliasing definition as U_ref, replacing ref->ref similarity by obs->ref.
    observation_similarity_gain: float = 10.0
    alias_similarity_threshold: float = 0.20
    alias_similarity_power: float = 2.0
    D_min: float = 512.0

    # Observation-driven PF reinvigoration.  Online deployment observations
    # naturally disagree somewhat with a reference image acquired at another
    # timepoint, so the first 20% of disagreement (1-G) is treated as normal
    # cross-time appearance variation rather than evidence of a lost PF.
    reinvigoration_rho_max: float = 0.50
    reinvigoration_disagreement_tolerance: float = 0.20
    reinvigoration_deadzone: float = 0.05  # deprecated; kept for config compatibility
    reinvigoration_jitter_px: float = 128.0
    # One-step probationary proposal PF.  The reinvigoration score above is now
    # only a trigger; it never directly replaces rho*N operational particles.
    proposal_n_particles: int = 25
    proposal_trigger_rho: float = 0.05
    proposal_confirm_ratio: float = 1.50

    # Map bounds / Phase-A-compatible boundary behavior
    map_min_x: float = 0.0
    map_min_y: float = 0.0
    map_max_x: float = 10000.0
    map_max_y: float = 10000.0

    # Instantaneous belief classification
    belief_r90_good_px: float = 682.6666666667
    belief_r90_risk_px: float = 1024.0
    gmm_max_k: int = 3
    multimodal_icl_margin: float = 10.0
    multimodal_min_component_mass: float = 0.15
    multimodal_core_responsibility: float = 0.80
    multimodal_min_core_mass: float = 0.10
    multimodal_min_separation: float = 3.0

    # Persistent risk entry.  Recovery EXIT is immediate on one strict GOOD
    # posterior (unimodal and R90 < belief_r90_good_px).
    risk_persistence_n: int = 5

    # Relative online recovery corridor
    # L = search_length_persistence_factor * risk_persistence_n * action_magnitude
    search_length_persistence_factor: float = 2.0
    search_width_px: float = 2048.0
    corridor_backward_fraction: float = 1.0 / 3.0
    search_expand_factor: float = 1.5
    max_search_expansions: int = 2

    # Candidate family.  Candidate length is NOT fixed; paths may first move
    # backward inside the relative corridor and then progress forward.  Every
    # complete planned candidate ends at zero net lateral displacement.
    candidate_longitudinal_step_px: float = 256.0
    candidate_lateral_step_px: float = 256.0
    candidate_max_lateral_change_px: float = 256.0
    candidate_beam_width: int = 800
    candidate_pool_size: int = 100
    route_score_distance_epsilon: float = 4096.0
    # Extra directional penalty applied only to negative longitudinal recovery
    # motion (away from the known goal direction).  Ordinary forward/lateral
    # detour scoring remains unchanged.
    candidate_backward_penalty: float = 1.0
    max_candidates_to_try: int = 5

    # Belief samples for cheap relative-field construction.  Systematic sampling
    # is used; this fraction does not mutate the actual PF.
    online_pf_sample_fraction: float = 0.20

    # Full Monte-Carlo confirmation, tried sequentially in cheap-score order.
    confirm_n_rollouts: int = 15
    confirm_success_fraction: float = 0.80

    # Mission evaluation
    target_crop_size_px: float = 512.0
    real_success_region_size_px: float = 1024.0
    real_success_belief_mass: float = 0.90
    online_max_route_length_factor: float = 2.0

    # Diagnostics / visualization (do not affect navigation decisions)
    save_gif: bool = True
    gif_frame_stride: int = 1
    gif_max_side_px: int = 1200
    gif_duration_ms: int = 180
    gif_particle_max_draw: int = 1000

    # Runtime / safety
    seed: int = 0
    max_steps: int = 5000


def apply_dataset_profile(cfg: OnlineConfig, dataset_kind: str) -> OnlineConfig:
    """Apply geometry/noise thresholds for SwissView or xBD.

    xBD uses 1024x1024 images with 128px crops and 64px stride/action.
    Pixel-valued navigation, PF-confidence, recovery, and observation-distance
    parameters are scaled by 64/256 = 1/4 from the SwissView profile.
    Dimensionless parameters are unchanged.
    """
    kind = str(dataset_kind).lower()

    if kind == "swissview":
        return replace(
            cfg,
            dataset_kind="swissview",
            crop_size=512,
            stride=256,
            action_magnitude=256.0,
            magnitude_noise_std=32.0,
            wind_magnitude_mean=16.0,
            wind_magnitude_std=8.0,
            wind_gust_std=3.0,
            lookahead_distance=256.0,
            D_min=512.0,
            reinvigoration_jitter_px=128.0,
            map_min_x=0.0,
            map_min_y=0.0,
            map_max_x=10000.0,
            map_max_y=10000.0,
            belief_r90_good_px=682.6666666667,
            belief_r90_risk_px=1024.0,
            search_width_px=2048.0,
            candidate_longitudinal_step_px=256.0,
            candidate_lateral_step_px=256.0,
            candidate_max_lateral_change_px=256.0,
            route_score_distance_epsilon=4096.0,
            target_crop_size_px=512.0,
            real_success_region_size_px=1024.0,
        )

    if kind == "xbd":
        return replace(
            cfg,
            dataset_kind="xbd",
            crop_size=128,
            stride=64,
            action_magnitude=64.0,
            magnitude_noise_std=8.0,
            wind_magnitude_mean=4.0,
            wind_magnitude_std=2.0,
            wind_gust_std=0.75,
            lookahead_distance=64.0,
            D_min=128.0,
            reinvigoration_jitter_px=32.0,
            map_min_x=0.0,
            map_min_y=0.0,
            map_max_x=1024.0,
            map_max_y=1024.0,
            belief_r90_good_px=170.666666666675,
            belief_r90_risk_px=256.0,
            search_width_px=512.0,
            candidate_longitudinal_step_px=64.0,
            candidate_lateral_step_px=64.0,
            candidate_max_lateral_change_px=64.0,
            route_score_distance_epsilon=1024.0,
            target_crop_size_px=128.0,
            real_success_region_size_px=256.0,
        )

    raise ValueError(f"Unknown dataset kind: {dataset_kind!r}")
