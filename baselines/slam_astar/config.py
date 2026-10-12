from dataclasses import dataclass, replace

@dataclass
class SLAMAStarConfig:
    dataset_kind: str = "swissview"
    split_seed: int = 2026

    # Geometry / cached AnyLoc grid
    map_min: float = 0.0
    map_max: float = 10000.0
    crop_size: int = 512
    grid_h: int = 38
    grid_w: int = 38
    step_px: float = 256.0
    # Half-width of the square goal region, matching the proposed pipeline.
    goal_tol_px: float = 256.0
    max_steps: int = 256
    max_distance_factor: float = 2.0
    min_start_goal_px: float = 2560.0

    # Scene split. SwissView: 38/10. xBD: 80/20, over the same 100-scene pool
    # inferred from cache_root (see data.py:_xbd_scene_pool).
    train_side_scenes: int = 38
    test_scenes: int = 10

    # Shared motion/disturbance model (matched to RL baselines)
    heading_noise_std: float = 0.13
    magnitude_noise_std: float = 32.0
    wind_mean: float = 16.0
    wind_std: float = 8.0
    gust_std: float = 3.0

    # AnyLoc retrieval -> geometric verification
    retrieval_top_k: int = 5
    retrieval_min_similarity: float = -1.0
    min_map_matches: int = 10
    min_map_inliers: int = 8
    min_map_inlier_ratio: float = 0.30
    ratio_test: float = 0.75
    ransac_reproj_px: float = 4.0

    enable_relocalization_fallback: bool = True
    relocalization_patience: int = 3

    max_verified_rotation_deg: float = 8.0
    min_verified_scale: float = 0.90
    max_verified_scale: float = 1.10

    # Visual odometry
    min_vo_matches: int = 10
    min_vo_inliers: int = 8
    min_vo_inlier_ratio: float = 0.30
    max_vo_translation_px: float = 700.0

    # Pose graph
    graph_window: int = 30
    vo_sigma_px: float = 35.0
    control_fallback_sigma_px: float = 90.0
    map_sigma_px: float = 45.0
    robust_loss: str = "soft_l1"
    robust_f_scale: float = 2.0

    # A*
    astar_connectivity: int = 8
    waypoint_reached_px: float = 160.0

    # Features / initialization
    sift_nfeatures: int = 1800
    reference_feature_cache_size: int = 256
    init_min_matches: int = 10
    init_min_inliers: int = 8
    init_min_inlier_ratio: float = 0.30
    init_global_grid_step: int = 1
    init_retrieval_top_k: int = 10
    anyloc_root: str = "/workspace/AnyLoc"
    anyloc_device: str = "cuda"
    anyloc_domain: str = "aerial"
    anyloc_num_clusters: int = 32

    seed: int = 0


def apply_dataset_profile(cfg: SLAMAStarConfig, dataset_kind: str) -> SLAMAStarConfig:
    """Scale only dataset-dependent geometry/noise; SwissView stays unchanged."""
    kind = dataset_kind.lower()
    if kind == "swissview":
        return replace(
            cfg, dataset_kind="swissview",
            train_side_scenes=38, test_scenes=10,
            map_min=0.0, map_max=10000.0,
            crop_size=512, grid_h=38, grid_w=38,
            step_px=256.0, goal_tol_px=256.0,
            magnitude_noise_std=32.0, wind_mean=16.0, wind_std=8.0, gust_std=3.0,
            ransac_reproj_px=4.0,
            max_vo_translation_px=700.0,
            vo_sigma_px=35.0, control_fallback_sigma_px=90.0, map_sigma_px=45.0,
            waypoint_reached_px=160.0,
        )
    if kind == "xbd":
        # Exact geometry used by the xBD RL baseline: 1024 map, 128 crop,
        # stride/action 64, 15x15 cache. Linear pixel thresholds/noises are
        # scaled by 64/256 = 1/4 where appropriate.
        return replace(
            cfg, dataset_kind="xbd",
            train_side_scenes=80, test_scenes=20,
            map_min=0.0, map_max=1024.0,
            crop_size=128, grid_h=15, grid_w=15,
            step_px=64.0, goal_tol_px=64.0,
            magnitude_noise_std=8.0, wind_mean=4.0, wind_std=2.0, gust_std=0.75,
            ransac_reproj_px=1.0,
            max_vo_translation_px=175.0,
            vo_sigma_px=8.75, control_fallback_sigma_px=22.5, map_sigma_px=11.25,
            waypoint_reached_px=40.0,
        )
    raise ValueError(f"Unknown dataset kind: {dataset_kind!r}")


# The evaluation minimum start/goal distance is defined once, in
# benchmark_manifest.default_eval_min_start_goal, so every script that
# builds or consumes the shared benchmark manifest agrees by construction.
