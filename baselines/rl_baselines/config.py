from dataclasses import dataclass, replace

from benchmark_manifest import default_eval_min_start_goal


@dataclass
class Common:
    # Dataset profile. "swissview" preserves the original behavior.
    dataset_kind: str = "swissview"  # {"swissview", "xbd"}
    dataset_root: str = "/workspace/swissimage"
    cache_root: str = "/workspace/AnyLoc/outputs"
    split_seed: int = 2026

    # Split semantics:
    # scene_split() first chooses train_scenes + test_scenes.
    # train_val_split() then removes val_scenes from the training side.
    # SwissView original: 38 training-side / 10 test, then 4 val -> 34/4/10.
    train_scenes: int = 38
    test_scenes: int = 10
    val_scenes: int = 4

    map_min: float = 0.0
    map_max: float = 10000.0
    grid_h: int = 38
    grid_w: int = 38

    step_px: float = 256.0
    min_start_goal_px: float = 2560.0
    max_distance_factor: float = 2.0
    # Half-width of the square goal region, matching the proposed pipeline.
    goal_tol_px: float = 256.0

    covariance_softmax_temperature: float = 0.05

    heading_noise_std: float = 0.13
    magnitude_noise_std: float = 32.0
    wind_mean: float = 16.0
    wind_std: float = 8.0
    gust_std: float = 3.0

    progress_scale: float = 1 / 256
    step_penalty: float = -0.01
    goal_reward: float = 20.0
    boundary_penalty: float = -10.0
    max_steps: int = 256

    batch: int = 128
    replay: int = 200000
    warmup: int = 2000
    gamma: float = 0.99
    feature_dim: int = 128

    eval_every: int = 100
    ckpt_every: int = 250
    patience: int = 8
    min_delta: float = 0.05
    val_eps_per_pair: int = 10
    test_eps_per_pair: int = 10

    episodes_per_pair_block: int = 4
    update_chunk_steps: int = 32
    similarity_lru_pairs: int = 12

    seed: int = 0
    device: str = "cuda"


@dataclass
class DQNConfig(Common):
    lr: float = 1e-4
    target_update: int = 1000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 20000


@dataclass
class DDPGConfig(Common):
    actor_lr: float = 3e-5
    critic_lr: float = 1e-4
    tau: float = 0.005
    noise_start: float = 0.25
    noise_end: float = 0.05
    noise_decay: int = 25000


def apply_dataset_profile(cfg: Common, dataset_kind: str) -> Common:
    """Return cfg with only dataset-dependent geometry/split values changed."""
    kind = dataset_kind.lower()
    if kind == "swissview":
        return replace(
            cfg,
            dataset_kind="swissview",
            train_scenes=38,
            test_scenes=10,
            val_scenes=4,
            map_min=0.0,
            map_max=10000.0,
            grid_h=38,
            grid_w=38,
            step_px=256.0,
            min_start_goal_px=2560.0,
            goal_tol_px=256.0,
            magnitude_noise_std=32.0,
            wind_mean=16.0,
            wind_std=8.0,
            gust_std=3.0,
            progress_scale=1.0 / 256.0,
        )
    if kind == "xbd":
        # 100 selected xBD pre/post scenes -> 60/20/20 scene split.
        # 128x128 crop, stride 64 -> 15x15 descriptor grid on 1024x1024.
        # Motion/noise magnitudes are scaled by 64/256 = 1/4 from SwissView.
        return replace(
            cfg,
            dataset_kind="xbd",
            train_scenes=80,   # 60 train + 20 val before train_val_split()
            test_scenes=20,
            val_scenes=20,
            map_min=0.0,
            map_max=1024.0,
            grid_h=15,
            grid_w=15,
            step_px=64.0,
            min_start_goal_px=640.0,
            goal_tol_px=64.0,
            magnitude_noise_std=8.0,
            wind_mean=4.0,
            wind_std=2.0,
            gust_std=0.75,
            progress_scale=1.0 / 64.0,
        )
    raise ValueError(f"Unknown dataset kind: {dataset_kind!r}")


def default_train_min_start_goal(dataset_kind: str) -> float:
    return default_eval_min_start_goal(dataset_kind)
