# GPS-Denied Visual Navigation: Pipeline and Baselines

This repository contains the code for the proposed online GPS-denied
navigator and the two baselines it is compared against:

- `online_navigator/` — **the proposed pipeline**: an online, unknown-start
  navigator that tracks a particle-filter (PF) belief over position and
  falls back to a risk-aware Monte-Carlo recovery planner when localization
  becomes unreliable.
- `baselines/rl_baselines/` — Double DQN and DDPG baselines trained with a
  compact, memoryless 5-D state (no PF, no recurrent state, no true pose).
- `baselines/slam_astar/` — a classical SLAM (visual odometry + map
  registration via SIFT/RANSAC + pose-graph optimization) + A* baseline.

All three methods share the same simulator conventions (motion/noise model,
success region, dataset geometry) and, for the paper's final numbers, the
same frozen episode manifest, so results are directly comparable.

Two dataset profiles are supported end to end: **SwissView** (10000x10000
maps) and **xBD** (1024x1024 maps, geometry scaled by 1/4). Pass `--dataset
swissview` or `--dataset xbd` to any script below.

## Repository layout

```
code/
├── requirements.txt
├── online_navigator/        # proposed pipeline
│   ├── main.py               # run one episode (debugging/visualization)
│   ├── benchmark.py          # run the full evaluation suite
│   └── ...
└── baselines/
    ├── rl_baselines/
    │   ├── train_dqn.py
    │   ├── train_ddpg.py
    │   ├── eval_rl_baselines.py
    │   └── ...
    └── slam_astar/
        ├── main.py            # run one episode (debugging/visualization)
        ├── benchmark.py       # run the full evaluation suite
        └── ...
```

Each directory is a flat, self-contained script collection (imports are not
package-relative), so every command below is run with that directory as the
current working directory, e.g. `cd code/online_navigator && python ...`.

## Installation

Requires Python 3.10+ (some modules use bare `X | Y` / `list[...]` type
hints without `from __future__ import annotations`).

```bash
pip install -r requirements.txt
```

### External dependencies

- **AnyLoc** (DINOv2 + VLAD retrieval, https://github.com/AnyLoc/AnyLoc) is
  required for (a) generating the descriptor caches consumed by all three
  methods, and (b) the SLAM+A* baseline's online query encoder
  (`baselines/slam_astar/anyloc_query.py`), which imports
  `utilities.py` from AnyLoc's `demo/` directory at runtime. Clone AnyLoc
  separately and install its own requirements, then point `--anyloc-root`
  at the clone (e.g. `/path/to/AnyLoc`).
- **Precomputed descriptor caches.** All three methods assume that, for
  every scene, AnyLoc descriptors have already been extracted on a regular
  grid and saved as `.pt` files containing at least `positions`
  (`[N, 2]`) and `descriptors` (`[N, D]`) tensors, with `N` equal to the
  grid size for the chosen dataset profile (38x38 = 1444 for SwissView,
  15x15 = 225 for xBD). This caching step is not part of this repository;
  point `--cache-root` at wherever those caches already live, mirroring the
  scene-folder layout of `--dataset-root`.
- For xBD, the frozen 100 pre/post scene-pair subset used for training and
  evaluation is always inferred directly from which scene folders exist
  under `--cache-root` (exactly 100 are expected): whichever 100 scenes you
  cache there are the ones used.

## Dataset layout

```
/path/to/swissimage/swisstopo_00/...      # or /path/to/xBD/<scene>/...
/path/to/AnyLoc/outputs/swisstopo_00/*.pt # or .../outputs_xbd/<scene>/*.pt
```

`--dataset-root` holds the raw scene images; `--cache-root` holds the
matching AnyLoc descriptor caches, one subfolder per scene.

## Step 1: generate the frozen shared benchmark manifest

Run once per dataset profile before evaluating any method. All three
methods read the same manifest via `--episodes-manifest`, which freezes the
test scenes, ordered reference/deployment pairs, per-episode seeds, and
start/goal coordinates so DQN, DDPG, SLAM+A*, and the proposed pipeline are
evaluated on identical episodes.

```bash
# SwissView
python generate_benchmark_episodes.py \
  --dataset swissview \
  --dataset-root /path/to/swissimage \
  --cache-root /path/to/AnyLoc/outputs \
  --out manifests/swissview_manifest.json \
  --episodes-per-pair 10

# xBD
python generate_benchmark_episodes.py \
  --dataset xbd \
  --dataset-root /path/to/xBD \
  --cache-root /path/to/AnyLoc/outputs_xbd \
  --out manifests/xbd_manifest.json \
  --episodes-per-pair 10
```

Run from any of the three directories. Write it once to a shared location
(e.g. a top-level `manifests/` folder) and pass that same path to
`--episodes-manifest` in every command below.

## Step 2a: train and evaluate the RL baselines (`baselines/rl_baselines`)

```bash
cd baselines/rl_baselines
```

Train Double DQN and DDPG (SwissView):

```bash
python train_dqn.py \
  --dataset swissview \
  --dataset-root /path/to/swissimage \
  --cache-root /path/to/AnyLoc/outputs \
  --out runs/swissview_double_dqn \
  --episodes 20000 \
  --device cuda

python train_ddpg.py \
  --dataset swissview \
  --dataset-root /path/to/swissimage \
  --cache-root /path/to/AnyLoc/outputs \
  --out runs/swissview_ddpg \
  --episodes 20000 \
  --device cuda
```

Train on xBD:

```bash
python train_dqn.py \
  --dataset xbd \
  --dataset-root /path/to/xBD \
  --cache-root /path/to/AnyLoc/outputs_xbd \
  --out runs/xbd_double_dqn \
  --episodes 20000 \
  --device cuda

python train_ddpg.py \
  --dataset xbd \
  --dataset-root /path/to/xBD \
  --cache-root /path/to/AnyLoc/outputs_xbd \
  --out runs/xbd_ddpg \
  --episodes 20000 \
  --device cuda
```

Evaluate both checkpoints on the frozen shared manifest (recommended) —
this is the paper-comparable path:

```bash
python eval_rl_baselines.py \
  --dataset xbd \
  --episodes-manifest /path/to/manifests/xbd_manifest.json \
  --dqn-checkpoint runs/xbd_double_dqn/best.pt \
  --ddpg-checkpoint runs/xbd_ddpg/best.pt \
  --out-dir runs/xbd_eval \
  --device cuda
```

Without `--episodes-manifest`, `eval_rl_baselines.py` builds the same
manifest `generate_benchmark_episodes.py` would from
`--dataset-root`/`--cache-root`/`--split-seed`/`--eval-seed`, using the same
default minimum start/goal distance — pass `--min-start-goal-distance` to
override it.

## Step 2b: evaluate the SLAM+A* baseline (`baselines/slam_astar`)

```bash
cd baselines/slam_astar
```

Recommended path, using the frozen shared manifest:

```bash
python benchmark.py \
  --dataset xbd \
  --episodes-manifest /path/to/manifests/xbd_manifest.json \
  --out runs/xbd_slam_astar \
  --anyloc-root /path/to/AnyLoc \
  --anyloc-device cuda \
  --reference-feature-cache-size 225
```

```bash
python benchmark.py \
  --dataset swissview \
  --episodes-manifest /path/to/manifests/swissview_manifest.json \
  --out runs/swissview_slam_astar \
  --anyloc-root /path/to/AnyLoc \
  --anyloc-device cuda
```

For debugging or visualizing one episode (writes `trajectory.png` and the
true/estimated trajectories):

```bash
python main.py \
  --dataset swissview \
  --dataset-root /path/to/swissimage \
  --reference-cache /path/to/AnyLoc/outputs/swisstopo_00/ref.pt \
  --deployment-cache /path/to/AnyLoc/outputs/swisstopo_00/obs.pt \
  --out runs/one_episode \
  --anyloc-root /path/to/AnyLoc \
  --anyloc-device cuda
```

## Step 3: evaluate the proposed pipeline (`online_navigator`)

```bash
cd online_navigator
```

Recommended path, using the frozen shared manifest — `--dataset` is
inferred from the manifest if omitted:

```bash
python benchmark.py \
  --episodes-manifest /path/to/manifests/xbd_manifest.json \
  --out runs/xbd_online \
  --mode risk_aware
```

```bash
python benchmark.py \
  --episodes-manifest /path/to/manifests/swissview_manifest.json \
  --out runs/swissview_online \
  --mode risk_aware
```

`--mode goal_only` runs the ablation that always steers from the PF's
robust center toward the goal, without the risk detector or recovery
planner. A diagnostic GIF of the PF belief and trajectory is saved for every
episode by default (`save_gif` in `config.py`); add `--background-tif` to
render it over each episode's actual deployment image instead of a blank
canvas. Neither affects navigation.

For a single debugging episode, define your preferred start and goal (the
start location stays unexposed to the navigator):

```bash
python main.py \
  --dataset swissview \
  --reference-cache /path/to/AnyLoc/outputs/swisstopo_00/ref.pt \
  --deployment-cache /path/to/AnyLoc/outputs/swisstopo_00/obs.pt \
  --start 8119.61 5572.17 \
  --goal 7152.26 4374.66 \
  --out runs/one_episode \
  --mode risk_aware \
  --seed 2100003
```

## Success criteria

All three methods use the same goal-centered square success region: the
true UAV position must end inside a square centered on the goal with
half-width equal to half of one deployment observation crop (256 px on
SwissView, 64 px on xBD), within a travel budget of `2x` the nominal
start-goal distance. The proposed pipeline additionally reports
`real_success` (`vanilla_success` plus the PF belief mass within a larger
region around the goal exceeding a threshold), since it is the only method
that must decide when to stop searching without ever seeing ground-truth
position.

## Outputs

`online_navigator/benchmark.py` and `baselines/slam_astar/benchmark.py`
write, under `--out`: `manifest.json` (the exact episode definitions used),
`episodes.json` (per-episode results), and `summary.json` (aggregate
success rate / SPL / SCT with bootstrap 95% CIs, plus method-specific
diagnostics such as PF recovery statistics or SLAM localization error).
`baselines/rl_baselines/eval_rl_baselines.py` writes the manifest actually
used as `episodes_manifest_used.json` under `--out-dir`, plus
`dqn_episodes.json` / `ddpg_episodes.json` and a `summary.json` with
SR/SPL/SCT for each of DQN and DDPG.
