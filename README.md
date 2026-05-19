# HunterGatherers

Patch-based hunter-gatherer reinforcement learning environments.

The first environment is `HunterGathererPatchEnv`, a hard-border local camp
patch with directional camp relocation bookkeeping.

## Current Environment

`HunterGathererPatchEnv v0.1` includes:

- 31 x 31 default local camp grid
- configurable bands with 15 members each by default
- one member maximum per grid cell
- grass, water, edible plant food, camp, member, energy, and season layers
- seasonal resource dynamics
- hard-border local movement
- directional camp relocation through macro coordinates without regenerating
  the local patch
- deterministic patch generation from `(global_seed, macro_x, macro_y)`
- Gymnasium-style `reset` and `step`

## Environment Model

The environment is a small test bed for hunter-gatherer band behavior. Each
band starts with 15 humans by default; experiments can use a single band or
multiple bands through `PatchEnvConfig`.

- Terrain is intentionally simple: cells are either grass or water.
- Plants are the only edible resource. Gathering lowers plant food in the
  current cell, and plant food regrows over time.
- Band camps can relocate when local food, water, or seasonal pressure makes
  the current camp too costly to sustain. Relocation does not regenerate the
  river, lake, terrain, or local resource layers.
- Seasonal effects change plant food and movement cost.
- Each environment step represents one month in the yearly cycle.

Environment defaults and resource tables live in
`hunter_gatherers/envs/patch_env_config.toml`. Runtime code loads those values
into `PatchEnvConfig`, and tests or experiments can still override individual
settings with `PatchEnvConfig(...)` or `PatchEnvConfig.from_toml(...)`.

## Actions

```text
0 stay
1 move north
2 move south
3 move west
4 move east
5 gather
6 hunt placeholder
7 rest
8 move camp north
9 move camp south
10 move camp west
11 move camp east
```

## Observation

The observation is a CNN-friendly `Box(0, 1)` with shape:

```text
10 x obs_range x obs_range
```

Channels:

```text
0 terrain
1 plant food
2 reserved zero plane
3 water
4 reserved zero plane
5 camp location
6 reserved zero plane
7 agent position
8 energy scalar plane
9 season scalar plane
```

## Quick Check

```bash
python -m unittest discover -s tests
python -m examples.random_rollout
```

## Band Member Multi-Agent API

`BandMemberPatchEnv` exposes an RLlib-shaped dict API without requiring Ray.
It controls the living members of one band and keeps the original patch,
resource, collision, and lifecycle state shared.

```python
from hunter_gatherers import BandMemberPatchEnv, PatchEnvConfig
from hunter_gatherers.envs.patch_env import Action

env = BandMemberPatchEnv(PatchEnvConfig(members_per_band=3))
observations, infos = env.reset(seed=123)

actions = {"band_0_member_0": Action.GATHER}
observations, rewards, terminateds, truncateds, infos = env.step(actions)
```

Missing member actions default to `STAY`. For now, only
`band_0_member_0` can move camp; other camp-move actions are treated as
`STAY`.

## RLlib Adapter

`RllibBandMemberPatchEnv` subclasses RLlib's `MultiAgentEnv` and delegates to
`BandMemberPatchEnv`. Ray is optional:

```bash
pip install "hunter-gatherers[rllib]"
python -m examples.rllib_band_ppo --iterations 1
```

All band members map to one shared PPO policy in the example. Use the Ray-free
`BandMemberPatchEnv` directly for local simulation tests that should not depend
on Ray.

## Pygame Viewer

```bash
python -m examples.pygame_viewer
python -m examples.pygame_random_band_viewer
```

The right sidebar shows current episode stats plus a legend for cells and
symbols. Viewer values live in
`examples/pygame_viewer_config.toml`.

Controls:

```text
arrow keys        move
space             stay
g                 gather
r                 rest
shift + arrow     move camp
a                 toggle random autoplay
n                 reset with a new seed
escape            quit
```

Use `python -m examples.pygame_random_band_viewer` to watch all living band 0
members act through independent random local actions.
