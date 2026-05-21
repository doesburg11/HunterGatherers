# HunterGatherers

Patch-based hunter-gatherer reinforcement learning environments.

The first environment is `HunterGathererPatchEnv`, a hard-border local camp
patch with automatic camp relocation when nearby resources are depleted.

## Current Environment

`HunterGathererPatchEnv v0.1` includes:

- 31 x 31 default local camp grid
- configurable bands with 15 members each by default
- one member maximum per grid cell
- grass, water, edible plant energy, camp, member, energy, hydration, and
  season layers
- seasonal resource dynamics
- hard-border local movement
- automatic camp relocation when the plant area around camp is depleted,
  without regenerating the local patch
- deterministic patch generation from `(global_seed, macro_x, macro_y)`
- Gymnasium-style `reset` and `step`

## Environment Model

The environment is a small test bed for hunter-gatherer band behavior. Each
band starts with 15 humans by default; experiments can use a single band or
multiple bands through `PatchEnvConfig`.

- Terrain is intentionally simple: cells are either grass or water.
- Plants are the only edible resource. Plant cells store direct energy, and
  members automatically eat from a plant cell when they enter it.
- By default, a full plant cell holds up to 8 energy and entering a plant cell
  transfers up to 2 energy.
- Members also need water. Hydration drops every step, and members
  automatically drink when they enter a water cell.
- Band camps relocate automatically when the plant cells around camp are
  depleted. The new camp's foraging radius cannot overlap the old camp's
  foraging radius. Relocation does not regenerate the river, lake, terrain, or
  local resource layers.
- Step info includes `camp_potential_energy`, the unreaped plant energy still
  available within the current camp radius.
- Seasonal effects change plant energy and movement cost.
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
```

## Observation

The observation is a CNN-friendly `Box(0, 1)` with shape:

```text
11 x obs_range x obs_range
```

Channels:

```text
0 terrain
1 plant energy, normalized to 0..1
2 reserved zero plane
3 water
4 reserved zero plane
5 camp location
6 plant depletion around resource cells
7 agent position
8 energy scalar plane
9 hydration scalar plane
10 season scalar plane
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

actions = {"band_0_member_0": Action.MOVE_EAST}
observations, rewards, terminateds, truncateds, infos = env.step(actions)
```

Missing member actions default to `STAY`. Camp relocation is not an agent
action; it is triggered automatically by depletion around the camp.

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

The right sidebar shows camp potential energy, depletion, a legend, and member
resource bars. Viewer values and the example environment parameters live in
`examples/pygame_viewer_config.toml`.

Controls:

```text
arrow keys        move
space             stay
a                 toggle random autoplay
n                 reset with a new seed
escape            quit
```

Use `python -m examples.pygame_random_band_viewer` to watch all living band 0
members act through independent random local actions.
