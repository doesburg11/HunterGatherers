# HunterGatherers

Patch-based hunter-gatherer reinforcement learning environments.

The first environment is `HunterGathererPatchEnv`, a hard-border local camp patch
with deterministic macro-position relocation.

## Current Environment

`HunterGathererPatchEnv v0.1` includes:

- 31 x 31 default local camp grid
- configurable bands with 15 members each by default
- one member maximum per grid cell
- terrain, plant food, animal density, water, danger, depletion, and trail memory layers
- seasonal resource dynamics
- hard-border local movement
- directional camp relocation through macro coordinates
- deterministic patch generation from `(global_seed, macro_x, macro_y)`
- Gymnasium-style `reset` and `step`

## Environment Model

The environment is a test bed for hunter-gatherer band behavior. Each band
starts with 15 humans by default; experiments can use a single band or multiple
bands through `PatchEnvConfig`.

- Humans use plants and animals as food resources, and water is represented as a
  terrain/resource layer.
- Grassland and water availability shape animal density and food availability.
- The controlled agent can move through the local patch, and animal density
  shifts over time as animals diffuse, regrow, and avoid depleted or heavily
  traveled cells.
- Band camps can relocate when local food, water, danger, depletion, or seasonal
  pressure makes the current camp too costly to sustain.
- Seasonal effects change the environment by reducing or increasing maximum
  plant food, animal density, water pressure, movement cost, and risk.
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
6 hunt
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
2 animal density
3 water
4 danger
5 camp location
6 depletion
7 agent position
8 energy scalar plane
9 season scalar plane
```

## Quick Check

```bash
python -m unittest discover -s tests
python -m examples.random_rollout
```

## Pygame Viewer

```bash
python -m examples.pygame_viewer
```

The right sidebar shows current episode stats plus a legend for terrain,
symbols, and overlays. Viewer values live in
`examples/pygame_viewer_config.toml`.

Controls:

```text
arrow keys        move
space             stay
g                 gather
h                 hunt
r                 rest
shift + arrow     move camp
1                 plant overlay
2                 animal overlay
3                 danger overlay
4                 depletion overlay
a                 toggle random autoplay
n                 reset with a new seed
escape            quit
```
