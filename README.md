# HunterGatherers

Patch-based hunter-gatherer reinforcement learning environments.

The first environment is `HunterGathererPatchEnv`, a hard-border local camp patch
with deterministic macro-position relocation.

## Current Environment

`HunterGathererPatchEnv v0.1` includes:

- 31 x 31 default local camp grid
- two bands with 15 members each
- one member maximum per grid cell
- terrain, plant food, animal density, water, danger, depletion, and trail memory layers
- seasonal resource dynamics
- hard-border local movement
- directional camp relocation through macro coordinates
- deterministic patch generation from `(global_seed, macro_x, macro_y)`
- Gymnasium-style `reset` and `step`

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
