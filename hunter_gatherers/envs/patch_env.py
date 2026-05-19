from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Terrain(IntEnum):
    GRASSLAND = 0
    WOODLAND = 1
    DENSE_FOREST = 2
    WATER = 3
    HILL = 4
    MARSH = 5


class Season(IntEnum):
    SPRING = 0
    SUMMER = 1
    AUTUMN = 2
    WINTER = 3


class Action(IntEnum):
    STAY = 0
    MOVE_NORTH = 1
    MOVE_SOUTH = 2
    MOVE_WEST = 3
    MOVE_EAST = 4
    GATHER = 5
    HUNT = 6
    REST = 7
    MOVE_CAMP_NORTH = 8
    MOVE_CAMP_SOUTH = 9
    MOVE_CAMP_WEST = 10
    MOVE_CAMP_EAST = 11


@dataclass(frozen=True)
class PatchEnvConfig:
    grid_size: int = 31
    obs_range: int = 11
    max_steps: int = 2000
    num_bands: int = 2
    members_per_band: int = 15
    initial_energy: float = 100.0
    max_energy: float = 150.0
    movement_cost: float = 1.0
    rest_recovery: float = 0.5
    plant_food_gain: float = 8.0
    animal_food_gain: float = 20.0
    gather_amount: float = 0.25
    hunt_amount: float = 0.2
    plant_regrowth_base: float = 0.01
    animal_regrowth_base: float = 0.005
    season_length: int = 250
    camp_move_cost: float = 20.0
    camp_move_distance: int = 10
    danger_damage_scale: float = 2.0
    trail_memory_decay: float = 0.995
    depletion_decay: float = 0.0005
    global_seed: int = 12345


TERRAIN_PLANT_BASE = np.array(
    [0.55, 0.75, 0.65, 0.05, 0.18, 0.45],
    dtype=np.float32,
)
TERRAIN_ANIMAL_BASE = np.array(
    [0.72, 0.58, 0.42, 0.18, 0.22, 0.48],
    dtype=np.float32,
)
TERRAIN_MOVEMENT_COST = np.array(
    [1.0, 1.35, 1.8, 3.0, 2.0, 2.2],
    dtype=np.float32,
)
TERRAIN_DANGER_BASE = np.array(
    [0.35, 0.2, 0.35, 0.1, 0.55, 0.6],
    dtype=np.float32,
)

SEASON_PLANT_MOD = np.array([1.35, 1.15, 0.75, 0.15], dtype=np.float32)
SEASON_ANIMAL_MOD = np.array([0.85, 1.15, 1.05, 0.55], dtype=np.float32)
SEASON_COST_MOD = np.array([1.05, 0.9, 1.05, 1.35], dtype=np.float32)
SEASON_RISK_MOD = np.array([0.85, 0.8, 1.05, 1.3], dtype=np.float32)


class HunterGathererPatchEnv(gym.Env[np.ndarray, int]):
    """Hard-border local patch env with deterministic macro relocation."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, config: PatchEnvConfig | dict[str, Any] | None = None):
        super().__init__()
        if config is None:
            self.config = PatchEnvConfig()
        elif isinstance(config, PatchEnvConfig):
            self.config = config
        else:
            self.config = replace(PatchEnvConfig(), **config)

        if self.config.grid_size < 5 or self.config.grid_size % 2 == 0:
            raise ValueError("grid_size must be an odd integer >= 5")
        if self.config.obs_range < 3 or self.config.obs_range % 2 == 0:
            raise ValueError("obs_range must be an odd integer >= 3")
        if self.config.season_length <= 0:
            raise ValueError("season_length must be positive")
        if self.config.num_bands <= 0:
            raise ValueError("num_bands must be positive")
        if self.config.members_per_band <= 0:
            raise ValueError("members_per_band must be positive")

        self.grid_size = self.config.grid_size
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(10, self.config.obs_range, self.config.obs_range),
            dtype=np.float32,
        )

        self._episode_rng = np.random.default_rng(self.config.global_seed)
        self._patch_rng = np.random.default_rng(self.config.global_seed)
        self._last_food_gained = 0.0
        self._last_energy_spent = 0.0
        self._last_danger_damage = 0.0
        self._last_camp_pressure = 0.0
        self._last_action = Action.STAY

        self.band_camp_positions: np.ndarray = np.empty(
            (self.config.num_bands, 2),
            dtype=np.int16,
        )
        self.member_positions: np.ndarray = np.empty(
            (
                self.config.num_bands,
                self.config.members_per_band,
                2,
            ),
            dtype=np.int16,
        )
        self.reset()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._episode_rng = np.random.default_rng(seed)

        options = options or {}
        self.step_count = 0
        self.day = 0
        self.season = Season.SPRING
        self.camp_id = 0
        self.macro_x = int(options.get("macro_x", 0))
        self.macro_y = int(options.get("macro_y", 0))
        self.camp_pos = (self.grid_size // 2, self.grid_size // 2)
        self.band_camp_positions = np.zeros(
            (self.config.num_bands, 2),
            dtype=np.int16,
        )
        self.member_positions = np.zeros(
            (
                self.config.num_bands,
                self.config.members_per_band,
                2,
            ),
            dtype=np.int16,
        )
        self.agent_pos = self.camp_pos
        self.energy = float(
            options.get("initial_energy", self.config.initial_energy)
        )
        self.stored_food = float(options.get("stored_food", 0.0))
        self.population = self.config.num_bands * self.config.members_per_band
        self.camp_age = 0
        self.local_depletion_level = 0.0
        self.num_camp_moves = 0
        self.successful_hunts = 0
        self.gathering_events = 0
        self.starvation_events = 0
        self.macro_distance_traveled = 0

        self._generate_new_patch()
        self._place_bands()
        self._last_food_gained = 0.0
        self._last_energy_spent = 0.0
        self._last_danger_damage = 0.0
        self._last_camp_pressure = self._camp_pressure()
        self._last_action = Action.STAY
        return self._build_observation(), self._build_info()

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = Action(int(action))
        self.step_count += 1
        self.day = self.step_count
        self.season = Season(
            (self.step_count // self.config.season_length) % len(Season)
        )
        self.camp_age += 1

        self._last_action = action
        self._last_food_gained = 0.0
        self._last_energy_spent = 0.0
        self._last_danger_damage = 0.0

        self.trail_memory *= self.config.trail_memory_decay
        y, x = self.agent_pos
        self.trail_memory[y, x] = min(1.0, self.trail_memory[y, x] + 0.08)

        if action in {
            Action.MOVE_NORTH,
            Action.MOVE_SOUTH,
            Action.MOVE_WEST,
            Action.MOVE_EAST,
        }:
            self._move_agent(action)
        elif action == Action.GATHER:
            self._gather()
        elif action == Action.HUNT:
            self._hunt()
        elif action == Action.REST:
            self._rest()
        elif action in {
            Action.MOVE_CAMP_NORTH,
            Action.MOVE_CAMP_SOUTH,
            Action.MOVE_CAMP_WEST,
            Action.MOVE_CAMP_EAST,
        }:
            self._move_camp(action)
        else:
            self._spend_energy(0.15)

        if action not in {
            Action.MOVE_CAMP_NORTH,
            Action.MOVE_CAMP_SOUTH,
            Action.MOVE_CAMP_WEST,
            Action.MOVE_CAMP_EAST,
        }:
            self._apply_danger()
            self._update_resources()
            self._update_animals()
            self.local_depletion_level = float(np.mean(self.depletion))

        self._last_camp_pressure = self._camp_pressure()
        terminated = self.energy <= 0.0
        if terminated:
            self.starvation_events += 1
            self.energy = 0.0
        truncated = self.step_count >= self.config.max_steps
        reward = self._calculate_reward(terminated)
        return (
            self._build_observation(),
            reward,
            terminated,
            truncated,
            self._build_info(),
        )

    def render(self) -> str:
        chars = {
            Terrain.GRASSLAND: ".",
            Terrain.WOODLAND: "w",
            Terrain.DENSE_FOREST: "f",
            Terrain.WATER: "~",
            Terrain.HILL: "^",
            Terrain.MARSH: "m",
        }
        lines: list[str] = []
        member_chars = self._member_char_layer()
        for y in range(self.grid_size):
            row: list[str] = []
            for x in range(self.grid_size):
                member_char = member_chars[y][x]
                if member_char:
                    row.append(member_char)
                elif (y, x) == self.camp_pos:
                    row.append("C")
                else:
                    row.append(chars[Terrain(int(self.terrain[y, x]))])
            lines.append("".join(row))
        return "\n".join(lines)

    def _member_char_layer(self) -> list[list[str]]:
        layer = [
            ["" for _ in range(self.grid_size)]
            for _ in range(self.grid_size)
        ]
        for band_id in range(self.config.num_bands):
            for member_id in range(self.config.members_per_band):
                y, x = self._member_position(band_id, member_id)
                if band_id == 0 and member_id == 0:
                    layer[y][x] = "A"
                elif band_id == 0:
                    layer[y][x] = "a"
                elif band_id == 1:
                    layer[y][x] = "b"
                else:
                    layer[y][x] = str(min(9, band_id))
        return layer

    def _generate_new_patch(self) -> None:
        seed = self._patch_seed(
            self.config.global_seed,
            self.macro_x,
            self.macro_y,
        )
        self._patch_rng = np.random.default_rng(seed)
        self.terrain = self._generate_terrain()
        self.water = (self.terrain == Terrain.WATER).astype(np.float32)
        self.plant_food = self._generate_plants()
        self.animal_density = self._generate_animals()
        self.danger = self._generate_danger()
        self.depletion = np.zeros(
            (self.grid_size, self.grid_size),
            dtype=np.float32,
        )
        self.trail_memory = np.zeros(
            (self.grid_size, self.grid_size),
            dtype=np.float32,
        )

        cy, cx = self.camp_pos
        if self.terrain[cy, cx] == Terrain.WATER:
            self.terrain[cy, cx] = Terrain.GRASSLAND
            self.water[cy, cx] = 0.0

    def _place_bands(self) -> None:
        occupied: set[tuple[int, int]] = set()
        for band_id in range(self.config.num_bands):
            anchor = self._band_anchor(band_id)
            camp = self._nearest_open_land_cell(anchor, occupied)
            self.band_camp_positions[band_id] = camp
            if band_id == 0:
                self.camp_pos = camp
            members = self._member_cells_near(camp, occupied)
            self.member_positions[band_id] = members
            occupied.update((int(y), int(x)) for y, x in members)

        self.agent_pos = self._member_position(0, 0)

    def _band_anchor(self, band_id: int) -> tuple[int, int]:
        center = self.grid_size // 2
        if self.config.num_bands == 1:
            return center, center
        if band_id == 0:
            return center, center
        if band_id == 1:
            return center, min(
                self.grid_size - 2,
                center + self.grid_size // 5,
            )

        angle = 2.0 * np.pi * band_id / self.config.num_bands
        radius = max(2, self.grid_size // 4)
        y = int(
            np.clip(center + np.sin(angle) * radius, 1, self.grid_size - 2)
        )
        x = int(
            np.clip(center + np.cos(angle) * radius, 1, self.grid_size - 2)
        )
        return y, x

    def _nearest_open_land_cell(
        self,
        start: tuple[int, int],
        occupied: set[tuple[int, int]],
    ) -> tuple[int, int]:
        candidates = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if (y, x) in occupied:
                    continue
                if self.terrain[y, x] == Terrain.WATER:
                    continue
                distance = abs(y - start[0]) + abs(x - start[1])
                candidates.append((distance, y, x))
        if not candidates:
            raise RuntimeError(
                "no open land cells available for band placement"
            )
        _, y, x = min(candidates)
        return int(y), int(x)

    def _member_cells_near(
        self,
        camp: tuple[int, int],
        occupied: set[tuple[int, int]],
    ) -> np.ndarray:
        candidates = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if (y, x) in occupied:
                    continue
                if self.terrain[y, x] == Terrain.WATER:
                    continue
                distance = abs(y - camp[0]) + abs(x - camp[1])
                candidates.append((distance, y, x))

        needed = self.config.members_per_band
        if len(candidates) < needed:
            raise RuntimeError("not enough open land cells for band members")

        candidates.sort()
        cells = [(y, x) for _, y, x in candidates[:needed]]
        return np.array(cells, dtype=np.int16)

    def _member_position(
        self,
        band_id: int,
        member_id: int,
    ) -> tuple[int, int]:
        y, x = self.member_positions[band_id, member_id]
        return int(y), int(x)

    def _set_member_position(
        self,
        band_id: int,
        member_id: int,
        pos: tuple[int, int],
    ) -> None:
        y, x = pos
        self.member_positions[band_id, member_id] = (y, x)

    def _cell_has_member(
        self,
        pos: tuple[int, int],
        ignore: tuple[int, int] | None = None,
    ) -> bool:
        y, x = pos
        for band_id in range(self.config.num_bands):
            for member_id in range(self.config.members_per_band):
                if ignore == (band_id, member_id):
                    continue
                member_y, member_x = self._member_position(band_id, member_id)
                if (member_y, member_x) == (y, x):
                    return True
        return False

    def _generate_terrain(self) -> np.ndarray:
        size = self.grid_size
        yy, xx = np.mgrid[0:size, 0:size]
        northness = np.clip(-self.macro_y / 100.0, -1.0, 1.0)
        eastness = np.clip(self.macro_x / 100.0, -1.0, 1.0)

        lake_center_x = size // 2 + self._patch_rng.integers(
            -size // 10,
            size // 10 + 1,
        )
        lake_center_y = max(3, int(size * 0.18)) + self._patch_rng.integers(
            -1,
            2,
        )
        lake_radius_x = self._patch_rng.uniform(size * 0.12, size * 0.17)
        lake_radius_y = self._patch_rng.uniform(size * 0.07, size * 0.11)
        lake_shape = (
            ((xx - lake_center_x) / lake_radius_x) ** 2
            + ((yy - lake_center_y) / lake_radius_y) ** 2
        )
        lake_mask = lake_shape <= 1.0

        phase = self._patch_rng.uniform(0, np.pi * 2)
        secondary_phase = self._patch_rng.uniform(0, np.pi * 2)
        amplitude = self._patch_rng.uniform(1.8, 3.8)
        secondary_amplitude = self._patch_rng.uniform(0.5, 1.4)
        offset = self._patch_rng.integers(-size // 12, size // 12 + 1)
        center = int(size * 0.70) + offset
        river_progress = xx / max(size - 1, 1)
        river_y = (
            center
            + amplitude * np.sin(river_progress * np.pi * 2 + phase)
            + secondary_amplitude
            * np.sin(river_progress * np.pi * 5 + secondary_phase)
        )
        river_min_y = max(
            int(size * 0.58),
            int(lake_center_y + lake_radius_y + 4),
        )
        river_y = np.clip(river_y, river_min_y, size - 3)
        river_width = self._patch_rng.uniform(1.2, 1.9)
        river_mask = np.abs(yy - river_y) <= river_width

        water_mask = river_mask | lake_mask

        water_distance = self._distance_from_mask(water_mask)
        water_influence = 1.0 - np.clip(water_distance / (size * 0.35), 0, 1)
        elevation = self._normalize_field(
            0.75 * self._smooth_noise(size, passes=7)
            + 0.25 * self._smooth_noise(size, passes=3)
            - 0.14 * water_influence
        )
        moisture = self._normalize_field(
            0.62 * self._smooth_noise(size, passes=7)
            + 0.30 * water_influence
            - 0.08 * elevation
        )
        tree_cover = self._normalize_field(
            0.68 * self._smooth_noise(size, passes=7)
            + 0.24 * moisture
            - 0.08 * elevation
        )

        hill_threshold = 0.75 - 0.05 * max(-eastness, 0.0)
        dense_threshold = 0.71 - 0.08 * northness
        woodland_threshold = 0.45 - 0.06 * northness
        marsh_threshold = 0.70 - 0.05 * max(eastness, 0.0)

        terrain = np.full(
            (size, size),
            Terrain.GRASSLAND,
            dtype=np.int8,
        )

        hill_mask = elevation > hill_threshold
        marsh_mask = (
            (moisture > marsh_threshold)
            & (elevation < 0.62)
            & (water_distance <= size * 0.30)
        )
        woodland_mask = (tree_cover > woodland_threshold) & ~hill_mask
        forest_mask = (
            (tree_cover > dense_threshold)
            & (moisture > 0.35)
            & ~hill_mask
        )

        terrain[woodland_mask] = Terrain.WOODLAND
        terrain[forest_mask] = Terrain.DENSE_FOREST
        terrain[marsh_mask & ~hill_mask] = Terrain.MARSH
        terrain[hill_mask] = Terrain.HILL
        terrain[water_mask] = Terrain.WATER

        return terrain

    def _generate_plants(self) -> np.ndarray:
        base = TERRAIN_PLANT_BASE[self.terrain]
        noise = 0.65 + 0.7 * self._smooth_noise(self.grid_size, passes=2)
        plants = base * noise * SEASON_PLANT_MOD[self.season]
        plants[self.water > 0.0] = 0.0
        return np.clip(plants, 0.0, 1.0).astype(np.float32)

    def _generate_animals(self) -> np.ndarray:
        base = TERRAIN_ANIMAL_BASE[self.terrain]
        noise = 0.6 + 0.75 * self._smooth_noise(self.grid_size, passes=2)
        animals = base * noise * SEASON_ANIMAL_MOD[self.season]
        animals[self.water > 0.0] *= 0.35
        return np.clip(animals, 0.0, 1.0).astype(np.float32)

    def _generate_danger(self) -> np.ndarray:
        danger = (
            TERRAIN_DANGER_BASE[self.terrain]
            * SEASON_RISK_MOD[self.season]
        )
        danger += 0.18 * self._smooth_noise(self.grid_size, passes=3)
        water_distance = self._distance_to_nearest_water()
        danger += np.clip(water_distance / self.grid_size, 0.0, 1.0) * 0.08
        return np.clip(danger, 0.0, 1.0).astype(np.float32)

    def _smooth_noise(self, size: int, passes: int) -> np.ndarray:
        noise = self._patch_rng.random((size, size), dtype=np.float32)
        for _ in range(passes):
            padded = np.pad(noise, 1, mode="edge")
            noise = (
                padded[:-2, :-2]
                + padded[:-2, 1:-1]
                + padded[:-2, 2:]
                + padded[1:-1, :-2]
                + padded[1:-1, 1:-1]
                + padded[1:-1, 2:]
                + padded[2:, :-2]
                + padded[2:, 1:-1]
                + padded[2:, 2:]
            ) / 9.0
        return noise

    @staticmethod
    def _normalize_field(field: np.ndarray) -> np.ndarray:
        field_min = float(np.min(field))
        field_max = float(np.max(field))
        span = field_max - field_min
        if span <= 1e-8:
            return np.zeros_like(field, dtype=np.float32)
        return ((field - field_min) / span).astype(np.float32)

    @staticmethod
    def _distance_from_mask(mask: np.ndarray) -> np.ndarray:
        yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
        positions = np.argwhere(mask)
        if len(positions) == 0:
            return np.full(mask.shape, np.inf, dtype=np.float32)
        distances = np.full(mask.shape, np.inf, dtype=np.float32)
        for py, px in positions:
            distances = np.minimum(
                distances,
                np.abs(yy - py) + np.abs(xx - px),
            )
        return distances.astype(np.float32)

    def _move_agent(self, action: Action) -> None:
        dy, dx = {
            Action.MOVE_NORTH: (-1, 0),
            Action.MOVE_SOUTH: (1, 0),
            Action.MOVE_WEST: (0, -1),
            Action.MOVE_EAST: (0, 1),
        }[action]
        y, x = self.agent_pos
        ny = int(np.clip(y + dy, 0, self.grid_size - 1))
        nx = int(np.clip(x + dx, 0, self.grid_size - 1))
        if self._cell_has_member((ny, nx), ignore=(0, 0)):
            self._spend_energy(0.15)
            return
        terrain = Terrain(int(self.terrain[ny, nx]))
        move_cost = float(
            TERRAIN_MOVEMENT_COST[terrain] * SEASON_COST_MOD[self.season]
        )
        if terrain == Terrain.WATER:
            move_cost *= 1.5
        distance_cost = 0.015 * self._distance_from_camp((ny, nx))
        self._set_member_position(0, 0, (ny, nx))
        self.agent_pos = (ny, nx)
        self._spend_energy(
            self.config.movement_cost * move_cost + distance_cost
        )

    def _gather(self) -> None:
        y, x = self.agent_pos
        available = float(self.plant_food[y, x])
        gathered = min(available, self.config.gather_amount)
        self.plant_food[y, x] = max(0.0, available - gathered)
        self.depletion[y, x] = min(1.0, self.depletion[y, x] + gathered * 0.65)
        food_gained = gathered * self.config.plant_food_gain
        self._last_food_gained += food_gained
        self.energy = min(self.config.max_energy, self.energy + food_gained)
        self.gathering_events += int(gathered > 0.0)
        self._spend_energy(0.35)

    def _hunt(self) -> None:
        y, x = self.agent_pos
        available = float(self.animal_density[y, x])
        danger_penalty = float(self.danger[y, x] * 0.25)
        success_probability = np.clip(
            0.15 + available * 0.75 - danger_penalty,
            0.02,
            0.9,
        )
        self._spend_energy(1.8)
        if self._episode_rng.random() >= success_probability:
            return
        hunted = min(available, self.config.hunt_amount)
        self.animal_density[y, x] = max(0.0, available - hunted)
        self.depletion[y, x] = min(1.0, self.depletion[y, x] + hunted * 0.45)
        food_gained = hunted * self.config.animal_food_gain
        self._last_food_gained += food_gained
        self.energy = min(self.config.max_energy, self.energy + food_gained)
        self.successful_hunts += int(hunted > 0.0)

    def _rest(self) -> None:
        y, x = self.agent_pos
        camp_bonus = 0.5 if (y, x) == self.camp_pos else 0.0
        self.energy = min(
            self.config.max_energy,
            self.energy + self.config.rest_recovery + camp_bonus,
        )
        self._spend_energy(0.1)

    def _move_camp(self, action: Action) -> None:
        direction = {
            Action.MOVE_CAMP_NORTH: (0, -1),
            Action.MOVE_CAMP_SOUTH: (0, 1),
            Action.MOVE_CAMP_WEST: (-1, 0),
            Action.MOVE_CAMP_EAST: (1, 0),
        }[action]
        dx, dy = direction
        distance = self.config.camp_move_distance
        self.macro_x += dx * distance
        self.macro_y += dy * distance
        self.camp_id += 1
        self.camp_age = 0
        self.num_camp_moves += 1
        self.macro_distance_traveled += distance
        self._spend_energy(self.config.camp_move_cost)
        self.camp_pos = (self.grid_size // 2, self.grid_size // 2)
        self._generate_new_patch()
        self._place_bands()
        self.local_depletion_level = 0.0

    def _spend_energy(self, amount: float) -> None:
        amount = max(0.0, float(amount))
        self.energy -= amount
        self._last_energy_spent += amount

    def _apply_danger(self) -> None:
        y, x = self.agent_pos
        damage = float(
            self.danger[y, x] * self.config.danger_damage_scale * 0.02
        )
        self.energy -= damage
        self._last_danger_damage += damage

    def _update_resources(self) -> None:
        season_mod = float(SEASON_PLANT_MOD[self.season])
        terrain_mod = TERRAIN_PLANT_BASE[self.terrain]
        effective_regrowth = (
            self.config.plant_regrowth_base
            * season_mod
            * terrain_mod
            * (1.0 - self.depletion)
        )
        self.plant_food = np.clip(
            self.plant_food + effective_regrowth,
            0.0,
            1.0,
        ).astype(np.float32)
        self.plant_food[self.water > 0.0] = 0.0
        self.depletion = np.clip(
            self.depletion - self.config.depletion_decay,
            0.0,
            1.0,
        ).astype(np.float32)

    def _update_animals(self) -> None:
        padded = np.pad(self.animal_density, 1, mode="edge")
        neighborhood = (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + padded[1:-1, 1:-1] * 4.0
        ) / 8.0
        avoid_humans = 1.0 - np.clip(
            self.trail_memory * 0.35 + self.depletion * 0.25,
            0.0,
            0.7,
        )
        regrowth = (
            self.config.animal_regrowth_base
            * SEASON_ANIMAL_MOD[self.season]
        )
        carrying_capacity = TERRAIN_ANIMAL_BASE[self.terrain]
        updated = (
            neighborhood * avoid_humans
            + regrowth * carrying_capacity * (1.0 - neighborhood)
        )
        self.animal_density = np.clip(updated, 0.0, 1.0).astype(np.float32)

    def _build_observation(self) -> np.ndarray:
        half = self.config.obs_range // 2
        channels = [
            self.terrain.astype(np.float32) / float(max(Terrain)),
            self.plant_food,
            self.animal_density,
            self.water,
            self.danger,
            self._camp_layer(),
            self.depletion,
            self._agent_layer(),
        ]
        padded_channels = []
        y, x = self.agent_pos
        for layer in channels:
            padded = np.pad(layer, half, mode="constant", constant_values=0.0)
            py = y + half
            px = x + half
            padded_channels.append(
                padded[
                    py - half:py + half + 1,
                    px - half:px + half + 1,
                ]
            )

        energy_plane = np.full(
            (self.config.obs_range, self.config.obs_range),
            np.clip(self.energy / self.config.max_energy, 0.0, 1.0),
            dtype=np.float32,
        )
        season_plane = np.full(
            (self.config.obs_range, self.config.obs_range),
            float(self.season) / float(max(Season)),
            dtype=np.float32,
        )
        padded_channels.extend([energy_plane, season_plane])
        return np.stack(padded_channels, axis=0).astype(np.float32)

    def _camp_layer(self) -> np.ndarray:
        layer = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for y, x in self.band_camp_positions:
            layer[int(y), int(x)] = 1.0
        return layer

    def _agent_layer(self) -> np.ndarray:
        layer = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for band_id in range(self.config.num_bands):
            value = 0.75 if band_id == 0 else 0.5
            for member_id in range(self.config.members_per_band):
                y, x = self._member_position(band_id, member_id)
                layer[y, x] = value
        y, x = self._member_position(0, 0)
        layer[y, x] = 1.0
        return layer

    def _calculate_reward(self, terminated: bool) -> float:
        reward = 0.05 * self._last_food_gained
        reward -= 0.01 * self._last_energy_spent
        reward -= 0.10 * self._last_danger_damage
        reward += 0.01
        if self._last_action in {
            Action.MOVE_CAMP_NORTH,
            Action.MOVE_CAMP_SOUTH,
            Action.MOVE_CAMP_WEST,
            Action.MOVE_CAMP_EAST,
        }:
            reward -= 0.05
        if terminated:
            reward -= 1.0
        return float(reward)

    def _camp_pressure(self) -> float:
        food_shortage = 1.0 - np.clip(
            self.energy / self.config.initial_energy,
            0.0,
            1.0,
        )
        local_depletion = float(np.mean(self.depletion))
        danger_pressure = float(np.mean(self.danger))
        water_distance = self._distance_to_nearest_water()[self.camp_pos]
        water_pressure = np.clip(water_distance / self.grid_size, 0, 1)
        winter_pressure = 1.0 if self.season == Season.WINTER else 0.0
        return float(
            0.35 * food_shortage
            + 0.25 * local_depletion
            + 0.20 * danger_pressure
            + 0.15 * water_pressure
            + 0.05 * winter_pressure
        )

    def _build_info(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "day": self.day,
            "season": Season(self.season).name.lower(),
            "camp_id": self.camp_id,
            "camp_age": self.camp_age,
            "camp_pressure": self._last_camp_pressure,
            "macro_x": self.macro_x,
            "macro_y": self.macro_y,
            "agent_pos": self.agent_pos,
            "camp_pos": self.camp_pos,
            "band_camp_positions": self.band_camp_positions.tolist(),
            "member_positions": self.member_positions.tolist(),
            "num_bands": self.config.num_bands,
            "members_per_band": self.config.members_per_band,
            "energy": self.energy,
            "stored_food": self.stored_food,
            "population": self.population,
            "local_depletion_level": self.local_depletion_level,
            "mean_plant_food": float(np.mean(self.plant_food)),
            "mean_animal_density": float(np.mean(self.animal_density)),
            "mean_depletion": float(np.mean(self.depletion)),
            "num_camp_moves": self.num_camp_moves,
            "macro_distance_traveled": self.macro_distance_traveled,
            "starvation_events": self.starvation_events,
            "successful_hunts": self.successful_hunts,
            "gathering_events": self.gathering_events,
            "last_food_gained": self._last_food_gained,
            "last_energy_spent": self._last_energy_spent,
            "last_danger_damage": self._last_danger_damage,
        }

    def _distance_from_camp(self, pos: tuple[int, int]) -> float:
        y, x = pos
        cy, cx = self.camp_pos
        return float(abs(y - cy) + abs(x - cx))

    def _distance_to_nearest_water(self) -> np.ndarray:
        if hasattr(self, "water"):
            water_positions = np.argwhere(self.water > 0.0)
        else:
            water_positions = np.empty((0, 2))
        yy, xx = np.mgrid[0:self.grid_size, 0:self.grid_size]
        if len(water_positions) == 0:
            center = np.array([[self.grid_size // 2, self.grid_size // 2]])
            water_positions = center
        distances = np.full(
            (self.grid_size, self.grid_size),
            np.inf,
            dtype=np.float32,
        )
        for wy, wx in water_positions:
            distances = np.minimum(
                distances,
                np.abs(yy - wy) + np.abs(xx - wx),
            )
        return distances.astype(np.float32)

    @staticmethod
    def _patch_seed(global_seed: int, macro_x: int, macro_y: int) -> int:
        value = (
            int(global_seed) * 0x9E3779B1
            ^ int(macro_x) * 0x85EBCA77
            ^ int(macro_y) * 0xC2B2AE3D
        )
        value ^= value >> 16
        return int(value & 0xFFFFFFFF)
