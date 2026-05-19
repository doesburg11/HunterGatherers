from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import IntEnum
from pathlib import Path
import tomllib
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


class MemberSex(IntEnum):
    FEMALE = 0
    MALE = 1


@dataclass(frozen=True)
class MemberState:
    agent_id: str
    band_id: int
    member_slot: int
    member_id: int
    alive: bool
    age: int
    sex: MemberSex
    energy: float
    position: tuple[int, int]


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


DEFAULT_ENV_CONFIG_PATH = Path(__file__).with_name("patch_env_config.toml")
_RAW_ENV_CONFIG = tomllib.loads(
    DEFAULT_ENV_CONFIG_PATH.read_text(encoding="utf-8")
)
_ENVIRONMENT_DEFAULTS = _RAW_ENV_CONFIG["environment"]


def _default_config_value(name: str, fallback: Any) -> Any:
    return _ENVIRONMENT_DEFAULTS.get(name, fallback)


def _config_array(section: str, name: str) -> np.ndarray:
    return np.array(_RAW_ENV_CONFIG[section][name], dtype=np.float32)


@dataclass(frozen=True)
class PatchEnvConfig:
    grid_size: int = _default_config_value("grid_size", 31)
    obs_range: int = _default_config_value("obs_range", 11)
    max_steps: int = _default_config_value("max_steps", 2000)
    num_bands: int = _default_config_value("num_bands", 2)
    members_per_band: int = _default_config_value("members_per_band", 15)
    initial_energy: float = _default_config_value("initial_energy", 100.0)
    max_energy: float = _default_config_value("max_energy", 150.0)
    movement_cost: float = _default_config_value("movement_cost", 1.0)
    rest_recovery: float = _default_config_value("rest_recovery", 0.5)
    plant_food_gain: float = _default_config_value("plant_food_gain", 8.0)
    animal_food_gain: float = _default_config_value("animal_food_gain", 20.0)
    gather_amount: float = _default_config_value("gather_amount", 0.25)
    hunt_amount: float = _default_config_value("hunt_amount", 0.2)
    plant_regrowth_base: float = _default_config_value(
        "plant_regrowth_base",
        0.01,
    )
    animal_regrowth_base: float = _default_config_value(
        "animal_regrowth_base",
        0.005,
    )
    season_length: int = _default_config_value("season_length", 250)
    camp_move_cost: float = _default_config_value("camp_move_cost", 20.0)
    camp_move_distance: int = _default_config_value("camp_move_distance", 10)
    danger_damage_scale: float = _default_config_value(
        "danger_damage_scale",
        2.0,
    )
    trail_memory_decay: float = _default_config_value(
        "trail_memory_decay",
        0.995,
    )
    depletion_decay: float = _default_config_value("depletion_decay", 0.0005)
    global_seed: int = _default_config_value("global_seed", 12345)

    @classmethod
    def from_toml(cls, path: str | Path) -> "PatchEnvConfig":
        raw_config = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        values = raw_config.get("environment", raw_config)
        field_names = {field.name for field in fields(cls)}
        unknown_fields = set(values) - field_names
        if unknown_fields:
            unknown = ", ".join(sorted(unknown_fields))
            raise ValueError(f"Unknown environment config fields: {unknown}")
        return cls(**values)


TERRAIN_PLANT_BASE = _config_array("terrain", "plant_base")
TERRAIN_ANIMAL_BASE = _config_array("terrain", "animal_base")
TERRAIN_MOVEMENT_COST = _config_array("terrain", "movement_cost")
TERRAIN_DANGER_BASE = _config_array("terrain", "danger_base")

SEASON_PLANT_MOD = _config_array("season", "plant_mod")
SEASON_ANIMAL_MOD = _config_array("season", "animal_mod")
SEASON_COST_MOD = _config_array("season", "cost_mod")
SEASON_RISK_MOD = _config_array("season", "risk_mod")


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
        self.member_ids: np.ndarray = np.empty(
            (self.config.num_bands, self.config.members_per_band),
            dtype=np.int32,
        )
        self.next_member_ids: np.ndarray = np.empty(
            self.config.num_bands,
            dtype=np.int32,
        )
        self.member_alive: np.ndarray = np.empty(
            (self.config.num_bands, self.config.members_per_band),
            dtype=bool,
        )
        self.member_age: np.ndarray = np.empty(
            (self.config.num_bands, self.config.members_per_band),
            dtype=np.int32,
        )
        self.member_sex: np.ndarray = np.empty(
            (self.config.num_bands, self.config.members_per_band),
            dtype=np.int8,
        )
        self.member_energy: np.ndarray = np.empty(
            (self.config.num_bands, self.config.members_per_band),
            dtype=np.float32,
        )
        self.member_last_food_gained: np.ndarray = np.empty(
            (self.config.num_bands, self.config.members_per_band),
            dtype=np.float32,
        )
        self.member_last_energy_spent: np.ndarray = np.empty(
            (self.config.num_bands, self.config.members_per_band),
            dtype=np.float32,
        )
        self.member_last_danger_damage: np.ndarray = np.empty(
            (self.config.num_bands, self.config.members_per_band),
            dtype=np.float32,
        )
        self.member_last_action: np.ndarray = np.empty(
            (self.config.num_bands, self.config.members_per_band),
            dtype=np.int8,
        )
        self.reset()

    @property
    def energy(self) -> float:
        """Backward-compatible energy for the controlled member."""
        return self._member_energy(0, 0)

    @energy.setter
    def energy(self, value: float) -> None:
        self._set_member_energy(0, 0, float(value))

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
        initial_energy = float(
            options.get("initial_energy", self.config.initial_energy)
        )
        self._initialize_member_lifecycle(initial_energy)
        self.stored_food = float(options.get("stored_food", 0.0))
        self.camp_age = 0
        self.local_depletion_level = 0.0
        self.num_camp_moves = 0
        self.successful_hunts = 0
        self.gathering_events = 0
        self.starvation_events = 0
        self.macro_distance_traveled = 0

        self._generate_new_patch()
        self._place_bands()
        self._clear_member_step_stats()
        self._last_camp_pressure = self._camp_pressure()
        return self._build_observation(), self._build_info()

    def _initialize_member_lifecycle(self, initial_energy: float) -> None:
        member_ids = np.arange(
            self.config.members_per_band,
            dtype=np.int32,
        )
        self.member_ids = np.tile(member_ids, (self.config.num_bands, 1))
        self.next_member_ids = np.full(
            self.config.num_bands,
            self.config.members_per_band,
            dtype=np.int32,
        )
        self.member_alive = np.ones(
            (self.config.num_bands, self.config.members_per_band),
            dtype=bool,
        )
        self.member_age = np.zeros(
            (self.config.num_bands, self.config.members_per_band),
            dtype=np.int32,
        )
        band_ids, member_slots = np.indices(
            (self.config.num_bands, self.config.members_per_band),
        )
        self.member_sex = ((band_ids + member_slots) % 2).astype(np.int8)
        self.member_energy = np.full(
            (self.config.num_bands, self.config.members_per_band),
            initial_energy,
            dtype=np.float32,
        )
        self._sync_population()
        self._clear_member_step_stats()

    def _clear_member_step_stats(self) -> None:
        shape = (self.config.num_bands, self.config.members_per_band)
        self.member_last_food_gained = np.zeros(shape, dtype=np.float32)
        self.member_last_energy_spent = np.zeros(shape, dtype=np.float32)
        self.member_last_danger_damage = np.zeros(shape, dtype=np.float32)
        self.member_last_action = np.full(
            shape,
            int(Action.STAY),
            dtype=np.int8,
        )
        self._last_food_gained = 0.0
        self._last_energy_spent = 0.0
        self._last_danger_damage = 0.0
        self._last_action = Action.STAY

    def _sync_population(self) -> None:
        self.population = int(np.count_nonzero(self.member_alive))

    def _advance_member_lifecycle(self) -> None:
        self.member_age[self.member_alive] += 1

    def _kill_member(self, band_id: int, member_id: int) -> None:
        if not self.member_alive[band_id, member_id]:
            return
        self.member_alive[band_id, member_id] = False
        self.member_energy[band_id, member_id] = 0.0
        self._sync_population()

    def _update_member_survival(self) -> int:
        dying_members = np.argwhere(
            self.member_alive & (self.member_energy <= 0.0)
        )
        for band_id, member_id in dying_members:
            self._kill_member(int(band_id), int(member_id))
        return int(len(dying_members))

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

        self._clear_member_step_stats()
        self._last_action = action
        self.member_last_action[0, 0] = int(action)

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

        self._advance_member_lifecycle()
        starvation_events = self._update_member_survival()
        self.starvation_events += starvation_events
        self._last_camp_pressure = self._camp_pressure()
        terminated = not bool(self.member_alive[0, 0])
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
                if not self.member_alive[band_id, member_id]:
                    continue
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
            alive_slots = [
                member_id
                for member_id in range(self.config.members_per_band)
                if self.member_alive[band_id, member_id]
            ]
            members = self._member_cells_near(
                camp,
                occupied,
                needed=len(alive_slots),
            )
            for member_id, member_pos in zip(alive_slots, members):
                self.member_positions[band_id, member_id] = member_pos
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
        needed: int | None = None,
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

        if needed is None:
            needed = self.config.members_per_band
        if needed == 0:
            return np.empty((0, 2), dtype=np.int16)
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
        if band_id == 0 and member_id == 0:
            self.agent_pos = (int(y), int(x))

    def _agent_id(self, band_id: int, member_id: int) -> str:
        member_identity = int(self.member_ids[band_id, member_id])
        return f"band_{band_id}_member_{member_identity}"

    def _member_energy(self, band_id: int, member_id: int) -> float:
        return float(self.member_energy[band_id, member_id])

    def _set_member_energy(
        self,
        band_id: int,
        member_id: int,
        value: float,
    ) -> None:
        self.member_energy[band_id, member_id] = float(value)

    def _add_member_energy(
        self,
        band_id: int,
        member_id: int,
        amount: float,
    ) -> None:
        energy = self._member_energy(band_id, member_id) + amount
        self._set_member_energy(
            band_id,
            member_id,
            min(self.config.max_energy, energy),
        )

    def member_state(self, band_id: int, member_id: int) -> MemberState:
        return MemberState(
            agent_id=self._agent_id(band_id, member_id),
            band_id=band_id,
            member_slot=member_id,
            member_id=int(self.member_ids[band_id, member_id]),
            alive=bool(self.member_alive[band_id, member_id]),
            age=int(self.member_age[band_id, member_id]),
            sex=MemberSex(int(self.member_sex[band_id, member_id])),
            energy=self._member_energy(band_id, member_id),
            position=self._member_position(band_id, member_id),
        )

    def active_agent_ids(self, band_id: int | None = None) -> list[str]:
        agent_ids: list[str] = []
        band_ids = (
            range(self.config.num_bands)
            if band_id is None
            else range(band_id, band_id + 1)
        )
        for current_band_id in band_ids:
            for member_id in range(self.config.members_per_band):
                if self.member_alive[current_band_id, member_id]:
                    agent_ids.append(self._agent_id(current_band_id, member_id))
        return agent_ids

    def _cell_has_member(
        self,
        pos: tuple[int, int],
        ignore: tuple[int, int] | None = None,
    ) -> bool:
        y, x = pos
        for band_id in range(self.config.num_bands):
            for member_id in range(self.config.members_per_band):
                if not self.member_alive[band_id, member_id]:
                    continue
                if ignore == (band_id, member_id):
                    continue
                member_y, member_x = self._member_position(band_id, member_id)
                if (member_y, member_x) == (y, x):
                    return True
        return False

    def _generate_terrain(self) -> np.ndarray:
        size = self.grid_size
        yy, xx = np.mgrid[0:size, 0:size]

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

        terrain = np.full(
            (size, size),
            Terrain.GRASSLAND,
            dtype=np.int8,
        )
        terrain[water_mask] = Terrain.WATER

        return terrain

    def _generate_plants(self) -> np.ndarray:
        base = self._patch_rng.random(
            (self.grid_size, self.grid_size),
            dtype=np.float32,
        )
        patches = self._smooth_noise(self.grid_size, passes=3)
        sparse_patches = np.clip((patches - 0.56) / 0.30, 0.0, 1.0)
        scattered_plants = (base > 0.78).astype(np.float32) * (
            0.45 + 0.55 * base
        )
        plant_capacity = np.maximum(sparse_patches, scattered_plants)
        plant_capacity[plant_capacity < 0.18] = 0.0
        plant_capacity[self.water > 0.0] = 0.0
        self.plant_capacity = plant_capacity.astype(np.float32)
        plants = np.minimum(
            self.plant_capacity,
            self.plant_capacity * SEASON_PLANT_MOD[self.season],
        )
        plants[self.water > 0.0] = 0.0
        return np.clip(plants, 0.0, 1.0).astype(np.float32)

    def _generate_animals(self) -> np.ndarray:
        return np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

    def _generate_danger(self) -> np.ndarray:
        return np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

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
        self._move_member(0, 0, action)

    def _move_member(
        self,
        band_id: int,
        member_id: int,
        action: Action,
    ) -> None:
        if not self.member_alive[band_id, member_id]:
            return
        dy, dx = {
            Action.MOVE_NORTH: (-1, 0),
            Action.MOVE_SOUTH: (1, 0),
            Action.MOVE_WEST: (0, -1),
            Action.MOVE_EAST: (0, 1),
        }[action]
        y, x = self._member_position(band_id, member_id)
        ny = int(np.clip(y + dy, 0, self.grid_size - 1))
        nx = int(np.clip(x + dx, 0, self.grid_size - 1))
        if self._cell_has_member((ny, nx), ignore=(band_id, member_id)):
            self._spend_member_energy(band_id, member_id, 0.15)
            return
        terrain = Terrain(int(self.terrain[ny, nx]))
        move_cost = float(
            TERRAIN_MOVEMENT_COST[terrain] * SEASON_COST_MOD[self.season]
        )
        if terrain == Terrain.WATER:
            move_cost *= 1.5
        distance_cost = 0.015 * self._distance_from_camp((ny, nx))
        self._set_member_position(band_id, member_id, (ny, nx))
        self._spend_member_energy(
            band_id,
            member_id,
            self.config.movement_cost * move_cost + distance_cost
        )

    def _gather(self) -> None:
        self._gather_member(0, 0)

    def _gather_member(self, band_id: int, member_id: int) -> None:
        if not self.member_alive[band_id, member_id]:
            return
        y, x = self._member_position(band_id, member_id)
        available = float(self.plant_food[y, x])
        gathered = min(available, self.config.gather_amount)
        self.plant_food[y, x] = max(0.0, available - gathered)
        food_gained = gathered * self.config.plant_food_gain
        self._record_food_gained(band_id, member_id, food_gained)
        self._add_member_energy(band_id, member_id, food_gained)
        self.gathering_events += int(gathered > 0.0)
        self._spend_member_energy(band_id, member_id, 0.35)

    def _hunt(self) -> None:
        self._hunt_member(0, 0)

    def _hunt_member(self, band_id: int, member_id: int) -> None:
        if not self.member_alive[band_id, member_id]:
            return
        self._spend_member_energy(band_id, member_id, 0.15)

    def _rest(self) -> None:
        self._rest_member(0, 0)

    def _rest_member(self, band_id: int, member_id: int) -> None:
        if not self.member_alive[band_id, member_id]:
            return
        y, x = self._member_position(band_id, member_id)
        camp_bonus = 0.5 if (y, x) == self.camp_pos else 0.0
        self._add_member_energy(
            band_id,
            member_id,
            self.config.rest_recovery + camp_bonus,
        )
        self._spend_member_energy(band_id, member_id, 0.1)

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
        self._place_bands()
        self.local_depletion_level = float(np.mean(self.depletion))

    def _spend_energy(self, amount: float) -> None:
        self._spend_member_energy(0, 0, amount)

    def _spend_member_energy(
        self,
        band_id: int,
        member_id: int,
        amount: float,
    ) -> None:
        amount = max(0.0, float(amount))
        self.member_energy[band_id, member_id] -= amount
        self.member_last_energy_spent[band_id, member_id] += amount
        if band_id == 0 and member_id == 0:
            self._last_energy_spent += amount

    def _record_food_gained(
        self,
        band_id: int,
        member_id: int,
        amount: float,
    ) -> None:
        amount = max(0.0, float(amount))
        self.member_last_food_gained[band_id, member_id] += amount
        if band_id == 0 and member_id == 0:
            self._last_food_gained += amount

    def _apply_danger(self) -> None:
        self._apply_member_danger(0, 0)

    def _apply_member_danger(self, band_id: int, member_id: int) -> None:
        if not self.member_alive[band_id, member_id]:
            return

    def _update_resources(self) -> None:
        season_mod = float(SEASON_PLANT_MOD[self.season])
        plant_capacity = getattr(
            self,
            "plant_capacity",
            np.where(self.plant_food > 0.0, 1.0, 0.0).astype(np.float32),
        )
        effective_regrowth = (
            self.config.plant_regrowth_base
            * season_mod
            * (plant_capacity > 0.0)
        )
        self.plant_food = np.clip(
            self.plant_food + effective_regrowth,
            0.0,
            plant_capacity,
        ).astype(np.float32)
        self.plant_food[self.water > 0.0] = 0.0

    def _update_animals(self) -> None:
        self.animal_density.fill(0.0)

    def _build_observation(self) -> np.ndarray:
        return self._build_member_observation(0, 0)

    def _build_member_observation(
        self,
        band_id: int,
        member_id: int,
    ) -> np.ndarray:
        half = self.config.obs_range // 2
        channels = [
            (self.terrain == Terrain.WATER).astype(np.float32),
            self.plant_food,
            self.animal_density,
            self.water,
            self.danger,
            self._camp_layer(),
            self.depletion,
            self._agent_layer((band_id, member_id)),
        ]
        padded_channels = []
        y, x = self._member_position(band_id, member_id)
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
            np.clip(
                self._member_energy(band_id, member_id)
                / self.config.max_energy,
                0.0,
                1.0,
            ),
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

    def _agent_layer(
        self,
        focal_member: tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        layer = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for band_id in range(self.config.num_bands):
            value = 0.75 if band_id == 0 else 0.5
            for member_id in range(self.config.members_per_band):
                if not self.member_alive[band_id, member_id]:
                    continue
                y, x = self._member_position(band_id, member_id)
                layer[y, x] = value
        band_id, member_id = focal_member
        if self.member_alive[band_id, member_id]:
            y, x = self._member_position(band_id, member_id)
            layer[y, x] = 1.0
        return layer

    def _calculate_reward(self, terminated: bool) -> float:
        return self._calculate_member_reward(0, 0, terminated)

    def _calculate_member_reward(
        self,
        band_id: int,
        member_id: int,
        terminated: bool,
    ) -> float:
        reward = 0.05 * float(
            self.member_last_food_gained[band_id, member_id]
        )
        reward -= 0.01 * float(
            self.member_last_energy_spent[band_id, member_id]
        )
        reward -= 0.10 * float(
            self.member_last_danger_damage[band_id, member_id]
        )
        reward += 0.01
        last_action = Action(int(self.member_last_action[band_id, member_id]))
        if last_action in {
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
        water_distance = self._distance_to_nearest_water()[self.camp_pos]
        water_pressure = np.clip(water_distance / self.grid_size, 0, 1)
        winter_pressure = 1.0 if self.season == Season.WINTER else 0.0
        return float(
            0.65 * food_shortage
            + 0.25 * water_pressure
            + 0.10 * winter_pressure
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
            "member_ids": self.member_ids.tolist(),
            "member_alive": self.member_alive.tolist(),
            "member_age": self.member_age.tolist(),
            "member_sex": [
                [
                    MemberSex(int(value)).name.lower()
                    for value in band_sexes
                ]
                for band_sexes in self.member_sex
            ],
            "member_energy": self.member_energy.tolist(),
            "active_agent_ids": self.active_agent_ids(),
            "controlled_agent_id": self._agent_id(0, 0),
            "num_bands": self.config.num_bands,
            "members_per_band": self.config.members_per_band,
            "energy": self.energy,
            "stored_food": self.stored_food,
            "population": self.population,
            "max_population": (
                self.config.num_bands * self.config.members_per_band
            ),
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
            "member_last_food_gained": (
                self.member_last_food_gained.tolist()
            ),
            "member_last_energy_spent": (
                self.member_last_energy_spent.tolist()
            ),
            "member_last_danger_damage": (
                self.member_last_danger_damage.tolist()
            ),
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


class BandMemberPatchEnv(HunterGathererPatchEnv):
    """Ray-free multi-agent env for controlling members from one band."""

    def __init__(
        self,
        config: PatchEnvConfig | dict[str, Any] | None = None,
        *,
        controlled_band_id: int = 0,
    ):
        self.controlled_band_id = int(controlled_band_id)
        self.possible_agents: list[str] = []
        self.agents: list[str] = []
        self._agent_to_member: dict[str, tuple[int, int]] = {}
        super().__init__(config)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        super().reset(seed=seed, options=options)
        self._refresh_agent_ids()
        return self._multi_agent_observations(), self._multi_agent_infos(
            self.agents
        )

    def step(
        self,
        action_dict: dict[str, int],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        self._validate_action_agents(action_dict)
        active_slots = self._active_controlled_slots()
        agent_ids = [
            self._agent_id(self.controlled_band_id, slot)
            for slot in active_slots
        ]

        self._begin_multi_agent_step(active_slots)
        actions = self._member_actions(action_dict, active_slots)
        if self._leader_moves_camp(actions):
            self._apply_leader_camp_move(actions[0])
        else:
            self._apply_member_moves(actions)
            self._apply_member_non_move_actions(actions)
            self._apply_controlled_band_danger(active_slots)
            self._update_resources()
            self._update_animals()
            self.local_depletion_level = float(np.mean(self.depletion))

        self._advance_member_lifecycle()
        self.starvation_events += self._update_member_survival()
        self._last_camp_pressure = self._camp_pressure()

        truncated = self.step_count >= self.config.max_steps
        self._refresh_agent_ids()
        all_terminated = len(self.agents) == 0

        rewards = {
            agent_id: self._calculate_member_reward(
                *self._member_for_agent(agent_id),
                terminated=not self._agent_alive(agent_id),
            )
            for agent_id in agent_ids
        }
        terminations = {
            agent_id: not self._agent_alive(agent_id)
            for agent_id in agent_ids
        }
        terminations["__all__"] = all_terminated
        truncations = {agent_id: truncated for agent_id in agent_ids}
        truncations["__all__"] = truncated
        infos = self._multi_agent_infos(agent_ids)

        if all_terminated or truncated:
            self.agents = []
            observations: dict[str, np.ndarray] = {}
        else:
            observations = self._multi_agent_observations()

        return observations, rewards, terminations, truncations, infos

    def _refresh_agent_ids(self) -> None:
        if self.controlled_band_id < 0:
            raise ValueError("controlled_band_id must be non-negative")
        if self.controlled_band_id >= self.config.num_bands:
            raise ValueError("controlled_band_id must be less than num_bands")

        self.possible_agents = [
            self._agent_id(self.controlled_band_id, member_id)
            for member_id in range(self.config.members_per_band)
        ]
        self._agent_to_member = {
            agent_id: (self.controlled_band_id, member_id)
            for member_id, agent_id in enumerate(self.possible_agents)
        }
        self.action_spaces = {
            agent_id: self.action_space for agent_id in self.possible_agents
        }
        self.observation_spaces = {
            agent_id: self.observation_space
            for agent_id in self.possible_agents
        }
        self.agents = self.active_agent_ids(self.controlled_band_id)

    def _active_controlled_slots(self) -> list[int]:
        return [
            member_id
            for member_id in range(self.config.members_per_band)
            if self.member_alive[self.controlled_band_id, member_id]
        ]

    def _validate_action_agents(self, action_dict: dict[str, int]) -> None:
        unknown_agents = sorted(set(action_dict) - set(self._agent_to_member))
        if unknown_agents:
            unknown = ", ".join(unknown_agents)
            raise ValueError(f"Unknown agent ids: {unknown}")

    def _member_actions(
        self,
        action_dict: dict[str, int],
        active_slots: list[int],
    ) -> dict[int, Action]:
        parsed_actions = {
            agent_id: Action(int(action))
            for agent_id, action in action_dict.items()
        }
        actions: dict[int, Action] = {}
        for member_id in active_slots:
            agent_id = self._agent_id(self.controlled_band_id, member_id)
            action = parsed_actions.get(agent_id, Action.STAY)
            if member_id != 0 and self._is_camp_move_action(action):
                action = Action.STAY
            actions[member_id] = action
            self.member_last_action[
                self.controlled_band_id,
                member_id,
            ] = int(action)
        self._last_action = actions.get(0, Action.STAY)
        return actions

    def _begin_multi_agent_step(self, active_slots: list[int]) -> None:
        self.step_count += 1
        self.day = self.step_count
        self.season = Season(
            (self.step_count // self.config.season_length) % len(Season)
        )
        self.camp_age += 1
        self._clear_member_step_stats()

        self.trail_memory *= self.config.trail_memory_decay
        for member_id in active_slots:
            y, x = self._member_position(self.controlled_band_id, member_id)
            self.trail_memory[y, x] = min(
                1.0,
                self.trail_memory[y, x] + 0.08,
            )

    def _leader_moves_camp(self, actions: dict[int, Action]) -> bool:
        return (
            self.controlled_band_id == 0
            and self._is_camp_move_action(actions.get(0, Action.STAY))
        )

    def _apply_leader_camp_move(self, action: Action) -> None:
        self.member_last_action[self.controlled_band_id, 0] = int(action)
        self._last_action = action
        self._move_camp(action)

    @staticmethod
    def _is_movement_action(action: Action) -> bool:
        return action in {
            Action.MOVE_NORTH,
            Action.MOVE_SOUTH,
            Action.MOVE_WEST,
            Action.MOVE_EAST,
        }

    @staticmethod
    def _is_camp_move_action(action: Action) -> bool:
        return action in {
            Action.MOVE_CAMP_NORTH,
            Action.MOVE_CAMP_SOUTH,
            Action.MOVE_CAMP_WEST,
            Action.MOVE_CAMP_EAST,
        }

    def _apply_member_moves(self, actions: dict[int, Action]) -> None:
        movement_actions = {
            member_id: action
            for member_id, action in actions.items()
            if self._is_movement_action(action)
        }
        proposals = {
            member_id: self._movement_target(member_id, action)
            for member_id, action in movement_actions.items()
        }
        target_counts: dict[tuple[int, int], int] = {}
        for target in proposals.values():
            target_counts[target] = target_counts.get(target, 0) + 1

        occupied = self._occupied_member_positions()
        for member_id, target in proposals.items():
            occupied_by = occupied.get(target)
            blocked = (
                target_counts[target] > 1
                or (
                    occupied_by is not None
                    and occupied_by != (self.controlled_band_id, member_id)
                )
            )
            if blocked:
                self._spend_member_energy(
                    self.controlled_band_id,
                    member_id,
                    0.15,
                )
                continue

            self._move_member_to_target(member_id, target)

    def _movement_target(
        self,
        member_id: int,
        action: Action,
    ) -> tuple[int, int]:
        dy, dx = {
            Action.MOVE_NORTH: (-1, 0),
            Action.MOVE_SOUTH: (1, 0),
            Action.MOVE_WEST: (0, -1),
            Action.MOVE_EAST: (0, 1),
        }[action]
        y, x = self._member_position(self.controlled_band_id, member_id)
        return (
            int(np.clip(y + dy, 0, self.grid_size - 1)),
            int(np.clip(x + dx, 0, self.grid_size - 1)),
        )

    def _occupied_member_positions(self) -> dict[tuple[int, int], tuple[int, int]]:
        occupied: dict[tuple[int, int], tuple[int, int]] = {}
        for band_id in range(self.config.num_bands):
            for member_id in range(self.config.members_per_band):
                if not self.member_alive[band_id, member_id]:
                    continue
                occupied[self._member_position(band_id, member_id)] = (
                    band_id,
                    member_id,
                )
        return occupied

    def _move_member_to_target(
        self,
        member_id: int,
        target: tuple[int, int],
    ) -> None:
        terrain = Terrain(int(self.terrain[target]))
        move_cost = float(
            TERRAIN_MOVEMENT_COST[terrain] * SEASON_COST_MOD[self.season]
        )
        if terrain == Terrain.WATER:
            move_cost *= 1.5
        distance_cost = 0.015 * self._distance_from_camp(target)
        self._set_member_position(self.controlled_band_id, member_id, target)
        self._spend_member_energy(
            self.controlled_band_id,
            member_id,
            self.config.movement_cost * move_cost + distance_cost,
        )

    def _apply_member_non_move_actions(
        self,
        actions: dict[int, Action],
    ) -> None:
        for member_id, action in actions.items():
            if self._is_movement_action(action):
                continue
            if action == Action.GATHER:
                self._gather_member(self.controlled_band_id, member_id)
            elif action == Action.HUNT:
                self._hunt_member(self.controlled_band_id, member_id)
            elif action == Action.REST:
                self._rest_member(self.controlled_band_id, member_id)
            else:
                self._spend_member_energy(
                    self.controlled_band_id,
                    member_id,
                    0.15,
                )

    def _apply_controlled_band_danger(self, active_slots: list[int]) -> None:
        for member_id in active_slots:
            self._apply_member_danger(self.controlled_band_id, member_id)

    def _agent_alive(self, agent_id: str) -> bool:
        band_id, member_id = self._member_for_agent(agent_id)
        return bool(self.member_alive[band_id, member_id])

    def _member_for_agent(self, agent_id: str) -> tuple[int, int]:
        return self._agent_to_member[agent_id]

    def _multi_agent_observations(self) -> dict[str, np.ndarray]:
        return {
            agent_id: self._build_member_observation(
                *self._member_for_agent(agent_id)
            )
            for agent_id in self.agents
        }

    def _multi_agent_infos(
        self,
        agent_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        return {
            agent_id: self._build_agent_info(agent_id)
            for agent_id in agent_ids
        }

    def _build_agent_info(self, agent_id: str) -> dict[str, Any]:
        band_id, member_id = self._member_for_agent(agent_id)
        state = self.member_state(band_id, member_id)
        return {
            "agent_id": state.agent_id,
            "band_id": state.band_id,
            "member_slot": state.member_slot,
            "member_id": state.member_id,
            "alive": state.alive,
            "age": state.age,
            "sex": state.sex.name.lower(),
            "energy": state.energy,
            "position": state.position,
            "step_count": self.step_count,
            "season": Season(self.season).name.lower(),
            "camp_id": self.camp_id,
            "camp_pressure": self._last_camp_pressure,
            "last_food_gained": float(
                self.member_last_food_gained[band_id, member_id]
            ),
            "last_energy_spent": float(
                self.member_last_energy_spent[band_id, member_id]
            ),
            "last_danger_damage": float(
                self.member_last_danger_damage[band_id, member_id]
            ),
        }
