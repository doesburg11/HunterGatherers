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
    hydration: float
    position: tuple[int, int]


class Action(IntEnum):
    STAY = 0
    MOVE_NORTH = 1
    MOVE_SOUTH = 2
    MOVE_WEST = 3
    MOVE_EAST = 4


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
    initial_hydration: float = _default_config_value(
        "initial_hydration",
        100.0,
    )
    max_hydration: float = _default_config_value("max_hydration", 100.0)
    thirst_per_step: float = _default_config_value("thirst_per_step", 1.0)
    drink_amount: float = _default_config_value("drink_amount", 25.0)
    movement_cost: float = _default_config_value("movement_cost", 1.0)
    grass_movement_cost: float = _default_config_value(
        "grass_movement_cost",
        1.0,
    )
    water_movement_cost: float = _default_config_value(
        "water_movement_cost",
        4.5,
    )
    movement_energy_rate: float = _default_config_value(
        "movement_energy_rate",
        0.01,
    )
    water_load_factor: float = _default_config_value("water_load_factor", 1.0)
    plant_energy_capacity: float = _default_config_value(
        "plant_energy_capacity",
        8.0,
    )
    plant_eat_amount: float = _default_config_value("plant_eat_amount", 2.0)
    food_carry_capacity: float = _default_config_value(
        "food_carry_capacity",
        30.0,
    )
    water_carry_capacity: float = _default_config_value(
        "water_carry_capacity",
        30.0,
    )
    water_collect_amount: float = _default_config_value(
        "water_collect_amount",
        25.0,
    )
    personal_energy_reserve: float = _default_config_value(
        "personal_energy_reserve",
        100.0,
    )
    camp_food_withdraw_threshold: float = _default_config_value(
        "camp_food_withdraw_threshold",
        80.0,
    )
    camp_food_withdraw_amount: float = _default_config_value(
        "camp_food_withdraw_amount",
        5.0,
    )
    camp_water_withdraw_threshold: float = _default_config_value(
        "camp_water_withdraw_threshold",
        80.0,
    )
    camp_water_withdraw_amount: float = _default_config_value(
        "camp_water_withdraw_amount",
        10.0,
    )
    plant_regrowth_base: float = _default_config_value(
        "plant_regrowth_base",
        0.01,
    )
    season_length: int = _default_config_value("season_length", 250)
    camp_depletion_radius: int = _default_config_value(
        "camp_depletion_radius",
        5,
    )
    camp_depletion_threshold: float = _default_config_value(
        "camp_depletion_threshold",
        0.75,
    )
    camp_relocation_cost: float = _default_config_value(
        "camp_relocation_cost",
        0.0,
    )
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


SEASON_PLANT_MOD = _config_array("season", "plant_mod")
SEASON_COST_MOD = _config_array("season", "cost_mod")


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
        if self.config.max_hydration <= 0:
            raise ValueError("max_hydration must be positive")
        if self.config.thirst_per_step < 0:
            raise ValueError("thirst_per_step must be non-negative")
        if self.config.camp_depletion_radius < 0:
            raise ValueError("camp_depletion_radius must be non-negative")
        if not 0.0 <= self.config.camp_depletion_threshold <= 1.0:
            raise ValueError(
                "camp_depletion_threshold must be between 0 and 1"
            )
        if self.config.camp_relocation_cost < 0.0:
            raise ValueError("camp_relocation_cost must be non-negative")
        if self.config.movement_energy_rate < 0.0:
            raise ValueError("movement_energy_rate must be non-negative")
        if self.config.water_load_factor < 0.0:
            raise ValueError("water_load_factor must be non-negative")
        if self.config.food_carry_capacity < 0.0:
            raise ValueError("food_carry_capacity must be non-negative")
        if self.config.water_carry_capacity < 0.0:
            raise ValueError("water_carry_capacity must be non-negative")
        if self.config.water_collect_amount < 0.0:
            raise ValueError("water_collect_amount must be non-negative")
        if self.config.personal_energy_reserve < 0.0:
            raise ValueError("personal_energy_reserve must be non-negative")
        if self.config.camp_food_withdraw_threshold < 0.0:
            raise ValueError(
                "camp_food_withdraw_threshold must be non-negative"
            )
        if self.config.camp_food_withdraw_amount < 0.0:
            raise ValueError("camp_food_withdraw_amount must be non-negative")
        if self.config.camp_water_withdraw_threshold < 0.0:
            raise ValueError(
                "camp_water_withdraw_threshold must be non-negative"
            )
        if self.config.camp_water_withdraw_amount < 0.0:
            raise ValueError("camp_water_withdraw_amount must be non-negative")
        self.grid_size = self.config.grid_size
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(15, self.config.obs_range, self.config.obs_range),
            dtype=np.float32,
        )

        self._episode_rng = np.random.default_rng(self.config.global_seed)
        self._patch_rng = np.random.default_rng(self.config.global_seed)
        self._last_food_gained = 0.0
        self._last_water_gained = 0.0
        self._last_energy_spent = 0.0
        self._last_hydration_spent = 0.0
        self._last_danger_damage = 0.0
        self._last_camp_pressure = 0.0
        self._last_camp_relocated = False
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
        self.member_hydration: np.ndarray = np.empty(
            (self.config.num_bands, self.config.members_per_band),
            dtype=np.float32,
        )
        self.member_carried_food: np.ndarray = np.empty(
            (self.config.num_bands, self.config.members_per_band),
            dtype=np.float32,
        )
        self.member_carried_water: np.ndarray = np.empty(
            (self.config.num_bands, self.config.members_per_band),
            dtype=np.float32,
        )
        self.camp_stored_food: np.ndarray = np.empty(
            self.config.num_bands,
            dtype=np.float32,
        )
        self.camp_stored_water: np.ndarray = np.empty(
            self.config.num_bands,
            dtype=np.float32,
        )
        self.member_last_food_gained: np.ndarray = np.empty(
            (self.config.num_bands, self.config.members_per_band),
            dtype=np.float32,
        )
        self.member_last_water_gained: np.ndarray = np.empty(
            (self.config.num_bands, self.config.members_per_band),
            dtype=np.float32,
        )
        self.member_last_energy_spent: np.ndarray = np.empty(
            (self.config.num_bands, self.config.members_per_band),
            dtype=np.float32,
        )
        self.member_last_hydration_spent: np.ndarray = np.empty(
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

    @property
    def hydration(self) -> float:
        """Backward-compatible hydration shortcut for the controlled member."""
        return self._member_hydration(0, 0)

    @hydration.setter
    def hydration(self, value: float) -> None:
        self._set_member_hydration(0, 0, float(value))

    @property
    def plant_food(self) -> np.ndarray:
        """Backward-compatible alias for plant energy cells."""
        return self.plant_energy

    @plant_food.setter
    def plant_food(self, value: np.ndarray) -> None:
        self.plant_energy = value

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
        initial_hydration = float(
            options.get("initial_hydration", self.config.initial_hydration)
        )
        self._initialize_member_lifecycle(initial_energy, initial_hydration)
        self.camp_stored_food = np.full(
            self.config.num_bands,
            float(options.get("stored_food", 0.0)),
            dtype=np.float32,
        )
        self.camp_stored_water = np.full(
            self.config.num_bands,
            float(options.get("stored_water", 0.0)),
            dtype=np.float32,
        )
        self._sync_stored_food_alias()
        self.camp_age = 0
        self.local_depletion_level = 0.0
        self.num_camp_moves = 0
        self.plant_eating_events = 0
        self.water_drinking_events = 0
        self.starvation_events = 0
        self.dehydration_events = 0
        self.macro_distance_traveled = 0

        self._generate_new_patch()
        self._place_bands()
        self._update_depletion()
        self.local_depletion_level = self._local_camp_depletion(
            self.camp_pos
        )
        self._clear_member_step_stats()
        self._last_camp_pressure = self._camp_pressure()
        return self._build_observation(), self._build_info()

    def _initialize_member_lifecycle(
        self,
        initial_energy: float,
        initial_hydration: float,
    ) -> None:
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
        self.member_hydration = np.full(
            (self.config.num_bands, self.config.members_per_band),
            initial_hydration,
            dtype=np.float32,
        )
        self.member_carried_food = np.zeros(
            (self.config.num_bands, self.config.members_per_band),
            dtype=np.float32,
        )
        self.member_carried_water = np.zeros(
            (self.config.num_bands, self.config.members_per_band),
            dtype=np.float32,
        )
        self._sync_population()
        self._clear_member_step_stats()

    def _clear_member_step_stats(self) -> None:
        shape = (self.config.num_bands, self.config.members_per_band)
        self.member_last_food_gained = np.zeros(shape, dtype=np.float32)
        self.member_last_water_gained = np.zeros(shape, dtype=np.float32)
        self.member_last_energy_spent = np.zeros(shape, dtype=np.float32)
        self.member_last_hydration_spent = np.zeros(shape, dtype=np.float32)
        self.member_last_danger_damage = np.zeros(shape, dtype=np.float32)
        self.member_last_action = np.full(
            shape,
            int(Action.STAY),
            dtype=np.int8,
        )
        self._last_food_gained = 0.0
        self._last_water_gained = 0.0
        self._last_energy_spent = 0.0
        self._last_hydration_spent = 0.0
        self._last_danger_damage = 0.0
        self._last_camp_relocated = False
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
        self.member_hydration[band_id, member_id] = 0.0
        self.member_carried_food[band_id, member_id] = 0.0
        self.member_carried_water[band_id, member_id] = 0.0
        self._sync_population()

    def _update_member_survival(self) -> tuple[int, int]:
        starving = self.member_alive & (self.member_energy <= 0.0)
        dehydrating = self.member_alive & (self.member_hydration <= 0.0)
        dying_members = np.argwhere(starving | dehydrating)
        starvation_count = int(np.count_nonzero(starving))
        dehydration_count = int(np.count_nonzero(dehydrating))
        for band_id, member_id in dying_members:
            self._kill_member(int(band_id), int(member_id))
        return starvation_count, dehydration_count

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

        if action in {
            Action.MOVE_NORTH,
            Action.MOVE_SOUTH,
            Action.MOVE_WEST,
            Action.MOVE_EAST,
        }:
            self._move_agent(action)
        else:
            self._spend_energy(0.15)

        self._exchange_camp_resources_for_alive_members()
        self._apply_danger()
        self._update_resources()
        self._update_animals()
        self._maybe_relocate_depleted_camp()

        self._apply_thirst_to_alive_members()
        self._advance_member_lifecycle()
        starvation_events, dehydration_events = self._update_member_survival()
        self.starvation_events += starvation_events
        self.dehydration_events += dehydration_events
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
        self.plant_energy = self._generate_plants()
        self.animal_density = self._generate_animals()
        self.danger = self._generate_danger()
        self.depletion = np.zeros(
            (self.grid_size, self.grid_size),
            dtype=np.float32,
        )
        cy, cx = self.camp_pos
        if self.terrain[cy, cx] == Terrain.WATER:
            self.terrain[cy, cx] = Terrain.GRASSLAND
            self.water[cy, cx] = 0.0
        self._update_depletion()

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

    def _member_hydration(self, band_id: int, member_id: int) -> float:
        return float(self.member_hydration[band_id, member_id])

    def _set_member_energy(
        self,
        band_id: int,
        member_id: int,
        value: float,
    ) -> None:
        self.member_energy[band_id, member_id] = float(value)

    def _set_member_hydration(
        self,
        band_id: int,
        member_id: int,
        value: float,
    ) -> None:
        self.member_hydration[band_id, member_id] = float(value)

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

    def _add_member_hydration(
        self,
        band_id: int,
        member_id: int,
        amount: float,
    ) -> None:
        hydration = self._member_hydration(band_id, member_id) + amount
        self._set_member_hydration(
            band_id,
            member_id,
            min(self.config.max_hydration, hydration),
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
            hydration=self._member_hydration(band_id, member_id),
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
        plant_mask = np.maximum(sparse_patches, scattered_plants)
        plant_mask[plant_mask < 0.18] = 0.0
        plant_mask[self.water > 0.0] = 0.0
        plant_capacity = plant_mask * self.config.plant_energy_capacity
        self.plant_capacity = plant_capacity.astype(np.float32)
        plants = np.minimum(
            self.plant_capacity,
            self.plant_capacity * SEASON_PLANT_MOD[self.season],
        )
        plants[self.water > 0.0] = 0.0
        return np.clip(plants, 0.0, self.plant_capacity).astype(np.float32)

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
        self._set_member_position(band_id, member_id, (ny, nx))
        self._spend_member_energy(
            band_id,
            member_id,
            self._member_movement_cost(band_id, member_id, terrain, (ny, nx)),
        )
        self._consume_cell_resources(band_id, member_id)

    def _movement_cost_for_terrain(self, terrain: Terrain) -> float:
        if terrain == Terrain.WATER:
            terrain_cost = self.config.water_movement_cost
        else:
            terrain_cost = self.config.grass_movement_cost
        return float(terrain_cost * SEASON_COST_MOD[self.season])

    def _member_movement_cost(
        self,
        band_id: int,
        member_id: int,
        terrain: Terrain,
        target: tuple[int, int],
    ) -> float:
        moved_energy = (
            max(0.0, self._member_energy(band_id, member_id))
            + float(self.member_carried_food[band_id, member_id])
            + (
                self.config.water_load_factor
                * float(self.member_carried_water[band_id, member_id])
            )
        )
        terrain_cost = self._movement_cost_for_terrain(terrain)
        distance_cost = 0.015 * self._distance_from_camp(target)
        return float(
            self.config.movement_cost
            * self.config.movement_energy_rate
            * moved_energy
            * terrain_cost
            + distance_cost
        )

    def _consume_cell_resources(self, band_id: int, member_id: int) -> None:
        if not self.member_alive[band_id, member_id]:
            return
        self._consume_plant_energy(band_id, member_id)
        self._consume_water(band_id, member_id)

    def _consume_plant_energy(self, band_id: int, member_id: int) -> None:
        y, x = self._member_position(band_id, member_id)
        available_energy = float(self.plant_energy[y, x])
        body_need = max(
            0.0,
            min(
                self.config.personal_energy_reserve,
                self.config.max_energy,
            )
            - self._member_energy(band_id, member_id),
        )
        carry_room = max(
            0.0,
            self.config.food_carry_capacity
            - float(self.member_carried_food[band_id, member_id]),
        )
        harvested_energy = min(
            available_energy,
            self.config.plant_eat_amount,
            body_need + carry_room,
        )
        eaten_energy = min(harvested_energy, body_need)
        carried_energy = min(harvested_energy - eaten_energy, carry_room)
        self.plant_energy[y, x] = max(
            0.0,
            available_energy - harvested_energy,
        )
        self._update_depletion()
        self._record_food_gained(band_id, member_id, harvested_energy)
        self._add_member_energy(band_id, member_id, eaten_energy)
        self.member_carried_food[band_id, member_id] += carried_energy
        self.plant_eating_events += int(harvested_energy > 0.0)

    def _consume_water(self, band_id: int, member_id: int) -> None:
        y, x = self._member_position(band_id, member_id)
        water_gained = 0.0
        if self.water[y, x] > 0.0:
            hydration_need = max(
                0.0,
                self.config.max_hydration
                - self._member_hydration(band_id, member_id),
            )
            drunk_water = min(self.config.drink_amount, hydration_need)
            carry_room = max(
                0.0,
                self.config.water_carry_capacity
                - float(self.member_carried_water[band_id, member_id]),
            )
            carried_water = min(self.config.water_collect_amount, carry_room)
            self._add_member_hydration(band_id, member_id, drunk_water)
            self.member_carried_water[band_id, member_id] += carried_water
            water_gained = drunk_water + carried_water
        self._record_water_gained(band_id, member_id, water_gained)
        self.water_drinking_events += int(water_gained > 0.0)

    def _maybe_relocate_depleted_camp(self) -> None:
        self._update_depletion()
        current_depletion = self._local_camp_depletion(self.camp_pos)
        self.local_depletion_level = current_depletion
        if current_depletion < self.config.camp_depletion_threshold:
            return

        new_camp = self._best_camp_location()
        if new_camp == self.camp_pos:
            return
        if (
            self._local_camp_depletion(new_camp)
            >= self.config.camp_depletion_threshold
        ):
            return

        self._relocate_camp(new_camp)
        self.local_depletion_level = self._local_camp_depletion(
            self.camp_pos
        )

    def _update_depletion(self) -> None:
        capacity = getattr(
            self,
            "plant_capacity",
            np.zeros_like(self.plant_energy, dtype=np.float32),
        )
        depletion = np.zeros_like(self.plant_energy, dtype=np.float32)
        plant_cells = capacity > 0.0
        depletion[plant_cells] = 1.0 - np.clip(
            self.plant_energy[plant_cells] / capacity[plant_cells],
            0.0,
            1.0,
        )
        self.depletion = depletion.astype(np.float32)

    def _local_camp_depletion(self, camp: tuple[int, int]) -> float:
        in_area = self._camp_radius_mask(camp)
        plant_cells = self.plant_capacity > 0.0
        resource_cells = in_area & plant_cells
        if not np.any(resource_cells):
            return 1.0
        return float(np.mean(self.depletion[resource_cells]))

    def camp_potential_energy(
        self,
        camp: tuple[int, int] | None = None,
    ) -> float:
        """Remaining unreaped plant energy inside a camp's foraging radius."""
        if camp is None:
            camp = self.camp_pos
        in_area = self._camp_radius_mask(camp)
        return float(np.sum(self.plant_energy[in_area]))

    def _camp_radius_mask(self, camp: tuple[int, int]) -> np.ndarray:
        radius = self.config.camp_depletion_radius
        cy, cx = camp
        yy, xx = np.mgrid[0:self.grid_size, 0:self.grid_size]
        return (np.abs(yy - cy) + np.abs(xx - cx)) <= radius

    def _best_camp_location(self) -> tuple[int, int]:
        current = self.camp_pos
        water_distance = self._distance_to_nearest_water()
        best_key: tuple[float, float, int, int, int] | None = None
        best_camp = current
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.terrain[y, x] == Terrain.WATER:
                    continue
                candidate = (y, x)
                travel_distance = self._camp_distance(current, candidate)
                if self._camp_foraging_areas_overlap(
                    current,
                    candidate,
                ):
                    continue
                depletion = self._local_camp_depletion(candidate)
                key = (
                    depletion,
                    float(water_distance[candidate]),
                    travel_distance,
                    y,
                    x,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_camp = candidate
        return best_camp

    def _camp_foraging_areas_overlap(
        self,
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> bool:
        radius = self.config.camp_depletion_radius
        return self._camp_distance(first, second) <= 2 * radius

    @staticmethod
    def _camp_distance(
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    def _relocate_camp(self, new_camp: tuple[int, int]) -> None:
        old_y, old_x = self.camp_pos
        new_y, new_x = new_camp
        travel_distance = self._camp_distance(self.camp_pos, new_camp)
        self.macro_x += new_x - old_x
        self.macro_y += new_y - old_y
        self.camp_id += 1
        self.camp_age = 0
        self.num_camp_moves += 1
        self.macro_distance_traveled += travel_distance
        self.camp_pos = new_camp
        self.band_camp_positions[0] = new_camp
        self._last_camp_relocated = True
        self._spend_band_relocation_energy(0)
        self._place_band_members_near(0, new_camp)

    def _spend_band_relocation_energy(self, band_id: int) -> None:
        cost = self.config.camp_relocation_cost
        if cost <= 0.0:
            return
        for member_id in range(self.config.members_per_band):
            if self.member_alive[band_id, member_id]:
                self._spend_member_energy(band_id, member_id, cost)

    def _place_band_members_near(
        self,
        band_id: int,
        camp: tuple[int, int],
    ) -> None:
        occupied = {
            self._member_position(other_band_id, member_id)
            for other_band_id in range(self.config.num_bands)
            if other_band_id != band_id
            for member_id in range(self.config.members_per_band)
            if self.member_alive[other_band_id, member_id]
        }
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
        if band_id == 0:
            self.agent_pos = self._member_position(0, 0)

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

    def _spend_member_hydration(
        self,
        band_id: int,
        member_id: int,
        amount: float,
    ) -> None:
        amount = max(0.0, float(amount))
        self.member_hydration[band_id, member_id] -= amount
        self.member_last_hydration_spent[band_id, member_id] += amount
        if band_id == 0 and member_id == 0:
            self._last_hydration_spent += amount

    def _apply_thirst_to_alive_members(self) -> None:
        for band_id in range(self.config.num_bands):
            for member_id in range(self.config.members_per_band):
                if self.member_alive[band_id, member_id]:
                    self._spend_member_hydration(
                        band_id,
                        member_id,
                        self.config.thirst_per_step,
                    )

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

    def _record_water_gained(
        self,
        band_id: int,
        member_id: int,
        amount: float,
    ) -> None:
        amount = max(0.0, float(amount))
        self.member_last_water_gained[band_id, member_id] += amount
        if band_id == 0 and member_id == 0:
            self._last_water_gained += amount

    def _sync_stored_food_alias(self) -> None:
        self.stored_food = (
            float(self.camp_stored_food[0])
            if len(self.camp_stored_food) > 0
            else 0.0
        )

    def _member_at_band_camp(self, band_id: int, member_id: int) -> bool:
        return self._member_position(band_id, member_id) == tuple(
            int(value) for value in self.band_camp_positions[band_id]
        )

    def _exchange_camp_resources_for_alive_members(self) -> None:
        for band_id in range(self.config.num_bands):
            for member_id in range(self.config.members_per_band):
                if not self.member_alive[band_id, member_id]:
                    continue
                if not self._member_at_band_camp(band_id, member_id):
                    continue
                self._deposit_member_resources_at_camp(band_id, member_id)

            for member_id in range(self.config.members_per_band):
                if not self.member_alive[band_id, member_id]:
                    continue
                if not self._member_at_band_camp(band_id, member_id):
                    continue
                self._withdraw_member_resources_from_camp(band_id, member_id)

        self._sync_stored_food_alias()

    def _deposit_member_resources_at_camp(
        self,
        band_id: int,
        member_id: int,
    ) -> None:
        carried_food = float(self.member_carried_food[band_id, member_id])
        carried_water = float(self.member_carried_water[band_id, member_id])
        if carried_food > 0.0:
            self.camp_stored_food[band_id] += carried_food
            self.member_carried_food[band_id, member_id] = 0.0
        if carried_water > 0.0:
            self.camp_stored_water[band_id] += carried_water
            self.member_carried_water[band_id, member_id] = 0.0

    def _withdraw_member_resources_from_camp(
        self,
        band_id: int,
        member_id: int,
    ) -> None:
        energy_need = max(
            0.0,
            min(
                self.config.camp_food_withdraw_threshold,
                self.config.max_energy,
            )
            - self._member_energy(band_id, member_id),
        )
        food_taken = min(
            energy_need,
            self.config.camp_food_withdraw_amount,
            float(self.camp_stored_food[band_id]),
        )
        if food_taken > 0.0:
            self.camp_stored_food[band_id] -= food_taken
            self._add_member_energy(band_id, member_id, food_taken)
            self._record_food_gained(band_id, member_id, food_taken)

        hydration_need = max(
            0.0,
            min(
                self.config.camp_water_withdraw_threshold,
                self.config.max_hydration,
            )
            - self._member_hydration(band_id, member_id),
        )
        water_taken = min(
            hydration_need,
            self.config.camp_water_withdraw_amount,
            float(self.camp_stored_water[band_id]),
        )
        if water_taken > 0.0:
            self.camp_stored_water[band_id] -= water_taken
            self._add_member_hydration(band_id, member_id, water_taken)
            self._record_water_gained(band_id, member_id, water_taken)

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
            np.where(
                self.plant_energy > 0.0,
                self.config.plant_energy_capacity,
                0.0,
            ).astype(np.float32),
        )
        effective_regrowth = (
            self.config.plant_regrowth_base
            * season_mod
            * (plant_capacity > 0.0)
        )
        self.plant_energy = np.clip(
            self.plant_energy + effective_regrowth,
            0.0,
            plant_capacity,
        ).astype(np.float32)
        self.plant_energy[self.water > 0.0] = 0.0
        self._update_depletion()

    def _update_animals(self) -> None:
        self.animal_density.fill(0.0)

    def _plant_energy_observation_layer(self) -> np.ndarray:
        capacity = max(1.0, float(self.config.plant_energy_capacity))
        return np.clip(
            self.plant_energy / capacity,
            0.0,
            1.0,
        ).astype(np.float32)

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
            self._plant_energy_observation_layer(),
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
        hydration_plane = np.full(
            (self.config.obs_range, self.config.obs_range),
            np.clip(
                self._member_hydration(band_id, member_id)
                / self.config.max_hydration,
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
        carried_food_plane = np.full(
            (self.config.obs_range, self.config.obs_range),
            np.clip(
                self.member_carried_food[band_id, member_id]
                / max(1.0, self.config.food_carry_capacity),
                0.0,
                1.0,
            ),
            dtype=np.float32,
        )
        carried_water_plane = np.full(
            (self.config.obs_range, self.config.obs_range),
            np.clip(
                self.member_carried_water[band_id, member_id]
                / max(1.0, self.config.water_carry_capacity),
                0.0,
                1.0,
            ),
            dtype=np.float32,
        )
        camp_food_plane = np.full(
            (self.config.obs_range, self.config.obs_range),
            np.clip(
                self.camp_stored_food[band_id]
                / max(1.0, self.config.food_carry_capacity),
                0.0,
                1.0,
            ),
            dtype=np.float32,
        )
        camp_water_plane = np.full(
            (self.config.obs_range, self.config.obs_range),
            np.clip(
                self.camp_stored_water[band_id]
                / max(1.0, self.config.water_carry_capacity),
                0.0,
                1.0,
            ),
            dtype=np.float32,
        )
        padded_channels.extend(
            [
                energy_plane,
                hydration_plane,
                season_plane,
                carried_food_plane,
                carried_water_plane,
                camp_food_plane,
                camp_water_plane,
            ]
        )
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
        reward += 0.02 * float(
            self.member_last_water_gained[band_id, member_id]
        )
        reward -= 0.01 * float(
            self.member_last_energy_spent[band_id, member_id]
        )
        reward -= 0.10 * float(
            self.member_last_danger_damage[band_id, member_id]
        )
        reward += 0.01
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
            "member_hydration": self.member_hydration.tolist(),
            "member_carried_food": self.member_carried_food.tolist(),
            "member_carried_water": self.member_carried_water.tolist(),
            "active_agent_ids": self.active_agent_ids(),
            "controlled_agent_id": self._agent_id(0, 0),
            "num_bands": self.config.num_bands,
            "members_per_band": self.config.members_per_band,
            "energy": self.energy,
            "hydration": self.hydration,
            "stored_food": self.stored_food,
            "camp_stored_food": self.camp_stored_food.tolist(),
            "camp_stored_water": self.camp_stored_water.tolist(),
            "carried_food": float(self.member_carried_food[0, 0]),
            "carried_water": float(self.member_carried_water[0, 0]),
            "population": self.population,
            "max_population": (
                self.config.num_bands * self.config.members_per_band
            ),
            "local_depletion_level": self.local_depletion_level,
            "camp_potential_energy": self.camp_potential_energy(),
            "mean_plant_energy": float(np.mean(self.plant_energy)),
            "mean_plant_food": float(np.mean(self.plant_energy)),
            "mean_animal_density": float(np.mean(self.animal_density)),
            "mean_depletion": float(np.mean(self.depletion)),
            "num_camp_moves": self.num_camp_moves,
            "camp_relocated": self._last_camp_relocated,
            "macro_distance_traveled": self.macro_distance_traveled,
            "starvation_events": self.starvation_events,
            "dehydration_events": self.dehydration_events,
            "plant_eating_events": self.plant_eating_events,
            "water_drinking_events": self.water_drinking_events,
            "last_food_gained": self._last_food_gained,
            "last_water_gained": self._last_water_gained,
            "last_energy_spent": self._last_energy_spent,
            "last_hydration_spent": self._last_hydration_spent,
            "last_danger_damage": self._last_danger_damage,
            "member_last_food_gained": (
                self.member_last_food_gained.tolist()
            ),
            "member_last_water_gained": (
                self.member_last_water_gained.tolist()
            ),
            "member_last_energy_spent": (
                self.member_last_energy_spent.tolist()
            ),
            "member_last_hydration_spent": (
                self.member_last_hydration_spent.tolist()
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
        self._apply_member_moves(actions)
        self._apply_member_non_move_actions(actions)
        self._exchange_camp_resources_for_alive_members()
        self._apply_controlled_band_danger(active_slots)
        self._update_resources()
        self._update_animals()
        self._maybe_relocate_depleted_camp()

        self._apply_thirst_to_alive_members()
        self._advance_member_lifecycle()
        starvation_events, dehydration_events = self._update_member_survival()
        self.starvation_events += starvation_events
        self.dehydration_events += dehydration_events
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

    @staticmethod
    def _is_movement_action(action: Action) -> bool:
        return action in {
            Action.MOVE_NORTH,
            Action.MOVE_SOUTH,
            Action.MOVE_WEST,
            Action.MOVE_EAST,
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
        self._set_member_position(self.controlled_band_id, member_id, target)
        self._spend_member_energy(
            self.controlled_band_id,
            member_id,
            self._member_movement_cost(
                self.controlled_band_id,
                member_id,
                terrain,
                target,
            ),
        )
        self._consume_cell_resources(self.controlled_band_id, member_id)

    def _apply_member_non_move_actions(
        self,
        actions: dict[int, Action],
    ) -> None:
        for member_id, action in actions.items():
            if self._is_movement_action(action):
                continue
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
            "hydration": state.hydration,
            "position": state.position,
            "step_count": self.step_count,
            "season": Season(self.season).name.lower(),
            "camp_id": self.camp_id,
            "camp_pressure": self._last_camp_pressure,
            "local_depletion_level": self.local_depletion_level,
            "camp_potential_energy": self.camp_potential_energy(),
            "last_food_gained": float(
                self.member_last_food_gained[band_id, member_id]
            ),
            "last_water_gained": float(
                self.member_last_water_gained[band_id, member_id]
            ),
            "last_energy_spent": float(
                self.member_last_energy_spent[band_id, member_id]
            ),
            "last_hydration_spent": float(
                self.member_last_hydration_spent[band_id, member_id]
            ),
            "last_danger_damage": float(
                self.member_last_danger_damage[band_id, member_id]
            ),
        }
