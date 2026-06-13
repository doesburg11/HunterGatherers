from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import IntEnum
import math
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
    body_mass_kg: float
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
    max_members_per_band: int = _default_config_value(
        "max_members_per_band",
        0,
    )
    initial_energy: float = _default_config_value("initial_energy", 100.0)
    max_energy: float = _default_config_value("max_energy", 150.0)
    initial_hydration: float = _default_config_value(
        "initial_hydration",
        100.0,
    )
    max_hydration: float = _default_config_value("max_hydration", 100.0)
    thirst_per_step: float = _default_config_value("thirst_per_step", 1.0)
    drink_amount: float = _default_config_value("drink_amount", 25.0)
    use_physical_energetics: bool = _default_config_value(
        "use_physical_energetics",
        False,
    )
    adult_body_mass_kg: float = _default_config_value(
        "adult_body_mass_kg",
        70.0,
    )
    newborn_body_mass_kg: float = _default_config_value(
        "newborn_body_mass_kg",
        12.0,
    )
    initial_energy_per_kg: float = _default_config_value(
        "initial_energy_per_kg",
        30.0,
    )
    max_energy_per_kg: float = _default_config_value(
        "max_energy_per_kg",
        35.0,
    )
    newborn_energy_per_kg: float = _default_config_value(
        "newborn_energy_per_kg",
        30.0,
    )
    cell_distance_km: float = _default_config_value("cell_distance_km", 0.25)
    walking_kcal_per_kg_km: float = _default_config_value(
        "walking_kcal_per_kg_km",
        1.0,
    )
    food_energy_density_kcal_per_kg: float = _default_config_value(
        "food_energy_density_kcal_per_kg",
        2500.0,
    )
    water_liters_per_unit: float = _default_config_value(
        "water_liters_per_unit",
        1.0,
    )
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
    plant_carry_share: float = _default_config_value(
        "plant_carry_share",
        0.5,
    )
    grass_energy_capacity: float = _default_config_value(
        "grass_energy_capacity",
        1000.0,
    )
    grass_regrowth_base: float = _default_config_value(
        "grass_regrowth_base",
        0.25,
    )
    animal_energy_capacity: float = _default_config_value(
        "animal_energy_capacity",
        2500.0,
    )
    initial_animals: int = _default_config_value("initial_animals", 35)
    max_animals: int = _default_config_value("max_animals", 120)
    initial_animal_energy: float = _default_config_value(
        "initial_animal_energy",
        1200.0,
    )
    max_animal_energy: float = _default_config_value(
        "max_animal_energy",
        2500.0,
    )
    animal_move_cost: float = _default_config_value("animal_move_cost", 8.0)
    animal_eat_amount: float = _default_config_value(
        "animal_eat_amount",
        120.0,
    )
    animal_reproduction_energy_threshold: float = _default_config_value(
        "animal_reproduction_energy_threshold",
        2000.0,
    )
    animal_reproduction_cost: float = _default_config_value(
        "animal_reproduction_cost",
        900.0,
    )
    newborn_animal_energy: float = _default_config_value(
        "newborn_animal_energy",
        800.0,
    )
    animal_hunt_amount: float = _default_config_value(
        "animal_hunt_amount",
        800.0,
    )
    animal_carry_share: float = _default_config_value(
        "animal_carry_share",
        1.0,
    )
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
    carried_food_consume_threshold: float = _default_config_value(
        "carried_food_consume_threshold",
        80.0,
    )
    carried_water_consume_threshold: float = _default_config_value(
        "carried_water_consume_threshold",
        80.0,
    )
    camp_storage_radius: int = _default_config_value(
        "camp_storage_radius",
        1,
    )
    camp_max_stored_food: float = _default_config_value(
        "camp_max_stored_food",
        0.0,
    )
    camp_max_stored_water: float = _default_config_value(
        "camp_max_stored_water",
        0.0,
    )
    camp_food_decay_rate: float = _default_config_value(
        "camp_food_decay_rate",
        0.0,
    )
    river_crossing_width: int = _default_config_value(
        "river_crossing_width",
        3,
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
    camp_emergency_depletion_threshold: float = _default_config_value(
        "camp_emergency_depletion_threshold",
        0.90,
    )
    camp_relocation_cost: float = _default_config_value(
        "camp_relocation_cost",
        0.0,
    )
    camp_store_departure_steps: int = _default_config_value(
        "camp_store_departure_steps",
        10,
    )
    camp_max_water_distance: int = _default_config_value(
        "camp_max_water_distance",
        0,
    )
    division_of_labor_enabled: bool = _default_config_value(
        "division_of_labor_enabled",
        False,
    )
    reproduction_enabled: bool = _default_config_value(
        "reproduction_enabled",
        True,
    )
    reproduction_adult_age: int = _default_config_value(
        "reproduction_adult_age",
        250,
    )
    max_age: int = _default_config_value("max_age", 1000)
    # Fraction of max energy needed to trigger asexual reproduction.
    reproduction_energy_threshold: float = _default_config_value(
        "reproduction_energy_threshold", 0.8
    )
    # Steps before a member can reproduce again after giving birth.
    reproduction_cooldown_steps: int = _default_config_value(
        "reproduction_cooldown_steps", 500
    )
    # Probability of reproducing per step when above the energy threshold.
    birth_rate: float = _default_config_value("birth_rate", 1.0)
    # Minimum camp stored food required for a birth (consumed on birth).
    birth_food_cost: float = _default_config_value("birth_food_cost", 0.0)
    # Age in years at which probabilistic old-age mortality begins.
    mortality_onset_years: float = _default_config_value(
        "mortality_onset_years", 25.0
    )
    # Yearly mortality rate increase per year beyond onset age.
    mortality_rate_per_year: float = _default_config_value(
        "mortality_rate_per_year", 0.05
    )
    # Maximum juveniles per adult before random juvenile mortality kicks in.
    max_juveniles_per_adult: float = _default_config_value(
        "max_juveniles_per_adult", 2.0
    )
    newborn_hydration: float = _default_config_value(
        "newborn_hydration",
        80.0,
    )
    sexual_reproduction_enabled: bool = _default_config_value(
        "sexual_reproduction_enabled",
        False,
    )
    male_proximity_distance: int = _default_config_value(
        "male_proximity_distance",
        5,
    )
    plant_food_reward: float = _default_config_value(
        "plant_food_reward",
        0.05,
    )
    animal_food_reward: float = _default_config_value(
        "animal_food_reward",
        0.10,
    )
    food_deposit_reward: float = _default_config_value(
        "food_deposit_reward",
        0.03,
    )
    camp_food_withdraw_reward: float = _default_config_value(
        "camp_food_withdraw_reward",
        0.02,
    )
    camp_water_withdraw_reward: float = _default_config_value(
        "camp_water_withdraw_reward",
        0.20,
    )
    water_collect_reward: float = _default_config_value(
        "water_collect_reward",
        0.10,
    )
    water_deposit_reward: float = _default_config_value(
        "water_deposit_reward",
        0.30,
    )
    birth_reward: float = _default_config_value("birth_reward", 2.0)
    population_reward: float = _default_config_value(
        "population_reward",
        0.001,
    )
    juvenile_survival_reward: float = _default_config_value(
        "juvenile_survival_reward",
        0.002,
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

    @property
    def member_capacity_per_band(self) -> int:
        if self.config.max_members_per_band <= 0:
            return int(self.config.members_per_band)
        return int(self.config.max_members_per_band)

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
        if (
            self.config.max_members_per_band > 0
            and self.config.max_members_per_band < self.config.members_per_band
        ):
            raise ValueError(
                "max_members_per_band must be at least members_per_band"
            )
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
        if self.config.camp_store_departure_steps < 0:
            raise ValueError("camp_store_departure_steps must be non-negative")
        if self.config.reproduction_adult_age < 0:
            raise ValueError("reproduction_adult_age must be non-negative")
        if not 0.0 <= self.config.reproduction_energy_threshold <= 1.0:
            raise ValueError(
                "reproduction_energy_threshold must be between 0 and 1"
            )
        if self.config.reproduction_cooldown_steps < 0:
            raise ValueError(
                "reproduction_cooldown_steps must be non-negative"
            )
        if not 0.0 <= self.config.birth_rate <= 1.0:
            raise ValueError("birth_rate must be between 0 and 1")
        if self.config.birth_food_cost < 0.0:
            raise ValueError("birth_food_cost must be non-negative")
        if self.config.mortality_onset_years < 0.0:
            raise ValueError("mortality_onset_years must be non-negative")
        if self.config.mortality_rate_per_year < 0.0:
            raise ValueError(
                "mortality_rate_per_year must be non-negative"
            )
        if self.config.newborn_hydration < 0.0:
            raise ValueError("newborn_hydration must be non-negative")
        if self.config.adult_body_mass_kg <= 0.0:
            raise ValueError("adult_body_mass_kg must be positive")
        if self.config.newborn_body_mass_kg <= 0.0:
            raise ValueError("newborn_body_mass_kg must be positive")
        if self.config.initial_energy_per_kg < 0.0:
            raise ValueError("initial_energy_per_kg must be non-negative")
        if self.config.max_energy_per_kg <= 0.0:
            raise ValueError("max_energy_per_kg must be positive")
        if self.config.newborn_energy_per_kg < 0.0:
            raise ValueError("newborn_energy_per_kg must be non-negative")
        if self.config.cell_distance_km <= 0.0:
            raise ValueError("cell_distance_km must be positive")
        if self.config.walking_kcal_per_kg_km < 0.0:
            raise ValueError("walking_kcal_per_kg_km must be non-negative")
        if self.config.food_energy_density_kcal_per_kg <= 0.0:
            raise ValueError(
                "food_energy_density_kcal_per_kg must be positive"
            )
        if self.config.water_liters_per_unit < 0.0:
            raise ValueError("water_liters_per_unit must be non-negative")
        if self.config.plant_food_reward < 0.0:
            raise ValueError("plant_food_reward must be non-negative")
        if self.config.animal_food_reward < 0.0:
            raise ValueError("animal_food_reward must be non-negative")
        if self.config.food_deposit_reward < 0.0:
            raise ValueError("food_deposit_reward must be non-negative")
        if self.config.water_collect_reward < 0.0:
            raise ValueError("water_collect_reward must be non-negative")
        if self.config.water_deposit_reward < 0.0:
            raise ValueError("water_deposit_reward must be non-negative")
        if self.config.birth_reward < 0.0:
            raise ValueError("birth_reward must be non-negative")
        if self.config.population_reward < 0.0:
            raise ValueError("population_reward must be non-negative")
        if self.config.juvenile_survival_reward < 0.0:
            raise ValueError("juvenile_survival_reward must be non-negative")
        if self.config.movement_energy_rate < 0.0:
            raise ValueError("movement_energy_rate must be non-negative")
        if self.config.water_load_factor < 0.0:
            raise ValueError("water_load_factor must be non-negative")
        if self.config.food_carry_capacity < 0.0:
            raise ValueError("food_carry_capacity must be non-negative")
        if not 0.0 <= self.config.plant_carry_share <= 1.0:
            raise ValueError("plant_carry_share must be between 0 and 1")
        if self.config.grass_energy_capacity < 0.0:
            raise ValueError("grass_energy_capacity must be non-negative")
        if self.config.grass_regrowth_base < 0.0:
            raise ValueError("grass_regrowth_base must be non-negative")
        if self.config.animal_energy_capacity < 0.0:
            raise ValueError("animal_energy_capacity must be non-negative")
        if self.config.initial_animals < 0:
            raise ValueError("initial_animals must be non-negative")
        if self.config.max_animals < 0:
            raise ValueError("max_animals must be non-negative")
        if self.config.initial_animals > self.config.max_animals:
            raise ValueError("initial_animals cannot exceed max_animals")
        if self.config.initial_animal_energy < 0.0:
            raise ValueError("initial_animal_energy must be non-negative")
        if self.config.max_animal_energy <= 0.0:
            raise ValueError("max_animal_energy must be positive")
        if self.config.animal_move_cost < 0.0:
            raise ValueError("animal_move_cost must be non-negative")
        if self.config.animal_eat_amount < 0.0:
            raise ValueError("animal_eat_amount must be non-negative")
        if self.config.animal_reproduction_energy_threshold < 0.0:
            raise ValueError(
                "animal_reproduction_energy_threshold must be non-negative"
            )
        if self.config.animal_reproduction_cost < 0.0:
            raise ValueError("animal_reproduction_cost must be non-negative")
        if self.config.newborn_animal_energy < 0.0:
            raise ValueError("newborn_animal_energy must be non-negative")
        if self.config.animal_hunt_amount < 0.0:
            raise ValueError("animal_hunt_amount must be non-negative")
        if not 0.0 <= self.config.animal_carry_share <= 1.0:
            raise ValueError("animal_carry_share must be between 0 and 1")
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
        if self.config.carried_food_consume_threshold < 0.0:
            raise ValueError(
                "carried_food_consume_threshold must be non-negative"
            )
        if self.config.carried_water_consume_threshold < 0.0:
            raise ValueError(
                "carried_water_consume_threshold must be non-negative"
            )
        if self.config.camp_storage_radius < 0:
            raise ValueError("camp_storage_radius must be non-negative")
        if self.config.camp_max_stored_food < 0.0:
            raise ValueError("camp_max_stored_food must be non-negative")
        if self.config.camp_max_stored_water < 0.0:
            raise ValueError("camp_max_stored_water must be non-negative")
        if not 0.0 <= self.config.camp_food_decay_rate <= 1.0:
            raise ValueError(
                "camp_food_decay_rate must be between 0 and 1"
            )
        if self.config.river_crossing_width < 0:
            raise ValueError("river_crossing_width must be non-negative")
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
                self.member_capacity_per_band,
                2,
            ),
            dtype=np.int16,
        )
        self.member_ids: np.ndarray = np.empty(
            (self.config.num_bands, self.member_capacity_per_band),
            dtype=np.int32,
        )
        self.next_member_ids: np.ndarray = np.empty(
            self.config.num_bands,
            dtype=np.int32,
        )
        self.member_alive: np.ndarray = np.empty(
            (self.config.num_bands, self.member_capacity_per_band),
            dtype=bool,
        )
        self.member_age: np.ndarray = np.empty(
            (self.config.num_bands, self.member_capacity_per_band),
            dtype=np.int32,
        )
        self.member_sex: np.ndarray = np.empty(
            (self.config.num_bands, self.member_capacity_per_band),
            dtype=np.int8,
        )
        self.member_energy: np.ndarray = np.empty(
            (self.config.num_bands, self.member_capacity_per_band),
            dtype=np.float32,
        )
        self.member_body_mass_kg: np.ndarray = np.empty(
            (self.config.num_bands, self.member_capacity_per_band),
            dtype=np.float32,
        )
        self.member_hydration: np.ndarray = np.empty(
            (self.config.num_bands, self.member_capacity_per_band),
            dtype=np.float32,
        )
        self.member_carried_food: np.ndarray = np.empty(
            (self.config.num_bands, self.member_capacity_per_band),
            dtype=np.float32,
        )
        self.member_carried_water: np.ndarray = np.empty(
            (self.config.num_bands, self.member_capacity_per_band),
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
            (self.config.num_bands, self.member_capacity_per_band),
            dtype=np.float32,
        )
        self.member_last_animal_food_gained: np.ndarray = np.empty(
            (self.config.num_bands, self.member_capacity_per_band),
            dtype=np.float32,
        )
        self.member_last_water_gained: np.ndarray = np.empty(
            (self.config.num_bands, self.member_capacity_per_band),
            dtype=np.float32,
        )
        self.member_last_food_deposited: np.ndarray = np.empty(
            (self.config.num_bands, self.member_capacity_per_band),
            dtype=np.float32,
        )
        self.member_last_water_deposited: np.ndarray = np.empty(
            (self.config.num_bands, self.member_capacity_per_band),
            dtype=np.float32,
        )
        self.member_last_energy_spent: np.ndarray = np.empty(
            (self.config.num_bands, self.member_capacity_per_band),
            dtype=np.float32,
        )
        self.member_last_hydration_spent: np.ndarray = np.empty(
            (self.config.num_bands, self.member_capacity_per_band),
            dtype=np.float32,
        )
        self.member_last_action: np.ndarray = np.empty(
            (self.config.num_bands, self.member_capacity_per_band),
            dtype=np.int8,
        )
        self.member_reproduction_cooldown: np.ndarray = np.empty(
            (self.config.num_bands, self.member_capacity_per_band),
            dtype=np.int32,
        )
        self.animal_alive: np.ndarray = np.empty(
            self.config.max_animals,
            dtype=bool,
        )
        self.animal_positions: np.ndarray = np.empty(
            (self.config.max_animals, 2),
            dtype=np.int16,
        )
        self.animal_individual_energy: np.ndarray = np.empty(
            self.config.max_animals,
            dtype=np.float32,
        )
        # Division of labor tracking (plants gathered, animals hunted)
        self.plants_gathered_by_females: float = 0.0
        self.plants_gathered_by_males: float = 0.0
        self.animals_hunted_by_females: float = 0.0
        self.animals_hunted_by_males: float = 0.0
        self.reset()

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
                self.member_capacity_per_band,
                2,
            ),
            dtype=np.int16,
        )
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
        self._prev_camp_pos: tuple[int, int] | None = None
        self._migration_target: tuple[int, int] | None = None
        self._migration_start_pos: tuple[int, int] | None = None
        self._migration_carries_stores: bool = False
        self.local_depletion_level = 0.0
        self.num_camp_moves = 0
        self.plant_eating_events = 0
        self.water_drinking_events = 0
        self.birth_events = 0
        self.starvation_events = 0
        self.dehydration_events = 0
        self.old_age_events = 0
        self.macro_distance_traveled = 0
        self.plants_gathered_by_females = 0.0
        self.plants_gathered_by_males = 0.0
        self.animals_hunted_by_females = 0.0
        self.animals_hunted_by_males = 0.0

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
        member_ids = np.full(
            self.member_capacity_per_band, -1, dtype=np.int32
        )
        member_ids[: self.config.members_per_band] = np.arange(
            self.config.members_per_band, dtype=np.int32
        )
        self.member_ids = np.tile(member_ids, (self.config.num_bands, 1))
        self.next_member_ids = np.full(
            self.config.num_bands,
            self.config.members_per_band,
            dtype=np.int32,
        )
        self.member_alive = np.zeros(
            (self.config.num_bands, self.member_capacity_per_band),
            dtype=bool,
        )
        self.member_alive[:, : self.config.members_per_band] = True
        self.member_age = np.zeros(
            (self.config.num_bands, self.member_capacity_per_band),
            dtype=np.int32,
        )
        self.member_age[
            :,
            : self.config.members_per_band,
        ] = self.config.reproduction_adult_age
        self.member_sex = np.array(
            [
                [
                    (
                        band_id
                        + max(int(self.member_ids[band_id, slot]), 0)
                    )
                    % 2
                    for slot in range(self.member_capacity_per_band)
                ]
                for band_id in range(self.config.num_bands)
            ],
            dtype=np.int8,
        )
        self.member_energy = np.full(
            (self.config.num_bands, self.member_capacity_per_band),
            0.0,
            dtype=np.float32,
        )
        self.member_body_mass_kg = np.zeros(
            (self.config.num_bands, self.member_capacity_per_band),
            dtype=np.float32,
        )
        self.member_body_mass_kg[
            :,
            : self.config.members_per_band,
        ] = self.config.adult_body_mass_kg
        self.member_energy[
            :,
            : self.config.members_per_band,
        ] = self._initial_adult_energy(initial_energy)
        self.member_hydration = np.full(
            (self.config.num_bands, self.member_capacity_per_band),
            0.0,
            dtype=np.float32,
        )
        self.member_hydration[
            :,
            : self.config.members_per_band,
        ] = initial_hydration
        self.member_carried_food = np.zeros(
            (self.config.num_bands, self.member_capacity_per_band),
            dtype=np.float32,
        )
        self.member_carried_water = np.zeros(
            (self.config.num_bands, self.member_capacity_per_band),
            dtype=np.float32,
        )
        self.member_reproduction_cooldown = np.zeros(
            (self.config.num_bands, self.member_capacity_per_band),
            dtype=np.int32,
        )
        self._sync_population()
        self._clear_member_step_stats()

    def _clear_member_step_stats(self) -> None:
        shape = (self.config.num_bands, self.member_capacity_per_band)
        self.member_last_food_gained = np.zeros(shape, dtype=np.float32)
        self.member_last_animal_food_gained = np.zeros(
            shape, dtype=np.float32
        )
        self.member_last_water_gained = np.zeros(shape, dtype=np.float32)
        self.member_last_food_deposited = np.zeros(shape, dtype=np.float32)
        self.member_last_water_deposited = np.zeros(shape, dtype=np.float32)
        self.member_last_food_withdrawn = np.zeros(shape, dtype=np.float32)
        self.member_last_water_withdrawn = np.zeros(shape, dtype=np.float32)
        self.member_last_energy_spent = np.zeros(shape, dtype=np.float32)
        self.member_last_hydration_spent = np.zeros(shape, dtype=np.float32)
        self.member_last_action = np.full(
            shape,
            int(Action.STAY),
            dtype=np.int8,
        )
        self._last_camp_relocated = False
        self._last_action = Action.STAY
        self._last_birth_events = np.zeros(
            self.config.num_bands,
            dtype=np.int32,
        )

    def _sync_population(self) -> None:
        self.population = int(np.count_nonzero(self.member_alive))

    def _advance_member_lifecycle(self) -> None:
        self.member_age[self.member_alive] += 1
        self._update_member_body_masses()
        # Tick reproduction cooldowns for all living members.
        ticking = self.member_alive & (self.member_reproduction_cooldown > 0)
        self.member_reproduction_cooldown[ticking] -= 1

    def _update_member_body_masses(self) -> None:
        if not self.config.use_physical_energetics:
            return
        alive = self.member_alive
        if self.config.reproduction_adult_age <= 0:
            self.member_body_mass_kg[alive] = self.config.adult_body_mass_kg
            return
        age_fraction = np.clip(
            self.member_age.astype(np.float32)
            / float(self.config.reproduction_adult_age),
            0.0,
            1.0,
        )
        juvenile_mass = (
            self.config.newborn_body_mass_kg
            + age_fraction
            * (
                self.config.adult_body_mass_kg
                - self.config.newborn_body_mass_kg
            )
        )
        self.member_body_mass_kg[alive] = juvenile_mass[alive]

    def _kill_member(self, band_id: int, member_id: int) -> None:
        if not self.member_alive[band_id, member_id]:
            return
        self.member_alive[band_id, member_id] = False
        self.member_energy[band_id, member_id] = 0.0
        self.member_body_mass_kg[band_id, member_id] = 0.0
        self.member_hydration[band_id, member_id] = 0.0
        self.member_carried_food[band_id, member_id] = 0.0
        self.member_carried_water[band_id, member_id] = 0.0
        self.member_reproduction_cooldown[band_id, member_id] = 0
        self._sync_population()

    def _cull_excess_juveniles(self) -> int:
        if self.config.max_juveniles_per_adult <= 0:
            return 0
        killed = 0
        for band_id in range(self.config.num_bands):
            alive = np.flatnonzero(self.member_alive[band_id])
            ages = self.member_age[band_id, alive]
            is_juvenile = ages < self.config.reproduction_adult_age
            adult_count = int(np.sum(~is_juvenile))
            juvenile_ids = alive[is_juvenile]
            max_juveniles = int(
                adult_count * self.config.max_juveniles_per_adult
            )
            excess = len(juvenile_ids) - max_juveniles
            if excess <= 0:
                continue
            chosen = self._episode_rng.choice(
                juvenile_ids, size=excess, replace=False
            )
            for member_id in chosen:
                self._kill_member(band_id, int(member_id))
                killed += 1
        return killed

    def _update_member_survival(self) -> tuple[int, int, int]:
        starving = self.member_alive & (self.member_energy <= 0.0)
        dehydrating = self.member_alive & (self.member_hydration <= 0.0)
        aged_out = self.member_alive & (self.member_age >= self.config.max_age)
        aged_out |= self._probabilistic_old_age_mortality()
        dying_members = np.argwhere(starving | dehydrating | aged_out)
        starvation_count = int(np.count_nonzero(starving))
        dehydration_count = int(np.count_nonzero(dehydrating))
        old_age_count = int(
            np.count_nonzero(aged_out & ~starving & ~dehydrating)
        )
        for band_id, member_id in dying_members:
            self._kill_member(int(band_id), int(member_id))
        return starvation_count, dehydration_count, old_age_count

    def _probabilistic_old_age_mortality(self) -> np.ndarray:
        if self.config.mortality_rate_per_year <= 0.0:
            return np.zeros_like(self.member_alive)
        steps_per_year = (
            self.config.reproduction_adult_age
            / max(1.0, 15.0)
        )
        age_years = (
            self.member_age.astype(np.float32)
            / max(1.0, steps_per_year)
        )
        years_over = np.maximum(
            0.0, age_years - self.config.mortality_onset_years
        )
        yearly_rate = np.minimum(
            1.0, years_over * self.config.mortality_rate_per_year
        )
        p_step = yearly_rate / max(1.0, steps_per_year)
        roll = self._episode_rng.random(p_step.shape).astype(np.float32)
        return self.member_alive & (roll < p_step)

    def _maybe_reproduce_bands(self) -> None:
        if not self.config.reproduction_enabled:
            return
        for band_id in range(self.config.num_bands):
            self._maybe_reproduce_band(band_id)

    def _maybe_reproduce_band(self, band_id: int) -> None:
        """Reproduction: a member above the energy threshold reproduces.

        When sexual_reproduction_enabled=False: Asexual reproduction where any
        member (male or female) can reproduce, splitting energy 1:1 with child.

        When sexual_reproduction_enabled=True: Sexual reproduction where only
        females reproduce, and only if a male exists within
        male_proximity_distance.
        """
        for member_id in range(self.member_capacity_per_band):
            if not self.member_alive[band_id, member_id]:
                continue
            if (
                self.member_age[band_id, member_id]
                < self.config.reproduction_adult_age
            ):
                continue

            # Sexual reproduction: only females can reproduce
            if self.config.sexual_reproduction_enabled:
                if self._member_sex(band_id, member_id) != MemberSex.FEMALE:
                    continue
                # Female must have a nearby male to reproduce
                if not self._has_nearby_male(band_id, member_id):
                    continue

            if self.member_reproduction_cooldown[band_id, member_id] > 0:
                continue
            max_energy = self._member_max_energy(band_id, member_id)
            if self._member_energy(band_id, member_id) < (
                self.config.reproduction_energy_threshold * max_energy
            ):
                continue
            if (
                self.config.birth_rate < 1.0
                and self._episode_rng.random() >= self.config.birth_rate
            ):
                continue
            if (
                self.config.birth_food_cost > 0.0
                and float(self.camp_stored_food[band_id])
                < self.config.birth_food_cost
            ):
                continue
            empty_slots = np.flatnonzero(~self.member_alive[band_id])
            if len(empty_slots) == 0:
                break
            try:
                newborn_pos = self._newborn_position(band_id)
            except RuntimeError:
                continue
            if self.config.birth_food_cost > 0.0:
                self.camp_stored_food[band_id] -= self.config.birth_food_cost
                self._sync_stored_food_alias()
            parent_energy = self._member_energy(band_id, member_id)
            child_energy = parent_energy * 0.2
            self._set_member_energy(band_id, member_id, parent_energy * 0.8)
            if self.config.reproduction_cooldown_steps > 0:
                self.member_reproduction_cooldown[band_id, member_id] = (
                    self.config.reproduction_cooldown_steps
                )
            slot = int(empty_slots[0])
            self._activate_newborn(band_id, slot, newborn_pos, child_energy)
            self.birth_events += 1
            self._last_birth_events[band_id] += 1
            if self.config.sexual_reproduction_enabled:
                parent_sex = self._member_sex(band_id, member_id)
                print(
                    f"[Step {self.step_count}] Sexual reproduction: "
                    f"female (band {band_id}, slot {member_id}) gave birth. "
                    f"Newborn in slot {slot}."
                )
            else:
                parent_sex = self._member_sex(band_id, member_id)
                is_female = parent_sex == MemberSex.FEMALE
                parent_sex_label = "female" if is_female else "male"
                print(
                    f"[Step {self.step_count}] Asexual reproduction: "
                    f"{parent_sex_label} (band {band_id}, slot {member_id}) "
                    f"reproduced. Newborn in slot {slot}."
                )

    def _member_is_adult(self, band_id: int, member_id: int) -> bool:
        return bool(
            self.member_age[band_id, member_id]
            >= self.config.reproduction_adult_age
        )

    def _newborn_position_near(
        self, band_id: int, center: tuple[int, int]
    ) -> tuple[int, int]:
        occupied = {
            self._member_position(other_band_id, mid)
            for other_band_id in range(self.config.num_bands)
            for mid in range(self.member_capacity_per_band)
            if self.member_alive[other_band_id, mid]
        }
        cells = self._member_cells_near(center, occupied, needed=1)
        y, x = cells[0]
        return int(y), int(x)

    def _newborn_position(self, band_id: int) -> tuple[int, int]:
        camp_pos = tuple(
            int(v) for v in self.band_camp_positions[band_id]
        )
        return self._newborn_position_near(band_id, camp_pos)

    def _activate_newborn(
        self,
        band_id: int,
        slot: int,
        pos: tuple[int, int],
        energy: float | None = None,
    ) -> None:
        self.member_ids[band_id, slot] = self.next_member_ids[band_id]
        self.next_member_ids[band_id] += 1
        self.member_sex[band_id, slot] = (
            band_id + int(self.member_ids[band_id, slot])
        ) % 2
        self.member_alive[band_id, slot] = True
        self.member_age[band_id, slot] = 0
        self.member_body_mass_kg[band_id, slot] = (
            self.config.newborn_body_mass_kg
        )
        self.member_energy[band_id, slot] = min(
            self._member_max_energy(band_id, slot),
            energy if energy is not None else self._newborn_energy(),
        )
        self.member_hydration[band_id, slot] = min(
            self._member_max_hydration(band_id, slot),
            self.config.newborn_hydration,
        )
        self.member_carried_food[band_id, slot] = 0.0
        self.member_carried_water[band_id, slot] = 0.0
        self.member_last_food_gained[band_id, slot] = 0.0
        self.member_last_animal_food_gained[band_id, slot] = 0.0
        self.member_last_water_gained[band_id, slot] = 0.0
        self.member_last_food_deposited[band_id, slot] = 0.0
        self.member_last_water_deposited[band_id, slot] = 0.0
        self.member_last_food_withdrawn[band_id, slot] = 0.0
        self.member_last_water_withdrawn[band_id, slot] = 0.0
        self.member_last_energy_spent[band_id, slot] = 0.0
        self.member_last_hydration_spent[band_id, slot] = 0.0
        self.member_last_action[band_id, slot] = int(Action.STAY)
        self.member_reproduction_cooldown[band_id, slot] = 0
        self._set_member_position(band_id, slot, pos)
        self._sync_population()

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

        if self._member_is_adult(0, 0) and action in {
            Action.MOVE_NORTH,
            Action.MOVE_SOUTH,
            Action.MOVE_WEST,
            Action.MOVE_EAST,
        }:
            self._move_member(0, 0, action)
        else:
            self._spend_member_energy(0, 0, 0.15)
            # Drink from the current cell even without moving — plants and
            # animals are only harvested on arrival (via _move_member).
            self._consume_water(0, 0)

        self._exchange_camp_resources_for_alive_members()
        self._update_resources()
        self._update_animals()
        self._maybe_relocate_depleted_camp()
        self._step_migration()

        self._apply_thirst_to_alive_members()
        self._advance_member_lifecycle()
        (
            starvation_events,
            dehydration_events,
            old_age_events,
        ) = self._update_member_survival()
        self._cull_excess_juveniles()
        self.starvation_events += starvation_events
        self.dehydration_events += dehydration_events
        self.old_age_events += old_age_events
        self._maybe_reproduce_bands()
        self._last_camp_pressure = self._camp_pressure()
        terminated = not bool(self.member_alive[0, 0])
        truncated = self.step_count >= self.config.max_steps
        # Print division of labor stats every 100 steps or at episode end
        if self.step_count % 100 == 0 or terminated or truncated:
            self._print_division_of_labor_stats()
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
            for member_id in range(self.member_capacity_per_band):
                if not self.member_alive[band_id, member_id]:
                    continue
                y, x = self._member_position(band_id, member_id)
                if band_id == 0:
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
        self.grass_energy = self._generate_grass()
        self.plant_energy = self._generate_plants()
        self._initialize_animals()
        self.depletion = np.zeros(
            (self.grid_size, self.grid_size),
            dtype=np.float32,
        )
        cy, cx = self.camp_pos
        if self.terrain[cy, cx] == Terrain.WATER:
            self.terrain[cy, cx] = Terrain.GRASSLAND
            self.water[cy, cx] = 0.0
            self.grass_energy[cy, cx] = self.config.grass_energy_capacity
            self.animal_energy[cy, cx] = 0.0
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
                for member_id in range(self.member_capacity_per_band)
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
            needed = self.member_capacity_per_band
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

    def _agent_id(self, band_id: int, member_id: int) -> str:
        member_identity = int(self.member_ids[band_id, member_id])
        return f"band_{band_id}_member_{member_identity}"

    def _member_energy(self, band_id: int, member_id: int) -> float:
        return float(self.member_energy[band_id, member_id])

    def _initial_adult_energy(self, fallback_energy: float) -> float:
        if not self.config.use_physical_energetics:
            return float(fallback_energy)
        return float(
            self.config.adult_body_mass_kg * self.config.initial_energy_per_kg
        )

    def _member_max_energy(self, band_id: int, member_id: int) -> float:
        if not self.config.use_physical_energetics:
            return float(self.config.max_energy)
        return float(
            self.member_body_mass_kg[band_id, member_id]
            * self.config.max_energy_per_kg
        )

    def _member_max_hydration(self, band_id: int, member_id: int) -> float:
        mass_fraction = (
            float(self.member_body_mass_kg[band_id, member_id])
            / self.config.adult_body_mass_kg
        )
        return float(self.config.max_hydration * mass_fraction)

    def _member_personal_energy_reserve(
        self,
        band_id: int,
        member_id: int,
    ) -> float:
        if not self.config.use_physical_energetics:
            return float(self.config.personal_energy_reserve)
        return self._member_max_energy(band_id, member_id)

    def _newborn_energy(self) -> float:
        return float(
            self.config.newborn_body_mass_kg
            * self.config.newborn_energy_per_kg
        )

    def _member_total_move_mass_kg(
        self,
        band_id: int,
        member_id: int,
    ) -> float:
        carried_food_kg = (
            float(self.member_carried_food[band_id, member_id])
            / self.config.food_energy_density_kcal_per_kg
        )
        carried_water_kg = (
            float(self.member_carried_water[band_id, member_id])
            * self.config.water_liters_per_unit
        )
        return float(
            self.member_body_mass_kg[band_id, member_id]
            + carried_food_kg
            + carried_water_kg
        )

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
            min(self._member_max_energy(band_id, member_id), energy),
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
            min(self._member_max_hydration(band_id, member_id), hydration),
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
            body_mass_kg=float(self.member_body_mass_kg[band_id, member_id]),
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
            for member_id in range(self.member_capacity_per_band):
                if not self.member_alive[current_band_id, member_id]:
                    continue
                if (
                    self.member_age[current_band_id, member_id]
                    < self.config.reproduction_adult_age
                ):
                    continue
                agent_ids.append(
                    self._agent_id(current_band_id, member_id)
                )
        return agent_ids

    def _cell_has_member(
        self,
        pos: tuple[int, int],
        ignore: tuple[int, int] | None = None,
    ) -> bool:
        y, x = pos
        for band_id in range(self.config.num_bands):
            for member_id in range(self.member_capacity_per_band):
                if not self.member_alive[band_id, member_id]:
                    continue
                if ignore == (band_id, member_id):
                    continue
                member_y, member_x = self._member_position(band_id, member_id)
                if (member_y, member_x) == (y, x):
                    return True
        return False

    def _find_band_member_at(
        self,
        band_id: int,
        pos: tuple[int, int],
    ) -> int | None:
        y, x = pos
        for m in range(self.member_capacity_per_band):
            if not self.member_alive[band_id, m]:
                continue
            my, mx = self._member_position(band_id, m)
            if my == y and mx == x:
                return m
        return None

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

        if self.config.river_crossing_width > 0:
            half = self.config.river_crossing_width // 2
            cx = int(self._patch_rng.integers(
                size // 4, 3 * size // 4
            ))
            x_lo = max(0, cx - half)
            x_hi = min(size, cx + half + 1)
            river_mask[:, x_lo:x_hi] = False

        water_mask = river_mask | lake_mask

        terrain = np.full(
            (size, size),
            Terrain.GRASSLAND,
            dtype=np.int8,
        )
        terrain[water_mask] = Terrain.WATER

        return terrain

    def _generate_grass(self) -> np.ndarray:
        grass = np.where(
            self.terrain == Terrain.GRASSLAND,
            self.config.grass_energy_capacity,
            0.0,
        )
        return grass.astype(np.float32)

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

    def _initialize_animals(self) -> None:
        self.animal_alive = np.zeros(self.config.max_animals, dtype=bool)
        self.animal_positions = np.zeros(
            (self.config.max_animals, 2),
            dtype=np.int16,
        )
        self.animal_individual_energy = np.zeros(
            self.config.max_animals,
            dtype=np.float32,
        )
        if self.config.max_animals == 0 or self.config.initial_animals == 0:
            self._sync_animal_energy_layer()
            return

        land_cells = np.argwhere(self.water <= 0.0)
        if len(land_cells) == 0:
            self._sync_animal_energy_layer()
            return
        animal_count = min(
            self.config.initial_animals, self.config.max_animals
        )
        chosen = self._patch_rng.choice(
            len(land_cells),
            size=animal_count,
            replace=len(land_cells) < animal_count,
        )
        self.animal_alive[:animal_count] = True
        self.animal_positions[:animal_count] = land_cells[chosen]
        self.animal_individual_energy[:animal_count] = min(
            self.config.max_animal_energy,
            self.config.initial_animal_energy,
        )
        self._sync_animal_energy_layer()

    def _sync_animal_energy_layer(self) -> None:
        self.animal_energy = np.zeros(
            (self.grid_size, self.grid_size),
            dtype=np.float32,
        )
        if self.config.max_animals == 0:
            return
        for animal_id in np.flatnonzero(self.animal_alive):
            y, x = self.animal_positions[int(animal_id)]
            energy = float(self.animal_individual_energy[int(animal_id)])
            self.animal_energy[int(y), int(x)] += energy

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
            blocker = self._find_band_member_at(band_id, (ny, nx))
            if (
                self._migration_target is not None
                and self._member_is_adult(band_id, member_id)
                and blocker is not None
                and not self._member_is_adult(band_id, blocker)
            ):
                self._set_member_position(band_id, blocker, (y, x))
                self._set_member_position(band_id, member_id, (ny, nx))
                terrain = Terrain(int(self.terrain[ny, nx]))
                self._spend_member_energy(
                    band_id,
                    member_id,
                    self._member_movement_cost(
                        band_id, member_id, terrain, (ny, nx)
                    ),
                )
                self._consume_cell_resources(band_id, member_id)
            else:
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
        terrain_cost = self._movement_cost_for_terrain(terrain)
        if self.config.use_physical_energetics:
            return float(
                self.config.movement_cost
                * self.config.walking_kcal_per_kg_km
                * self.config.cell_distance_km
                * self._member_total_move_mass_kg(band_id, member_id)
                * terrain_cost
            )
        moved_energy = (
            max(0.0, self._member_energy(band_id, member_id))
            + float(self.member_carried_food[band_id, member_id])
            + (
                self.config.water_load_factor
                * float(self.member_carried_water[band_id, member_id])
            )
        )
        distance_cost = 0.015 * self._distance_from_camp(target)
        return float(
            self.config.movement_cost
            * self.config.movement_energy_rate
            * moved_energy
            * terrain_cost
            + distance_cost
        )

    def _member_food_carry_room(
        self, band_id: int, member_id: int
    ) -> float:
        personal = max(
            0.0,
            self.config.food_carry_capacity
            - float(self.member_carried_food[band_id, member_id]),
        )
        if personal <= 0.0:
            return 0.0
        cap = self.config.camp_max_stored_food
        if cap > 0.0 and float(self.camp_stored_food[band_id]) >= cap:
            return 0.0
        return personal

    def _member_water_carry_room(
        self, band_id: int, member_id: int
    ) -> float:
        personal = max(
            0.0,
            self.config.water_carry_capacity
            - float(self.member_carried_water[band_id, member_id]),
        )
        if personal <= 0.0:
            return 0.0
        cap = self.config.camp_max_stored_water
        if cap > 0.0 and float(self.camp_stored_water[band_id]) >= cap:
            return 0.0
        return personal

    def _consume_cell_resources(self, band_id: int, member_id: int) -> None:
        if not self.member_alive[band_id, member_id]:
            return
        self._consume_plant_energy(band_id, member_id)
        self._consume_animal_energy(band_id, member_id)
        self._consume_water(band_id, member_id)

    def _member_sex(self, band_id: int, member_id: int) -> MemberSex:
        return MemberSex(int(self.member_sex[band_id, member_id]))

    def _has_nearby_male(self, band_id: int, member_id: int) -> bool:
        """Check if there's an alive male within male_proximity_distance."""
        if not self.config.sexual_reproduction_enabled:
            return True  # Always allow if sexual reproduction disabled

        member_y, member_x = self._member_position(band_id, member_id)

        for other_id in range(self.member_capacity_per_band):
            if not self.member_alive[band_id, other_id]:
                continue
            if self._member_sex(band_id, other_id) != MemberSex.MALE:
                continue

            other_y, other_x = self._member_position(band_id, other_id)
            manhattan_dist = abs(member_y - other_y) + abs(member_x - other_x)

            if manhattan_dist <= self.config.male_proximity_distance:
                return True

        return False

    def _member_can_gather_plants(
        self, band_id: int, member_id: int
    ) -> bool:
        # Hard sex gating is disabled to allow role specialization to emerge.
        return True

    def _member_can_hunt_animals(
        self, band_id: int, member_id: int
    ) -> bool:
        # Hard sex gating is disabled to allow role specialization to emerge.
        return True

    def _consume_plant_energy(self, band_id: int, member_id: int) -> None:
        if not self._member_can_gather_plants(band_id, member_id):
            return
        y, x = self._member_position(band_id, member_id)
        available_energy = float(self.plant_energy[y, x])
        body_need = max(
            0.0,
            min(
                self._member_personal_energy_reserve(band_id, member_id),
                self._member_max_energy(band_id, member_id),
            )
            - self._member_energy(band_id, member_id),
        )
        carry_room = self._member_food_carry_room(band_id, member_id)
        harvested_energy = min(
            available_energy,
            self.config.plant_eat_amount,
            body_need + carry_room,
        )
        # Eat first up to personal_energy_reserve; carry the surplus.
        eaten_energy = min(harvested_energy, body_need)
        carried_energy = min(
            (harvested_energy - eaten_energy) * self.config.plant_carry_share,
            carry_room,
        )
        self.plant_energy[y, x] = max(
            0.0,
            available_energy - harvested_energy,
        )
        self._update_depletion()
        self._record_food_gained(band_id, member_id, harvested_energy)
        self._add_member_energy(band_id, member_id, eaten_energy)
        self.member_carried_food[band_id, member_id] += carried_energy
        self.plant_eating_events += int(harvested_energy > 0.0)

    def _consume_animal_energy(self, band_id: int, member_id: int) -> None:
        if not self._member_can_hunt_animals(band_id, member_id):
            return
        y, x = self._member_position(band_id, member_id)
        animal_ids = [
            int(animal_id)
            for animal_id in np.flatnonzero(self.animal_alive)
            if tuple(
                int(v) for v in self.animal_positions[int(animal_id)]
            ) == (y, x)
        ]
        available_energy = float(
            np.sum(self.animal_individual_energy[animal_ids])
        )
        body_need = max(
            0.0,
            min(
                self._member_personal_energy_reserve(band_id, member_id),
                self._member_max_energy(band_id, member_id),
            )
            - self._member_energy(band_id, member_id),
        )
        carry_room = self._member_food_carry_room(band_id, member_id)
        harvested_energy = min(
            available_energy,
            self.config.animal_hunt_amount,
            body_need + carry_room,
        )
        # Eat first up to personal_energy_reserve; carry the surplus.
        eaten_energy = min(harvested_energy, body_need)
        carried_energy = min(
            (harvested_energy - eaten_energy) * self.config.animal_carry_share,
            carry_room,
        )
        self._remove_hunted_animal_energy(animal_ids, harvested_energy)
        self._sync_animal_energy_layer()
        self._record_animal_food_gained(band_id, member_id, harvested_energy)
        self._add_member_energy(band_id, member_id, eaten_energy)
        self.member_carried_food[band_id, member_id] += carried_energy

    def _remove_hunted_animal_energy(
        self,
        animal_ids: list[int],
        harvested_energy: float,
    ) -> None:
        remaining = harvested_energy
        for animal_id in animal_ids:
            if remaining <= 0.0:
                return
            taken = min(
                remaining,
                float(self.animal_individual_energy[animal_id]),
            )
            self.animal_individual_energy[animal_id] -= taken
            remaining -= taken
            if self.animal_individual_energy[animal_id] <= 0.0:
                self.animal_alive[animal_id] = False

    def _consume_water(self, band_id: int, member_id: int) -> None:
        y, x = self._member_position(band_id, member_id)
        water_gained = 0.0
        if self.water[y, x] > 0.0:
            hydration_need = max(
                0.0,
                self._member_max_hydration(band_id, member_id)
                - self._member_hydration(band_id, member_id),
            )
            drunk_water = min(self.config.drink_amount, hydration_need)
            carry_room = self._member_water_carry_room(band_id, member_id)
            carried_water = min(self.config.water_collect_amount, carry_room)
            self._add_member_hydration(band_id, member_id, drunk_water)
            self.member_carried_water[band_id, member_id] += carried_water
            water_gained = drunk_water + carried_water
        self._record_water_gained(band_id, member_id, water_gained)
        self.water_drinking_events += int(water_gained > 0.0)

    def _camp_too_far_from_water(self, camp: tuple[int, int]) -> bool:
        """Return True if the camp exceeds camp_max_water_distance."""
        max_dist = self.config.camp_max_water_distance
        if max_dist <= 0:
            return False
        water_dist = self._distance_to_nearest_water()[camp]
        return float(water_dist) > max_dist

    def _band_relocation_food_cost(self, band_id: int, distance: int) -> float:
        return sum(
            self._member_relocation_cost(band_id, member_id, distance)
            for member_id in range(self.member_capacity_per_band)
            if self.member_alive[band_id, member_id]
        )

    def _maybe_relocate_depleted_camp(self) -> None:
        if self._migration_target is not None:
            return
        self._update_depletion()
        current_depletion = self._local_camp_depletion(self.camp_pos)
        self.local_depletion_level = current_depletion
        too_dry = self._camp_too_far_from_water(self.camp_pos)
        if (
            current_depletion < self.config.camp_depletion_threshold
            and not too_dry
        ):
            return

        new_camp = self._best_camp_location()
        if new_camp == self.camp_pos:
            return
        if (
            self._local_camp_depletion(new_camp)
            >= self.config.camp_depletion_threshold
            or self._camp_too_far_from_water(new_camp)
        ):
            return

        emergency = (
            current_depletion
            >= self.config.camp_emergency_depletion_threshold
        )
        if not emergency and self.config.camp_store_departure_steps > 0:
            alive = int(np.sum(self.member_alive[0]))
            stored_food = float(self.camp_stored_food[0])
            stored_water = float(self.camp_stored_water[0])
            food_buffer = (
                alive * self.config.camp_food_withdraw_amount
                * self.config.camp_store_departure_steps
            )
            water_buffer = (
                alive * self.config.thirst_per_step
                * self.config.camp_store_departure_steps
            )
            if stored_food > food_buffer and stored_water > water_buffer:
                return

        self._start_migration(new_camp)

    def _start_migration(self, new_camp: tuple[int, int]) -> None:
        band_id = 0
        alive = np.flatnonzero(self.member_alive[band_id])
        ages = self.member_age[band_id, alive]
        n_adults = int(np.sum(ages >= self.config.reproduction_adult_age))
        n_juveniles = int(np.sum(ages < self.config.reproduction_adult_age))
        adults_needed = math.ceil(n_juveniles / 2) if n_juveniles > 0 else 0
        surplus = max(0, n_adults - adults_needed)
        self._migration_target = new_camp
        self._migration_start_pos = self.camp_pos
        self._migration_carries_stores = surplus > 0
        if not self._migration_carries_stores:
            self.camp_stored_food[band_id] = 0.0
            self.camp_stored_water[band_id] = 0.0
            self._sync_stored_food_alias()

    def _place_migration_members(
        self,
        band_id: int,
        camp: tuple[int, int],
        adults: list[int],
        juveniles: list[int],
    ) -> None:
        """Juveniles fill cells nearest camp; adults fill the outer ring."""
        occupied = {
            self._member_position(other_id, m)
            for other_id in range(self.config.num_bands)
            if other_id != band_id
            for m in range(self.member_capacity_per_band)
            if self.member_alive[other_id, m]
        }
        total = len(juveniles) + len(adults)
        all_cells = self._member_cells_near(camp, occupied, needed=total)
        for member_id, cell in zip(
            juveniles + adults, all_cells
        ):
            self.member_positions[band_id, member_id] = cell

    def _migration_carry_mass_cost(
        self,
        band_id: int,
        carried_id: int,
        terrain: Terrain,
    ) -> float:
        """Energy cost to carry carried_id one cell (based on carried mass)."""
        terrain_cost = self._movement_cost_for_terrain(terrain)
        if self.config.use_physical_energetics:
            return float(
                self.config.movement_cost
                * self.config.walking_kcal_per_kg_km
                * self.config.cell_distance_km
                * float(self.member_body_mass_kg[band_id, carried_id])
                * terrain_cost
            )
        juvenile_energy = max(
            0.0, float(self.member_energy[band_id, carried_id])
        )
        return float(
            self.config.movement_cost
            * self.config.movement_energy_rate
            * juvenile_energy
            * terrain_cost
        )

    def _migration_stores_carry_cost(
        self,
        stored_food: float,
        stored_water: float,
        n_surplus: int,
        terrain: Terrain,
    ) -> float:
        """Energy cost for one surplus adult carrying their share one cell."""
        terrain_cost = self._movement_cost_for_terrain(terrain)
        if self.config.use_physical_energetics:
            food_kg = (
                (stored_food / n_surplus)
                / self.config.food_energy_density_kcal_per_kg
            )
            water_kg = (
                (stored_water / n_surplus)
                * self.config.water_liters_per_unit
            )
            return float(
                self.config.movement_cost
                * self.config.walking_kcal_per_kg_km
                * self.config.cell_distance_km
                * (food_kg + water_kg)
                * terrain_cost
            )
        return float(
            self.config.movement_cost
            * self.config.movement_energy_rate
            * (stored_food / n_surplus)
            * terrain_cost
        )

    def _step_migration(self) -> None:
        if self._migration_target is None:
            return
        band_id = 0
        cy, cx = self.camp_pos
        ty, tx = self._migration_target
        if cy == ty and cx == tx:
            self._complete_migration()
            return

        # Advance camp one Manhattan step toward target.
        dy = ty - cy
        dx = tx - cx
        if abs(dy) >= abs(dx):
            ny, nx = cy + (1 if dy > 0 else -1), cx
        else:
            ny, nx = cy, cx + (1 if dx > 0 else -1)
        new_pos = (ny, nx)

        self.camp_pos = new_pos
        self.band_camp_positions[band_id] = new_pos
        self.macro_x += nx - cx
        self.macro_y += ny - cy
        self.macro_distance_traveled += 1

        # Charge carry costs: adults carry juveniles (≤2 each) and stores.
        terrain = Terrain(int(self.terrain[ny, nx]))
        alive_slots = [
            m for m in range(self.member_capacity_per_band)
            if self.member_alive[band_id, m]
        ]
        adult_age = self.config.reproduction_adult_age
        adults = [
            m for m in alive_slots
            if self.member_age[band_id, m] >= adult_age
        ]
        juveniles = [
            m for m in alive_slots
            if self.member_age[band_id, m] < adult_age
        ]
        adults_needed = math.ceil(len(juveniles) / 2) if juveniles else 0
        surplus_adults = set(adults[adults_needed:])

        # Assign up to 2 juveniles to each carrying adult.
        jv_queue = list(juveniles)
        juvenile_assignments: dict[int, list[int]] = {}
        for adult in adults[:adults_needed]:
            assigned = jv_queue[:2]
            jv_queue = jv_queue[2:]
            juvenile_assignments[adult] = assigned
        for adult in adults[adults_needed:]:
            juvenile_assignments[adult] = []

        n_surplus = len(surplus_adults)
        stored_food = float(self.camp_stored_food[band_id])
        stored_water = float(self.camp_stored_water[band_id])

        for adult in adults:
            # Adult's own movement cost (body mass + what they already carry).
            cost = self._member_movement_cost(band_id, adult, terrain, new_pos)
            # Extra cost per juvenile carried, based on juvenile's own mass.
            for jv in juvenile_assignments[adult]:
                cost += self._migration_carry_mass_cost(
                    band_id, jv, terrain
                )
            # Surplus adults carry an equal share of camp stores.
            carries = (
                adult in surplus_adults
                and self._migration_carries_stores
                and n_surplus > 0
            )
            if carries:
                cost += self._migration_stores_carry_cost(
                    stored_food, stored_water, n_surplus, terrain
                )
            self._spend_member_energy(band_id, adult, cost)

        # Juveniles fill the inner ring; adults fill the outer ring.
        self._place_migration_members(band_id, new_pos, adults, juveniles)

    def _complete_migration(self) -> None:
        self._prev_camp_pos = self._migration_start_pos
        self.camp_id += 1
        self.camp_age = 0
        self.num_camp_moves += 1
        self._last_camp_relocated = True
        self._migration_target = None
        self._migration_start_pos = None
        self._migration_carries_stores = False
        self.local_depletion_level = self._local_camp_depletion(self.camp_pos)

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
        prev = self._prev_camp_pos
        water_distance = self._distance_to_nearest_water()
        max_water_dist = self.config.camp_max_water_distance
        best_key: tuple[float, float, int, int, int] | None = None
        best_camp = current
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.terrain[y, x] == Terrain.WATER:
                    continue
                candidate = (y, x)
                # Skip candidates too far from water.
                if (
                    max_water_dist > 0
                    and float(water_distance[candidate]) > max_water_dist
                ):
                    continue
                travel_distance = self._camp_distance(current, candidate)
                if self._camp_foraging_areas_overlap(current, candidate):
                    continue
                # Exclude the previous camp site to prevent
                # ping-pong oscillation.
                if prev is not None and self._camp_foraging_areas_overlap(
                    prev, candidate
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
        self._prev_camp_pos = self.camp_pos
        self.camp_pos = new_camp
        self.band_camp_positions[0] = new_camp
        self._last_camp_relocated = True
        self._spend_band_relocation_energy(0, travel_distance)
        self._place_band_members_near(0, new_camp)

    def _spend_band_relocation_energy(
        self,
        band_id: int,
        camp_distance: int,
    ) -> None:
        if self.config.camp_relocation_cost <= 0.0 or camp_distance <= 0:
            return
        for member_id in range(self.member_capacity_per_band):
            if self.member_alive[band_id, member_id]:
                self._spend_member_energy(
                    band_id,
                    member_id,
                    self._member_relocation_cost(
                        band_id,
                        member_id,
                        camp_distance,
                    ),
                )

    def _member_relocation_cost(
        self,
        band_id: int,
        member_id: int,
        camp_distance: int,
    ) -> float:
        if self.config.use_physical_energetics:
            return float(
                self.config.camp_relocation_cost
                * camp_distance
                * self.config.cell_distance_km
                * self.config.walking_kcal_per_kg_km
                * self._member_total_move_mass_kg(band_id, member_id)
            )
        moved_energy = (
            max(0.0, self._member_energy(band_id, member_id))
            + float(self.member_carried_food[band_id, member_id])
            + (
                self.config.water_load_factor
                * float(self.member_carried_water[band_id, member_id])
            )
        )
        return float(
            self.config.camp_relocation_cost
            * camp_distance
            * self.config.movement_energy_rate
            * moved_energy
        )

    def _place_band_members_near(
        self,
        band_id: int,
        camp: tuple[int, int],
    ) -> None:
        occupied = {
            self._member_position(other_band_id, member_id)
            for other_band_id in range(self.config.num_bands)
            if other_band_id != band_id
            for member_id in range(self.member_capacity_per_band)
            if self.member_alive[other_band_id, member_id]
        }
        alive_slots = [
            member_id
            for member_id in range(self.member_capacity_per_band)
            if self.member_alive[band_id, member_id]
        ]
        members = self._member_cells_near(
            camp,
            occupied,
            needed=len(alive_slots),
        )
        for member_id, member_pos in zip(alive_slots, members):
            self.member_positions[band_id, member_id] = member_pos

    def _spend_member_energy(
        self,
        band_id: int,
        member_id: int,
        amount: float,
    ) -> None:
        amount = max(0.0, float(amount))
        self.member_energy[band_id, member_id] -= amount
        self.member_last_energy_spent[band_id, member_id] += amount

    def _spend_member_hydration(
        self,
        band_id: int,
        member_id: int,
        amount: float,
    ) -> None:
        amount = max(0.0, float(amount))
        self.member_hydration[band_id, member_id] -= amount
        self.member_last_hydration_spent[band_id, member_id] += amount

    def _apply_thirst_to_alive_members(self) -> None:
        for band_id in range(self.config.num_bands):
            for member_id in range(self.member_capacity_per_band):
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
        # Track plants gathered by sex
        member_sex = self._member_sex(band_id, member_id)
        if member_sex == MemberSex.FEMALE:
            self.plants_gathered_by_females += amount
        else:
            self.plants_gathered_by_males += amount

    def _record_animal_food_gained(
        self,
        band_id: int,
        member_id: int,
        amount: float,
    ) -> None:
        amount = max(0.0, float(amount))
        self.member_last_food_gained[band_id, member_id] += amount
        self.member_last_animal_food_gained[band_id, member_id] += amount
        # Track animals hunted by sex
        member_sex = self._member_sex(band_id, member_id)
        if member_sex == MemberSex.FEMALE:
            self.animals_hunted_by_females += amount
        else:
            self.animals_hunted_by_males += amount

    def _record_water_gained(
        self,
        band_id: int,
        member_id: int,
        amount: float,
    ) -> None:
        amount = max(0.0, float(amount))
        self.member_last_water_gained[band_id, member_id] += amount

    def _record_food_deposited(
        self,
        band_id: int,
        member_id: int,
        amount: float,
    ) -> None:
        amount = max(0.0, float(amount))
        self.member_last_food_deposited[band_id, member_id] += amount

    def _record_water_deposited(
        self,
        band_id: int,
        member_id: int,
        amount: float,
    ) -> None:
        amount = max(0.0, float(amount))
        self.member_last_water_deposited[band_id, member_id] += amount

    def _record_food_withdrawn(
        self,
        band_id: int,
        member_id: int,
        amount: float,
    ) -> None:
        amount = max(0.0, float(amount))
        self.member_last_food_withdrawn[band_id, member_id] += amount

    def _record_water_withdrawn(
        self,
        band_id: int,
        member_id: int,
        amount: float,
    ) -> None:
        amount = max(0.0, float(amount))
        self.member_last_water_withdrawn[band_id, member_id] += amount

    def _print_division_of_labor_stats(self) -> None:
        """Print per-sex diet breakdown (plants% vs animals%)."""
        female_total = (
            self.plants_gathered_by_females + self.animals_hunted_by_females
        )
        male_total = (
            self.plants_gathered_by_males + self.animals_hunted_by_males
        )
        female_plant_pct = (
            (self.plants_gathered_by_females / female_total * 100)
            if female_total > 0
            else 0.0
        )
        female_animal_pct = 100.0 - female_plant_pct
        male_plant_pct = (
            (self.plants_gathered_by_males / male_total * 100)
            if male_total > 0
            else 0.0
        )
        male_animal_pct = 100.0 - male_plant_pct
        print(
            f"[Step {self.step_count}] Division of Labor — "
            f"females: {female_plant_pct:.1f}% plants / "
            f"{female_animal_pct:.1f}% animals | "
            f"males: {male_plant_pct:.1f}% plants / "
            f"{male_animal_pct:.1f}% animals"
        )

    def _sync_stored_food_alias(self) -> None:
        self.stored_food = (
            float(self.camp_stored_food[0])
            if len(self.camp_stored_food) > 0
            else 0.0
        )

    def _member_can_access_band_storage(
        self,
        band_id: int,
        member_id: int,
    ) -> bool:
        return (
            self._camp_distance(
                self._member_position(band_id, member_id),
                tuple(
                    int(v) for v in self.band_camp_positions[band_id]
                ),
            )
            <= self.config.camp_storage_radius
        )

    def _exchange_camp_resources_for_alive_members(self) -> None:
        for band_id in range(self.config.num_bands):
            for member_id in range(self.member_capacity_per_band):
                if not self.member_alive[band_id, member_id]:
                    continue
                if not self._member_can_access_band_storage(
                    band_id, member_id
                ):
                    continue
                self._deposit_member_resources_at_camp(band_id, member_id)

            for member_id in range(self.member_capacity_per_band):
                if not self.member_alive[band_id, member_id]:
                    continue
                self._consume_carried_resources(band_id, member_id)

            for member_id in range(self.member_capacity_per_band):
                if not self.member_alive[band_id, member_id]:
                    continue
                if not self._member_can_access_band_storage(
                    band_id, member_id
                ):
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
            current_food = float(self.camp_stored_food[band_id])
            cap_food = self.config.camp_max_stored_food
            room_food = (
                max(0.0, cap_food - current_food)
                if cap_food > 0.0
                else carried_food
            )
            deposited_food = min(carried_food, room_food)
            self.camp_stored_food[band_id] = current_food + deposited_food
            self.member_carried_food[band_id, member_id] -= deposited_food
            self._record_food_deposited(band_id, member_id, deposited_food)
        if carried_water > 0.0:
            current_water = float(self.camp_stored_water[band_id])
            cap_water = self.config.camp_max_stored_water
            room_water = (
                max(0.0, cap_water - current_water)
                if cap_water > 0.0
                else carried_water
            )
            deposited_water = min(carried_water, room_water)
            self.camp_stored_water[band_id] = current_water + deposited_water
            self.member_carried_water[band_id, member_id] -= deposited_water
            self._record_water_deposited(
                band_id, member_id, deposited_water
            )

    def _consume_carried_resources(
        self,
        band_id: int,
        member_id: int,
    ) -> None:
        carried_food = float(self.member_carried_food[band_id, member_id])
        if carried_food > 0.0:
            energy_need = max(
                0.0,
                min(
                    self.config.carried_food_consume_threshold,
                    self._member_max_energy(band_id, member_id),
                )
                - self._member_energy(band_id, member_id),
            )
            eaten = min(energy_need, carried_food)
            if eaten > 0.0:
                self.member_carried_food[band_id, member_id] -= eaten
                self._add_member_energy(band_id, member_id, eaten)

        carried_water = float(self.member_carried_water[band_id, member_id])
        if carried_water > 0.0:
            hydration_need = max(
                0.0,
                min(
                    self.config.carried_water_consume_threshold,
                    self._member_max_hydration(band_id, member_id),
                )
                - self._member_hydration(band_id, member_id),
            )
            drunk = min(hydration_need, carried_water)
            if drunk > 0.0:
                self.member_carried_water[band_id, member_id] -= drunk
                self._add_member_hydration(band_id, member_id, drunk)

    def _withdraw_member_resources_from_camp(
        self,
        band_id: int,
        member_id: int,
    ) -> None:
        energy_need = max(
            0.0,
            min(
                self.config.camp_food_withdraw_threshold,
                self._member_max_energy(band_id, member_id),
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
            self._record_food_withdrawn(band_id, member_id, food_taken)

        hydration_need = max(
            0.0,
            min(
                self.config.camp_water_withdraw_threshold,
                self._member_max_hydration(band_id, member_id),
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
            self._record_water_withdrawn(band_id, member_id, water_taken)

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
        grass_capacity = np.where(
            self.terrain == Terrain.GRASSLAND,
            self.config.grass_energy_capacity,
            0.0,
        ).astype(np.float32)
        self.grass_energy = np.clip(
            self.grass_energy
            + self.config.grass_regrowth_base * season_mod,
            0.0,
            grass_capacity,
        ).astype(np.float32)
        self.grass_energy[self.water > 0.0] = 0.0
        self._update_depletion()
        if self.config.camp_food_decay_rate > 0.0:
            self.camp_stored_food = np.maximum(
                0.0,
                self.camp_stored_food
                * (1.0 - self.config.camp_food_decay_rate),
            )
            self._sync_stored_food_alias()

    def _update_animals(self) -> None:
        if self.config.max_animals == 0:
            self._sync_animal_energy_layer()
            return
        active_ids = list(np.flatnonzero(self.animal_alive))
        self._move_animals(active_ids)
        self._feed_animals(active_ids)
        self._reproduce_animals(active_ids)
        self.animal_alive &= self.animal_individual_energy > 0.0
        self._sync_animal_energy_layer()

    def _move_animals(self, animal_ids: list[int]) -> None:
        directions = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        for animal_id in animal_ids:
            if not self.animal_alive[animal_id]:
                continue
            idx = int(self._episode_rng.integers(len(directions)))
            dy, dx = directions[idx]
            y, x = self.animal_positions[animal_id]
            ny = int(np.clip(int(y) + dy, 0, self.grid_size - 1))
            nx = int(np.clip(int(x) + dx, 0, self.grid_size - 1))
            if self.water[ny, nx] > 0.0:
                ny, nx = int(y), int(x)
            self.animal_positions[animal_id] = (ny, nx)
            if dy != 0 or dx != 0:
                self.animal_individual_energy[animal_id] -= (
                    self.config.animal_move_cost
                    * float(SEASON_COST_MOD[self.season])
                )

    def _feed_animals(self, animal_ids: list[int]) -> None:
        for animal_id in animal_ids:
            if not self.animal_alive[animal_id]:
                continue
            y, x = self.animal_positions[animal_id]
            eaten = min(
                float(self.grass_energy[int(y), int(x)]),
                self.config.animal_eat_amount,
                self.config.max_animal_energy
                - float(self.animal_individual_energy[animal_id]),
            )
            if eaten <= 0.0:
                continue
            self.grass_energy[int(y), int(x)] -= eaten
            self.animal_individual_energy[animal_id] += eaten

    def _reproduce_animals(self, animal_ids: list[int]) -> None:
        empty_slots = list(np.flatnonzero(~self.animal_alive))
        if not empty_slots:
            return
        for animal_id in animal_ids:
            if not empty_slots or not self.animal_alive[animal_id]:
                return
            if (
                self.animal_individual_energy[animal_id]
                < self.config.animal_reproduction_energy_threshold
            ):
                continue
            if (
                self.animal_individual_energy[animal_id]
                < self.config.animal_reproduction_cost
            ):
                continue
            parent_pos = tuple(
                int(v) for v in self.animal_positions[animal_id]
            )
            child_pos = self._animal_birth_position(parent_pos)
            if child_pos is None:
                continue
            child_id = int(empty_slots.pop(0))
            self.animal_individual_energy[animal_id] -= (
                self.config.animal_reproduction_cost
            )
            self.animal_alive[child_id] = True
            self.animal_positions[child_id] = child_pos
            self.animal_individual_energy[child_id] = min(
                self.config.max_animal_energy,
                self.config.newborn_animal_energy,
            )

    def _animal_birth_position(
        self,
        parent_pos: tuple[int, int],
    ) -> tuple[int, int] | None:
        y, x = parent_pos
        candidates = [(y, x), (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
        self._episode_rng.shuffle(candidates)
        for cy, cx in candidates:
            if not (0 <= cy < self.grid_size and 0 <= cx < self.grid_size):
                continue
            if self.water[cy, cx] > 0.0:
                continue
            return int(cy), int(cx)
        return None

    def _plant_energy_observation_layer(self) -> np.ndarray:
        capacity = max(1.0, float(self.config.plant_energy_capacity))
        return np.clip(
            self.plant_energy / capacity,
            0.0,
            1.0,
        ).astype(np.float32)

    def _animal_energy_observation_layer(self) -> np.ndarray:
        capacity = max(1.0, float(self.config.animal_energy_capacity))
        return np.clip(
            self.animal_energy / capacity,
            0.0,
            1.0,
        ).astype(np.float32)

    def _build_observation(self) -> np.ndarray:
        if not self.member_alive[0, 0]:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
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
            self._animal_energy_observation_layer(),
            self.water,
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
                / self._member_max_energy(band_id, member_id),
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
                np.zeros(
                    (self.config.obs_range, self.config.obs_range),
                    dtype=np.float32,
                ),
            ]
        )
        return np.stack(padded_channels, axis=0).astype(np.float32)

    def _camp_layer(self) -> np.ndarray:
        layer = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        gy, gx = np.mgrid[0:self.grid_size, 0:self.grid_size]
        radius = self.config.camp_depletion_radius
        for camp_y, camp_x in self.band_camp_positions:
            dist = np.abs(gy - int(camp_y)) + np.abs(gx - int(camp_x))
            strength = np.clip(1.0 - dist / (radius + 1), 0.0, 1.0)
            layer = np.maximum(layer, strength)
        return layer

    def _agent_layer(
        self,
        focal_member: tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        layer = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for band_id in range(self.config.num_bands):
            value = 0.75 if band_id == 0 else 0.5
            for member_id in range(self.member_capacity_per_band):
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
        animal_food = float(
            self.member_last_animal_food_gained[band_id, member_id]
        )
        plant_food = float(
            self.member_last_food_gained[band_id, member_id]
        ) - animal_food
        reward = (
            self.config.plant_food_reward * plant_food
            + self.config.animal_food_reward * animal_food
        )
        reward += self.config.water_collect_reward * float(
            self.member_last_water_gained[band_id, member_id]
        )
        reward += self.config.food_deposit_reward * float(
            self.member_last_food_deposited[band_id, member_id]
        )
        reward += self.config.water_deposit_reward * float(
            self.member_last_water_deposited[band_id, member_id]
        )
        reward += self.config.camp_food_withdraw_reward * float(
            self.member_last_food_withdrawn[band_id, member_id]
        )
        reward += self.config.camp_water_withdraw_reward * float(
            self.member_last_water_withdrawn[band_id, member_id]
        )
        reward += self.config.birth_reward * float(
            self._last_birth_events[band_id]
        )
        reward += self.config.population_reward * float(
            np.count_nonzero(self.member_alive[band_id])
        )
        juvenile_count = np.count_nonzero(
            self.member_alive[band_id]
            & (self.member_age[band_id] < self.config.reproduction_adult_age)
        )
        reward += self.config.juvenile_survival_reward * float(juvenile_count)
        reward -= 0.01 * float(
            self.member_last_energy_spent[band_id, member_id]
        )
        reward += 0.01
        if terminated:
            reward -= 1.0
        return float(reward)

    def _camp_pressure(self) -> float:
        initial_energy = self._initial_adult_energy(self.config.initial_energy)
        food_shortage = 1.0 - np.clip(
            self._member_energy(0, 0) / max(1.0, initial_energy),
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
            "camp_pos": self.camp_pos,
            "band_camp_positions": self.band_camp_positions.tolist(),
            "member_positions": self.member_positions.tolist(),
            "member_ids": self.member_ids.tolist(),
            "member_alive": self.member_alive.tolist(),
            "member_age": self.member_age.tolist(),
            "member_energy": self.member_energy.tolist(),
            "member_body_mass_kg": self.member_body_mass_kg.tolist(),
            "member_hydration": self.member_hydration.tolist(),
            "member_carried_food": self.member_carried_food.tolist(),
            "member_carried_water": self.member_carried_water.tolist(),
            "active_agent_ids": self.active_agent_ids(),
            "num_bands": self.config.num_bands,
            "members_per_band": self.config.members_per_band,
            "max_members_per_band": self.member_capacity_per_band,
            "stored_food": self.stored_food,
            "camp_stored_food": self.camp_stored_food.tolist(),
            "camp_stored_water": self.camp_stored_water.tolist(),
            "population": self.population,
            "max_population": (
                self.config.num_bands * self.member_capacity_per_band
            ),
            "local_depletion_level": self.local_depletion_level,
            "camp_potential_energy": self.camp_potential_energy(),
            "mean_plant_energy": float(np.mean(self.plant_energy)),
            "mean_plant_food": float(np.mean(self.plant_energy)),
            "mean_grass_energy": float(np.mean(self.grass_energy)),
            "mean_animal_energy": float(np.mean(self.animal_energy)),
            "mean_depletion": float(np.mean(self.depletion)),
            "num_camp_moves": self.num_camp_moves,
            "camp_relocated": self._last_camp_relocated,
            "macro_distance_traveled": self.macro_distance_traveled,
            "starvation_events": self.starvation_events,
            "dehydration_events": self.dehydration_events,
            "old_age_events": self.old_age_events,
            "birth_events": self.birth_events,
            "last_birth_events": self._last_birth_events.tolist(),
            "plant_eating_events": self.plant_eating_events,
            "water_drinking_events": self.water_drinking_events,
            "member_last_food_gained": (
                self.member_last_food_gained.tolist()
            ),
            "member_last_animal_food_gained": (
                self.member_last_animal_food_gained.tolist()
            ),
            "member_last_water_gained": (
                self.member_last_water_gained.tolist()
            ),
            "member_last_food_deposited": (
                self.member_last_food_deposited.tolist()
            ),
            "member_last_water_deposited": (
                self.member_last_water_deposited.tolist()
            ),
            "member_last_energy_spent": (
                self.member_last_energy_spent.tolist()
            ),
            "member_last_hydration_spent": (
                self.member_last_hydration_spent.tolist()
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
        # Capture (agent_id, slot) and member identity NOW — identity may
        # change during this step when a slot is vacated by death and
        # immediately filled by a newborn (same slot, new member_id).
        # After _refresh_agent_ids the original agent_id is stale.
        band = self.controlled_band_id
        # Restrict to adult slots only — juveniles never receive observations
        # so including them in rewards/terminations/truncations causes RLlib's
        # episode tracker to accumulate stale state that leads to KeyErrors at
        # training iteration boundaries (cut episodes).
        agent_slots: list[tuple[str, int]] = [
            (self._agent_id(band, slot), slot)
            for slot in active_slots
            if (
                self.member_age[band, slot]
                >= self.config.reproduction_adult_age
            )
        ]
        pre_sim_identity: dict[int, int] = {
            slot: int(self.member_ids[band, slot])
            for _, slot in agent_slots
        }
        agent_ids = [agent_id for agent_id, _ in agent_slots]

        self._begin_multi_agent_step(active_slots)
        actions = self._member_actions(action_dict, active_slots)
        self._apply_member_moves(actions)
        self._apply_member_non_move_actions(actions)
        self._exchange_camp_resources_for_alive_members()
        self._update_resources()
        self._update_animals()
        self._maybe_relocate_depleted_camp()
        self._step_migration()

        self._apply_thirst_to_alive_members()
        self._advance_member_lifecycle()
        (
            starvation_events,
            dehydration_events,
            old_age_events,
        ) = self._update_member_survival()
        self._cull_excess_juveniles()
        self.starvation_events += starvation_events
        self.dehydration_events += dehydration_events
        self.old_age_events += old_age_events
        self._maybe_reproduce_bands()
        self._last_camp_pressure = self._camp_pressure()

        # Print division of labor stats every 100 steps or at episode end
        all_alive = len(self._active_controlled_slots()) > 0
        if (
            self.step_count % 100 == 0
            or self.step_count >= self.config.max_steps
            or not all_alive
        ):
            self._print_division_of_labor_stats()

        truncated = self.step_count >= self.config.max_steps
        self._refresh_agent_ids()
        all_terminated = len(self.agents) == 0

        # Use slot-based lookups — agent_ids may no longer be in
        # _agent_to_member if their slot was reused by a newborn.
        # A slot is "terminated" for the original agent when either:
        #   (a) the slot is no longer alive (member died, not reused), or
        #   (b) the member identity in the slot changed (died AND slot
        #       was immediately filled by a newborn in the same step).
        # Without the identity check, member_alive[slot] would be True for
        # the newborn, making terminated=False for the now-gone original
        # agent — causing a KeyError in RLlib's episode tracker.
        def _agent_terminated(slot: int) -> bool:
            return (
                not bool(self.member_alive[band, slot])
                or int(self.member_ids[band, slot]) != pre_sim_identity[slot]
            )

        rewards = {
            agent_id: self._calculate_member_reward(
                band,
                slot,
                terminated=_agent_terminated(slot),
            )
            for agent_id, slot in agent_slots
        }
        newborn_agent_ids = [
            agent_id
            for agent_id in self.agents
            if agent_id not in set(agent_ids)
        ]
        rewards.update({agent_id: 0.0 for agent_id in newborn_agent_ids})
        terminations = {
            agent_id: _agent_terminated(slot)
            for agent_id, slot in agent_slots
        }
        terminations.update(
            {agent_id: False for agent_id in newborn_agent_ids}
        )
        terminations["__all__"] = all_terminated
        # Terminated agents must NOT also be marked truncated — they carry no
        # final observation, and RLlib requires one for any truncated agent
        # (for value-function bootstrapping). Alive agents get truncated=True
        # when the episode hits max_steps.
        truncations = {
            agent_id: (truncated and not _agent_terminated(slot))
            for agent_id, slot in agent_slots
        }
        truncations.update(
            {agent_id: truncated for agent_id in newborn_agent_ids}
        )
        truncations["__all__"] = truncated
        infos = {
            agent_id: self._build_agent_info(band, slot)
            for agent_id, slot in agent_slots
        }
        infos.update(self._multi_agent_infos(newborn_agent_ids))

        if truncated and not all_terminated:
            observations = self._multi_agent_observations()
            self.agents = []
        elif all_terminated:
            self.agents = []
            observations: dict[str, np.ndarray] = {}
        else:
            observations = self._multi_agent_observations()

        # RLlib sets `is_truncated = True` on the episode object BEFORE its
        # per-agent loop (from `truncateds["__all__"]`). This propagates to
        # every agent via `_truncated = individual_flag or is_truncated`, so
        # agents that terminated in this same step also get `_truncated=True`.
        # Without an observation that triggers CASE 3 ("truncated, no obs") →
        # error. Fix: supply a placeholder terminal observation for any agent
        # that died in the truncation step. The value is never used for
        # learning (RLlib's own comment on Case 3 terminated observations).
        if truncated:
            for agent_id, slot in agent_slots:
                if (
                    _agent_terminated(slot)
                    and agent_id not in observations
                ):
                    # Placeholder: shape must match the obs space but the
                    # value is never used for learning (RLlib treats this
                    # the same way it treats terminal obs in Case 3).
                    # Calling _build_member_observation would crash because
                    # member_body_mass_kg is zeroed out on death.
                    observations[agent_id] = np.zeros(
                        self.observation_space.shape, dtype=np.float32
                    )

        return observations, rewards, terminations, truncations, infos

    def _refresh_agent_ids(self) -> None:
        if self.controlled_band_id < 0:
            raise ValueError("controlled_band_id must be non-negative")
        if self.controlled_band_id >= self.config.num_bands:
            raise ValueError("controlled_band_id must be less than num_bands")

        self.possible_agents = [
            self._agent_id(self.controlled_band_id, member_id)
            for member_id in range(self.member_capacity_per_band)
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
            for member_id in range(self.member_capacity_per_band)
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
            if not self._member_is_adult(self.controlled_band_id, member_id):
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
        occupied = self._occupied_member_positions()
        order = list(proposals.keys())
        self._episode_rng.shuffle(order)
        for member_id in order:
            target = proposals[member_id]
            occupied_by = occupied.get(target)
            blocked = (
                occupied_by is not None
                and occupied_by != (self.controlled_band_id, member_id)
            )
            if blocked:
                bid = self.controlled_band_id
                can_swap = (
                    self._migration_target is not None
                    and self._member_is_adult(bid, member_id)
                    and occupied_by[0] == bid
                    and not self._member_is_adult(bid, occupied_by[1])
                )
                if can_swap:
                    blocker_id = occupied_by[1]
                    old_pos = self._member_position(bid, member_id)
                    occupied.pop(old_pos, None)
                    occupied[old_pos] = (bid, blocker_id)
                    occupied[target] = (bid, member_id)
                    self._set_member_position(bid, blocker_id, old_pos)
                    self._move_member_to_target(member_id, target)
                else:
                    self._spend_member_energy(
                        self.controlled_band_id, member_id, 0.15
                    )
                continue
            occupied.pop(
                self._member_position(self.controlled_band_id, member_id),
                None,
            )
            occupied[target] = (self.controlled_band_id, member_id)
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

    def _occupied_member_positions(
        self,
    ) -> dict[tuple[int, int], tuple[int, int]]:
        occupied: dict[tuple[int, int], tuple[int, int]] = {}
        for band_id in range(self.config.num_bands):
            for member_id in range(self.member_capacity_per_band):
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
            # Drink from the current cell without moving — plants and
            # animals are only harvested on arrival (via _move_member).
            self._consume_water(
                self.controlled_band_id,
                member_id,
            )

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
            agent_id: self._build_agent_info(
                *self._member_for_agent(agent_id)
            )
            for agent_id in agent_ids
        }

    def _build_agent_info(
        self, band_id: int, member_id: int
    ) -> dict[str, Any]:
        state = self.member_state(band_id, member_id)
        return {
            "agent_id": state.agent_id,
            "band_id": state.band_id,
            "member_slot": state.member_slot,
            "member_id": state.member_id,
            "alive": state.alive,
            "age": state.age,
            "energy": state.energy,
            "hydration": state.hydration,
            "body_mass_kg": state.body_mass_kg,
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
            "last_animal_food_gained": float(
                self.member_last_animal_food_gained[band_id, member_id]
            ),
            "last_water_gained": float(
                self.member_last_water_gained[band_id, member_id]
            ),
            "last_food_deposited": float(
                self.member_last_food_deposited[band_id, member_id]
            ),
            "last_water_deposited": float(
                self.member_last_water_deposited[band_id, member_id]
            ),
            "last_birth_events": int(self._last_birth_events[band_id]),
            "last_energy_spent": float(
                self.member_last_energy_spent[band_id, member_id]
            ),
            "last_hydration_spent": float(
                self.member_last_hydration_spent[band_id, member_id]
            ),
        }
