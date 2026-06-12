import unittest

import numpy as np

from hunter_gatherers import (
    BandMemberPatchEnv,
    HunterGathererPatchEnv,
    PatchEnvConfig,
)
from hunter_gatherers.envs.patch_env import Action, MemberSex, Season, Terrain


def legacy_config(**kwargs):
    legacy_defaults = {
        "use_physical_energetics": False,
        "drink_amount": 25.0,
        "plant_energy_capacity": 8.0,
        "plant_eat_amount": 2.0,
        "food_carry_capacity": 30.0,
        "animal_energy_capacity": 0.0,
        "initial_animals": 0,
        "max_animals": 0,
        "animal_hunt_amount": 0.0,
        "water_carry_capacity": 30.0,
        "water_collect_amount": 25.0,
        "personal_energy_reserve": 100.0,
        "camp_food_withdraw_threshold": 80.0,
        "camp_food_withdraw_amount": 5.0,
        "camp_water_withdraw_amount": 10.0,
        "birth_food_cost": 40.0,
        "birth_water_cost": 30.0,
        # Legacy tests: no gestation delay, birth is immediate.
        "gestation_steps": 0,
    }
    legacy_defaults.update(kwargs)
    return PatchEnvConfig(**legacy_defaults)


class HunterGathererPatchEnvTest(unittest.TestCase):
    def fill_land_with_plants(self, env):
        env.terrain[:, :] = Terrain.GRASSLAND
        env.water[:, :] = 0.0
        env.plant_capacity[:, :] = env.config.plant_energy_capacity
        env.plant_energy[:, :] = env.config.plant_energy_capacity
        env._update_depletion()

    def deplete_camp_area(self, env):
        cy, cx = env.camp_pos
        radius = env.config.camp_depletion_radius
        for y in range(env.grid_size):
            for x in range(env.grid_size):
                if abs(y - cy) + abs(x - cx) <= radius:
                    env.plant_energy[y, x] = 0.0
        env._update_depletion()

    def assert_foraging_areas_do_not_overlap(self, env, old_camp):
        distance = (
            abs(env.camp_pos[0] - old_camp[0])
            + abs(env.camp_pos[1] - old_camp[1])
        )
        self.assertGreater(distance, 2 * env.config.camp_depletion_radius)

    def unique_member_count(self, env):
        positions = env.member_positions.reshape(-1, 2)
        unique_positions = {(int(y), int(x)) for y, x in positions}
        return len(unique_positions)

    def water_component_sizes(self, water):
        seen = np.zeros_like(water, dtype=bool)
        sizes = []
        for start_y, start_x in np.argwhere(water):
            if seen[start_y, start_x]:
                continue
            stack = [(int(start_y), int(start_x))]
            seen[start_y, start_x] = True
            size = 0
            while stack:
                y, x = stack.pop()
                size += 1
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny = y + dy
                    nx = x + dx
                    if ny < 0 or nx < 0:
                        continue
                    if ny >= water.shape[0] or nx >= water.shape[1]:
                        continue
                    if water[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            sizes.append(size)
        return sizes

    def test_reset_returns_expected_observation(self):
        env = HunterGathererPatchEnv(
            PatchEnvConfig(grid_size=31, obs_range=11, num_bands=2)
        )

        obs, info = env.reset(seed=7)

        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(info["macro_x"], 0)
        self.assertEqual(info["macro_y"], 0)
        self.assertEqual(info["season"], "spring")
        self.assertEqual(info["num_bands"], 2)
        self.assertEqual(info["members_per_band"], 15)
        self.assertEqual(env.member_positions.shape, (2, 15, 2))
        self.assertEqual(self.unique_member_count(env), 30)

    def test_members_remain_unique_after_automatic_camp_relocation(self):
        env = HunterGathererPatchEnv(
            PatchEnvConfig(
                grid_size=7,
                obs_range=5,
                global_seed=12,
                num_bands=2,
                members_per_band=3,
                camp_depletion_radius=1,
                camp_depletion_threshold=0.5,
            )
        )
        env.reset(seed=13)
        self.fill_land_with_plants(env)
        old_camp = env.camp_pos
        self.deplete_camp_area(env)

        env.step(Action.STAY)

        self.assertNotEqual(env.camp_pos, old_camp)
        self.assert_foraging_areas_do_not_overlap(env, old_camp)
        self.assertEqual(env.num_camp_moves, 1)
        self.assertEqual(env.member_positions.shape, (2, 3, 2))
        self.assertEqual(self.unique_member_count(env), 6)
        self.assertEqual(env._member_position(0, 0), env.camp_pos)

    def test_controlled_member_cannot_move_into_occupied_cell(self):
        env = HunterGathererPatchEnv(
            PatchEnvConfig(global_seed=14, num_bands=2)
        )
        env.reset(seed=15)

        start = env._member_position(0, 0)
        actions = {
            Action.MOVE_NORTH: (-1, 0),
            Action.MOVE_SOUTH: (1, 0),
            Action.MOVE_WEST: (0, -1),
            Action.MOVE_EAST: (0, 1),
        }
        occupied = {
            (int(y), int(x))
            for y, x in env.member_positions.reshape(-1, 2)
        }
        occupied.remove(start)

        blocked_action = None
        for action, (dy, dx) in actions.items():
            target = (start[0] + dy, start[1] + dx)
            if target in occupied:
                blocked_action = action
                break

        self.assertIsNotNone(blocked_action)
        env.step(blocked_action)

        self.assertEqual(env._member_position(0, 0), start)
        self.assertEqual(self.unique_member_count(env), 30)

    def test_reset_initializes_member_lifecycle_state(self):
        env = HunterGathererPatchEnv(
            PatchEnvConfig(num_bands=2, members_per_band=4)
        )

        _, info = env.reset(seed=16)

        self.assertEqual(env.member_ids.shape, (2, 4))
        np.testing.assert_array_equal(env.member_ids[0], np.arange(4))
        self.assertEqual(env.next_member_ids.tolist(), [4, 4])
        self.assertTrue(np.all(env.member_alive))
        self.assertTrue(
            np.all(env.member_age == env.config.reproduction_adult_age)
        )
        self.assertEqual(env.member_sex[0, 0], MemberSex.FEMALE)
        self.assertEqual(env.member_sex[0, 1], MemberSex.MALE)
        self.assertEqual(env.member_energy.shape, (2, 4))
        self.assertEqual(env.member_hydration.shape, (2, 4))
        self.assertTrue(
            np.all(env.member_hydration == env.config.initial_hydration)
        )
        self.assertEqual(info["population"], 8)
        self.assertEqual(info["max_population"], 8)
        self.assertEqual(len(info["active_agent_ids"]), 8)

    def test_reset_can_reserve_inactive_member_slots(self):
        env = HunterGathererPatchEnv(
            PatchEnvConfig(
                num_bands=1,
                members_per_band=2,
                max_members_per_band=4,
            )
        )

        _, info = env.reset(seed=16)

        self.assertEqual(env.member_ids.shape, (1, 4))
        self.assertEqual(
            env.member_alive[0].tolist(),
            [True, True, False, False],
        )
        self.assertEqual(
            env.member_age[0].tolist(),
            [
                env.config.reproduction_adult_age,
                env.config.reproduction_adult_age,
                0,
                0,
            ],
        )
        self.assertEqual(env.next_member_ids.tolist(), [4])
        self.assertEqual(info["population"], 2)
        self.assertEqual(info["max_population"], 4)

    def test_band_reproduction_activates_reserved_slot(self):
        env = HunterGathererPatchEnv(
            legacy_config(
                grid_size=7,
                obs_range=5,
                num_bands=1,
                members_per_band=2,
                max_members_per_band=3,
                reproduction_adult_age=1,
                birth_rate=1.0,
                birth_food_cost=10.0,
                birth_water_cost=5.0,
                newborn_energy=50.0,
                newborn_hydration=60.0,
                thirst_per_step=0.0,
            )
        )
        env.reset(seed=17, options={"stored_food": 20.0, "stored_water": 20.0})

        _, reward, _, _, info = env.step(Action.STAY)

        self.assertEqual(env.population, 3)
        self.assertTrue(env.member_alive[0, 2])
        self.assertEqual(env.member_ids[0, 2], 3)
        self.assertEqual(env.next_member_ids.tolist(), [4])
        self.assertEqual(env.member_age[0, 2], 0)
        self.assertAlmostEqual(env.member_energy[0, 2], 50.0)
        self.assertAlmostEqual(env.member_hydration[0, 2], 60.0)
        self.assertAlmostEqual(env.camp_stored_food[0], 10.0)
        self.assertAlmostEqual(env.camp_stored_water[0], 15.0)
        self.assertEqual(info["birth_events"], 1)
        self.assertEqual(info["last_birth_events"], [1])
        self.assertGreater(reward, env.config.birth_reward)

    def test_band_reproduction_requires_adult_parents(self):
        env = HunterGathererPatchEnv(
            PatchEnvConfig(
                grid_size=7,
                obs_range=5,
                num_bands=1,
                members_per_band=2,
                max_members_per_band=3,
                reproduction_adult_age=10,
                birth_rate=1.0,
                thirst_per_step=0.0,
            )
        )
        env.reset(
            seed=18,
            options={"stored_food": 100.0, "stored_water": 100.0},
        )
        env.member_age[0, :2] = 0

        env.step(Action.STAY)

        self.assertEqual(env.population, 2)
        self.assertFalse(env.member_alive[0, 2])
        self.assertEqual(env.birth_events, 0)

    def test_juvenile_controlled_member_cannot_move_or_gather(self):
        env = HunterGathererPatchEnv(
            legacy_config(
                grid_size=7,
                obs_range=5,
                members_per_band=1,
                initial_energy=50.0,
                reproduction_adult_age=10,
                global_seed=19,
            )
        )
        env.reset(seed=20)
        env.member_age[0, 0] = 0
        env.terrain[:, :] = Terrain.GRASSLAND
        env.water[:, :] = 0.0
        env.plant_capacity[:, :] = 0.0
        env.plant_energy[:, :] = 0.0
        env._set_member_position(0, 0, (3, 3))
        env.plant_capacity[3, 4] = env.config.plant_energy_capacity
        env.plant_energy[3, 4] = env.config.plant_energy_capacity

        env.step(Action.MOVE_EAST)

        self.assertEqual(env._member_position(0, 0), (3, 3))
        self.assertAlmostEqual(env.member_last_food_gained[0, 0], 0.0)
        self.assertAlmostEqual(env.member_energy[0, 0], 49.85, places=5)

    def test_adult_controlled_member_can_move_and_gather(self):
        env = HunterGathererPatchEnv(
            legacy_config(
                grid_size=7,
                obs_range=5,
                members_per_band=1,
                initial_energy=50.0,
                reproduction_adult_age=0,
                global_seed=19,
            )
        )
        env.reset(seed=20)
        env.terrain[:, :] = Terrain.GRASSLAND
        env.water[:, :] = 0.0
        env.plant_capacity[:, :] = 0.0
        env.plant_energy[:, :] = 0.0
        env._set_member_position(0, 0, (3, 3))
        env.plant_capacity[3, 4] = env.config.plant_energy_capacity
        env.plant_energy[3, 4] = env.config.plant_energy_capacity

        env.step(Action.MOVE_EAST)

        self.assertEqual(env._member_position(0, 0), (3, 4))
        self.assertGreater(env.member_last_food_gained[0, 0], 0.0)

    def test_camp_potential_energy_sums_unreaped_energy_in_camp_radius(self):
        env = HunterGathererPatchEnv(
            legacy_config(
                grid_size=7,
                obs_range=5,
                members_per_band=1,
                camp_depletion_radius=1,
            )
        )
        env.reset(seed=18)
        env.terrain[:, :] = Terrain.GRASSLAND
        env.water[:, :] = 0.0
        env.plant_capacity[:, :] = 0.0
        env.plant_energy[:, :] = 0.0
        env.camp_pos = (3, 3)
        env.band_camp_positions[0] = env.camp_pos
        inside_energy = {
            (3, 3): 4.0,
            (2, 3): 3.0,
            (4, 3): 2.0,
            (3, 2): 1.0,
            (3, 4): 0.5,
        }
        for pos, energy in inside_energy.items():
            env.plant_capacity[pos] = env.config.plant_energy_capacity
            env.plant_energy[pos] = energy
        env.plant_capacity[3, 5] = env.config.plant_energy_capacity
        env.plant_energy[3, 5] = 8.0
        env._update_depletion()

        expected_energy = sum(inside_energy.values())

        self.assertAlmostEqual(env.camp_potential_energy(), expected_energy)
        self.assertAlmostEqual(
            env._build_info()["camp_potential_energy"],
            expected_energy,
        )

    def test_member_state_reports_lifecycle_fields(self):
        env = HunterGathererPatchEnv(PatchEnvConfig(members_per_band=3))
        env.reset(seed=18)

        state = env.member_state(0, 1)

        self.assertEqual(state.agent_id, "band_0_member_1")
        self.assertEqual(state.band_id, 0)
        self.assertEqual(state.member_slot, 1)
        self.assertEqual(state.member_id, 1)
        self.assertTrue(state.alive)
        self.assertEqual(state.age, env.config.reproduction_adult_age)
        self.assertEqual(state.sex, MemberSex.MALE)
        self.assertEqual(
            state.energy,
            env.config.adult_body_mass_kg * env.config.initial_energy_per_kg,
        )
        self.assertEqual(state.hydration, env.config.initial_hydration)
        self.assertEqual(state.body_mass_kg, env.config.adult_body_mass_kg)
        self.assertEqual(state.position, env._member_position(0, 1))

    def test_entering_plant_cell_eats_and_adds_energy(self):
        env = HunterGathererPatchEnv(
            legacy_config(
                grid_size=7,
                obs_range=5,
                members_per_band=1,
                camp_storage_radius=0,
                reproduction_adult_age=0,
                global_seed=1,
            )
        )
        env.reset(seed=2)
        env.terrain[:, :] = Terrain.GRASSLAND
        env.water[:, :] = 0.0
        env.plant_capacity[:, :] = 0.0
        env.plant_energy[:, :] = 0.0
        env._set_member_position(0, 0, (3, 3))
        env.camp_pos = (3, 3)
        target = (3, 4)
        env.plant_capacity[target] = 8.0
        env.plant_energy[target] = 8.0
        env.member_energy[0, 0] = 50.0

        _, reward, terminated, truncated, info = env.step(Action.MOVE_EAST)

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        expected_plant_energy = (
            8.0
            - env.config.plant_eat_amount
            + env.config.plant_regrowth_base * 1.35
        )
        self.assertAlmostEqual(
            env.plant_energy[target], expected_plant_energy, places=6
        )
        expected_depletion = 1.0 - (
            expected_plant_energy / env.config.plant_energy_capacity
        )
        self.assertAlmostEqual(env.depletion[target], expected_depletion)
        self.assertGreater(float(env.member_energy[0, 0]), 50.0)
        # Agent starts hungry (energy=50 < personal_reserve=100); eats all
        # harvested plant energy to fill body need, carries nothing.
        self.assertAlmostEqual(env.member_carried_food[0, 0], 0.0)
        self.assertGreater(reward, 0.0)
        self.assertEqual(info["plant_eating_events"], 1)

    def test_entering_animal_cell_harvests_animal_energy(self):
        env = HunterGathererPatchEnv(
            PatchEnvConfig(
                grid_size=7,
                obs_range=5,
                members_per_band=1,
                initial_energy_per_kg=20.0,
                animal_energy_capacity=2500.0,
                initial_animals=0,
                max_animals=2,
                animal_move_cost=0.0,
                animal_eat_amount=0.0,
                animal_hunt_amount=800.0,
                animal_carry_share=1.0,
                camp_storage_radius=0,
                reproduction_adult_age=0,
                global_seed=20,
            )
        )
        env.reset(seed=21)
        env.terrain[:, :] = Terrain.GRASSLAND
        env.water[:, :] = 0.0
        env.plant_energy[:, :] = 0.0
        env._set_member_position(0, 0, (3, 3))
        env.camp_pos = (3, 3)
        target = (3, 4)
        env.animal_alive[0] = True
        env.animal_positions[0] = target
        env.animal_individual_energy[0] = 1200.0
        env._sync_animal_energy_layer()

        _, reward, terminated, truncated, info = env.step(Action.MOVE_EAST)

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(env.animal_alive[0])
        self.assertAlmostEqual(env.animal_individual_energy[0], 400.0)
        self.assertAlmostEqual(np.sum(env.animal_energy), 400.0)
        self.assertAlmostEqual(env.member_last_food_gained[0, 0], 800.0)
        # Physical energetics: personal_reserve = max_energy
        # = 70 kg × 35 kcal/kg = 2450.
        # Agent at 1400 kcal; body_need 1050 > 800 hunted → eats all.
        # Carries nothing.
        self.assertAlmostEqual(env.member_carried_food[0, 0], 0.0)
        self.assertGreater(reward, 0.0)
        self.assertEqual(
            info["mean_animal_energy"], np.mean(env.animal_energy)
        )

    def test_division_of_labor_female_gathers_not_hunts(self):
        env = HunterGathererPatchEnv(
            PatchEnvConfig(
                grid_size=7,
                obs_range=5,
                members_per_band=1,
                initial_energy_per_kg=20.0,
                plant_energy_capacity=8.0,
                plant_eat_amount=8.0,
                plant_regrowth_base=0.0,
                animal_energy_capacity=2500.0,
                initial_animals=0,
                max_animals=1,
                animal_move_cost=0.0,
                animal_eat_amount=0.0,
                animal_hunt_amount=800.0,
                camp_storage_radius=0,
                reproduction_adult_age=0,
                division_of_labor_enabled=True,
                global_seed=200,
            )
        )
        env.reset(seed=201)
        env.terrain[:, :] = Terrain.GRASSLAND
        env.water[:, :] = 0.0
        env.plant_capacity[:, :] = 0.0
        env.plant_energy[:, :] = 0.0
        env._set_member_position(0, 0, (3, 3))
        target = (3, 4)
        env.plant_capacity[target] = 8.0
        env.plant_energy[target] = 8.0
        env.animal_alive[0] = True
        env.animal_positions[0] = target
        env.animal_individual_energy[0] = 1200.0
        env._sync_animal_energy_layer()

        env.step(Action.MOVE_EAST)

        self.assertAlmostEqual(env.plant_energy[target], 0.0)
        self.assertAlmostEqual(env.animal_individual_energy[0], 1200.0)
        self.assertAlmostEqual(env.member_last_animal_food_gained[0, 0], 0.0)
        self.assertEqual(env.plant_eating_events, 1)

    def test_division_of_labor_male_hunts_not_gathers(self):
        env = HunterGathererPatchEnv(
            PatchEnvConfig(
                grid_size=7,
                obs_range=5,
                members_per_band=1,
                initial_energy_per_kg=20.0,
                plant_energy_capacity=8.0,
                plant_eat_amount=8.0,
                plant_regrowth_base=0.0,
                animal_energy_capacity=2500.0,
                initial_animals=0,
                max_animals=1,
                animal_move_cost=0.0,
                animal_eat_amount=0.0,
                animal_hunt_amount=800.0,
                camp_storage_radius=0,
                reproduction_adult_age=0,
                division_of_labor_enabled=True,
                global_seed=202,
            )
        )
        env.reset(seed=203)
        env.member_sex[0, 0] = int(MemberSex.MALE)
        env.terrain[:, :] = Terrain.GRASSLAND
        env.water[:, :] = 0.0
        env.plant_capacity[:, :] = 0.0
        env.plant_energy[:, :] = 0.0
        env._set_member_position(0, 0, (3, 3))
        target = (3, 4)
        env.plant_capacity[target] = 8.0
        env.plant_energy[target] = 8.0
        env.animal_alive[0] = True
        env.animal_positions[0] = target
        env.animal_individual_energy[0] = 1200.0
        env._sync_animal_energy_layer()

        env.step(Action.MOVE_EAST)

        self.assertAlmostEqual(env.plant_energy[target], 8.0)
        self.assertAlmostEqual(env.animal_individual_energy[0], 400.0)
        self.assertAlmostEqual(env.member_last_animal_food_gained[0, 0], 800.0)
        self.assertEqual(env.plant_eating_events, 0)

    def test_animals_move_eat_grass_and_reproduce(self):
        env = HunterGathererPatchEnv(
            PatchEnvConfig(
                grid_size=7,
                obs_range=5,
                members_per_band=1,
                initial_animals=0,
                max_animals=3,
                initial_animal_energy=1000.0,
                max_animal_energy=2500.0,
                animal_move_cost=0.0,
                animal_eat_amount=200.0,
                animal_reproduction_energy_threshold=2000.0,
                animal_reproduction_cost=900.0,
                newborn_animal_energy=800.0,
                global_seed=21,
            )
        )
        env.reset(seed=22)
        env.terrain[:, :] = Terrain.GRASSLAND
        env.water[:, :] = 0.0
        env.plant_energy[:, :] = 200.0
        env.grass_energy[:, :] = 1000.0
        env.animal_alive[0] = True
        env.animal_positions[0] = (3, 3)
        env.animal_individual_energy[0] = 2000.0

        env._update_animals()

        self.assertEqual(np.count_nonzero(env.animal_alive), 2)
        self.assertAlmostEqual(np.sum(env.plant_energy), 9800.0)
        self.assertAlmostEqual(np.sum(env.grass_energy), 48800.0)
        self.assertAlmostEqual(
            float(np.sum(env.animal_individual_energy[env.animal_alive])),
            2100.0,
        )
        self.assertAlmostEqual(
            float(np.sum(env.animal_energy)),
            2100.0,
        )

    def test_entering_water_cell_restores_hydration(self):
        env = HunterGathererPatchEnv(
            legacy_config(
                grid_size=7,
                obs_range=5,
                members_per_band=1,
                initial_hydration=50.0,
                drink_amount=25.0,
                camp_storage_radius=0,
                reproduction_adult_age=0,
                thirst_per_step=1.0,
                global_seed=20,
            )
        )
        env.reset(seed=21)
        env.terrain[:, :] = Terrain.GRASSLAND
        env.water[:, :] = 0.0
        env.terrain[3, 4] = Terrain.WATER
        env.water[3, 4] = 1.0
        env._set_member_position(0, 0, (3, 3))
        env.camp_pos = (3, 3)

        _, reward, terminated, truncated, info = env.step(Action.MOVE_EAST)

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(float(env.member_hydration[0, 0]), 74.0)
        self.assertAlmostEqual(env.member_carried_water[0, 0], 25.0)
        self.assertEqual(info["water_drinking_events"], 1)
        self.assertAlmostEqual(
            float(env.member_last_water_gained[0, 0]), 50.0
        )
        self.assertGreater(reward, 0.0)

    def test_surplus_food_and_water_can_be_deposited_at_camp(self):
        env = HunterGathererPatchEnv(
            legacy_config(
                grid_size=7,
                obs_range=5,
                members_per_band=1,
                initial_energy=100.0,
                initial_hydration=100.0,
                camp_storage_radius=0,
                global_seed=22,
            )
        )
        env.reset(seed=23)
        env._set_member_position(0, 0, env.camp_pos)
        env.member_carried_food[0, 0] = 7.0
        env.member_carried_water[0, 0] = 11.0

        _, reward, _, _, info = env.step(Action.STAY)

        self.assertAlmostEqual(env.member_carried_food[0, 0], 0.0)
        self.assertAlmostEqual(env.member_carried_water[0, 0], 0.0)
        self.assertAlmostEqual(env.camp_stored_food[0], 7.0)
        self.assertAlmostEqual(env.camp_stored_water[0], 11.0)
        self.assertAlmostEqual(env.member_last_food_deposited[0, 0], 7.0)
        self.assertAlmostEqual(env.member_last_water_deposited[0, 0], 11.0)
        self.assertAlmostEqual(info["camp_stored_food"][0], 7.0)
        self.assertAlmostEqual(info["camp_stored_water"][0], 11.0)
        self.assertAlmostEqual(
            float(env.member_last_food_deposited[0, 0]), 7.0
        )
        self.assertAlmostEqual(
            float(env.member_last_water_deposited[0, 0]), 11.0
        )
        expected_reward = (
            7.0 * env.config.food_deposit_reward
            + 11.0 * env.config.water_deposit_reward
            + env.config.population_reward
            - 0.15 * 0.01
            + 0.01
        )
        self.assertAlmostEqual(reward, expected_reward)

    def test_camp_storage_feeds_and_hydrates_members_at_camp(self):
        env = HunterGathererPatchEnv(
            legacy_config(
                grid_size=7,
                obs_range=5,
                members_per_band=1,
                initial_energy=60.0,
                initial_hydration=50.0,
                global_seed=24,
            )
        )
        env.reset(seed=25, options={"stored_food": 20.0, "stored_water": 20.0})
        env._set_member_position(0, 0, env.camp_pos)

        env.step(Action.STAY)

        self.assertAlmostEqual(float(env.member_energy[0, 0]), 64.85, places=5)
        self.assertAlmostEqual(
            float(env.member_hydration[0, 0]), 59.8, places=5
        )
        self.assertAlmostEqual(env.camp_stored_food[0], 15.0)
        self.assertAlmostEqual(env.camp_stored_water[0], 10.0)

    def test_camp_storage_can_be_used_near_camp(self):
        env = HunterGathererPatchEnv(
            legacy_config(
                grid_size=7,
                obs_range=5,
                members_per_band=2,
                initial_energy=60.0,
                initial_hydration=50.0,
                camp_storage_radius=1,
                global_seed=25,
            )
        )
        env.reset(seed=26, options={"stored_food": 20.0, "stored_water": 20.0})
        cy, cx = env.camp_pos
        env._set_member_position(0, 0, (cy, cx - 1))
        env._set_member_position(0, 1, (cy, cx + 1))
        env.member_carried_food[0, 1] = 7.0
        env.member_carried_water[0, 1] = 11.0

        env.step(Action.STAY)

        self.assertAlmostEqual(env.member_energy[0, 0], 64.85, places=5)
        # Member 0 is at (cy, cx-1) which is a water cell; STAY now lets
        # members drink from their current cell. After drinking (50+25=75),
        # hydration exceeds the camp_water_withdraw_threshold (60), so no camp
        # withdraw fires. Only thirst is subtracted: 75 − 0.2 = 74.8.
        self.assertAlmostEqual(env.member_hydration[0, 0], 74.8)
        self.assertAlmostEqual(env.member_carried_food[0, 1], 0.0)
        self.assertAlmostEqual(env.member_carried_water[0, 1], 0.0)

    def test_movement_cost_scales_with_internal_and_carried_energy(self):
        env = HunterGathererPatchEnv(
            legacy_config(
                grid_size=7,
                obs_range=5,
                members_per_band=1,
                movement_energy_rate=0.01,
                water_load_factor=2.0,
                reproduction_adult_age=0,
                global_seed=26,
            )
        )
        env.reset(seed=27)
        env.terrain[:, :] = Terrain.GRASSLAND
        env.water[:, :] = 0.0
        env._set_member_position(0, 0, (3, 3))
        env.camp_pos = (3, 3)
        env.member_energy[0, 0] = 100.0
        env.member_carried_food[0, 0] = 20.0
        env.member_carried_water[0, 0] = 10.0

        env.step(Action.MOVE_EAST)

        expected_spent = (100.0 + 20.0 + 2.0 * 10.0) * 0.01 * 1.05 + 0.015
        self.assertAlmostEqual(
            env.member_last_energy_spent[0, 0],
            expected_spent,
            delta=1e-5,
        )

    def test_physical_movement_cost_uses_body_and_carried_mass(self):
        env = HunterGathererPatchEnv(
            PatchEnvConfig(
                grid_size=7,
                obs_range=5,
                members_per_band=1,
                use_physical_energetics=True,
                adult_body_mass_kg=100.0,
                initial_energy_per_kg=30.0,
                max_energy_per_kg=35.0,
                cell_distance_km=1.0,
                walking_kcal_per_kg_km=1.0,
                movement_cost=1.0,
                grass_movement_cost=1.0,
                food_energy_density_kcal_per_kg=2500.0,
                water_liters_per_unit=1.0,
                reproduction_adult_age=0,
                global_seed=27,
            )
        )
        env.reset(seed=28)
        env.terrain[:, :] = Terrain.GRASSLAND
        env.water[:, :] = 0.0
        env.plant_energy[:, :] = 0.0
        env._set_member_position(0, 0, (3, 3))
        env.camp_pos = (3, 3)
        env.member_carried_food[0, 0] = 2500.0
        env.member_carried_water[0, 0] = 2.0

        env.step(Action.MOVE_EAST)

        expected_spent = (100.0 + 1.0 + 2.0) * 1.0 * 1.0 * 1.05
        self.assertAlmostEqual(
            env.member_last_energy_spent[0, 0],
            expected_spent,
            delta=1e-5,
        )

    def test_physical_energy_cap_scales_with_body_mass(self):
        env = HunterGathererPatchEnv(
            PatchEnvConfig(
                members_per_band=1,
                use_physical_energetics=True,
                adult_body_mass_kg=50.0,
                initial_energy_per_kg=10.0,
                max_energy_per_kg=35.0,
                reproduction_adult_age=0,
                global_seed=28,
            )
        )
        env.reset(seed=29)

        env._add_member_energy(0, 0, 10000.0)

        self.assertAlmostEqual(env.member_energy[0, 0], 1750.0)

    def test_physical_juvenile_body_mass_grows_to_adult_mass(self):
        env = HunterGathererPatchEnv(
            PatchEnvConfig(
                members_per_band=1,
                max_members_per_band=2,
                use_physical_energetics=True,
                adult_body_mass_kg=70.0,
                newborn_body_mass_kg=10.0,
                reproduction_adult_age=10,
                global_seed=29,
            )
        )
        env.reset(seed=30)
        env._activate_newborn(0, 1, env.camp_pos)

        self.assertAlmostEqual(env.member_body_mass_kg[0, 1], 10.0)

        for _ in range(5):
            env._advance_member_lifecycle()
        self.assertAlmostEqual(env.member_body_mass_kg[0, 1], 40.0)

        for _ in range(5):
            env._advance_member_lifecycle()
        self.assertAlmostEqual(env.member_body_mass_kg[0, 1], 70.0)

    def test_depletion_relocates_camp_without_regenerating_patch(self):
        env = HunterGathererPatchEnv(
            PatchEnvConfig(
                grid_size=7,
                obs_range=5,
                members_per_band=3,
                global_seed=3,
                camp_depletion_radius=1,
                camp_depletion_threshold=0.5,
                initial_animals=0,
                max_animals=0,
            )
        )
        env.reset(seed=4)
        self.fill_land_with_plants(env)
        old_camp = env.camp_pos
        old_terrain = env.terrain.copy()
        old_water = env.water.copy()
        old_danger = env.danger.copy()
        self.deplete_camp_area(env)

        env.step(Action.STAY)

        self.assertNotEqual(env.camp_pos, old_camp)
        self.assert_foraging_areas_do_not_overlap(env, old_camp)
        self.assertEqual(env.camp_id, 1)
        self.assertEqual(env.num_camp_moves, 1)
        expected_distance = (
            abs(env.camp_pos[0] - old_camp[0])
            + abs(env.camp_pos[1] - old_camp[1])
        )
        self.assertEqual(env.macro_distance_traveled, expected_distance)
        self.assertEqual(env._member_position(0, 0), env.camp_pos)
        np.testing.assert_array_equal(env.terrain, old_terrain)
        np.testing.assert_array_equal(env.water, old_water)
        np.testing.assert_allclose(env.danger, old_danger)

    def test_camp_relocation_cost_scales_by_distance_and_member_load(self):
        env = HunterGathererPatchEnv(
            legacy_config(
                grid_size=7,
                obs_range=5,
                members_per_band=2,
                initial_energy=100.0,
                camp_relocation_cost=1.0,
                movement_energy_rate=0.01,
                water_load_factor=2.0,
                camp_storage_radius=0,
                camp_depletion_radius=1,
                camp_depletion_threshold=0.5,
            )
        )
        env.reset(seed=28)
        self.fill_land_with_plants(env)
        old_camp = env.camp_pos
        env._set_member_position(0, 0, (0, 0))
        env._set_member_position(0, 1, (0, 1))
        env.member_carried_food[0, 0] = 20.0
        env.member_carried_water[0, 0] = 10.0
        env.member_carried_food[0, 1] = 5.0
        self.deplete_camp_area(env)

        env.step(Action.STAY)

        camp_distance = (
            abs(env.camp_pos[0] - old_camp[0])
            + abs(env.camp_pos[1] - old_camp[1])
        )
        expected_member_0_cost = camp_distance * 0.01 * (
            99.85 + 20.0 + 2.0 * 10.0
        )
        expected_member_1_cost = camp_distance * 0.01 * (100.0 + 5.0)
        self.assertAlmostEqual(
            env.member_last_energy_spent[0, 0],
            0.15 + expected_member_0_cost,
            places=6,
        )
        self.assertAlmostEqual(
            env.member_last_energy_spent[0, 1],
            expected_member_1_cost,
            places=6,
        )

    def test_patch_generation_is_deterministic_by_macro_coordinate(self):
        config = PatchEnvConfig(global_seed=5)
        env_a = HunterGathererPatchEnv(config)
        env_b = HunterGathererPatchEnv(config)

        env_a.reset(options={"macro_x": 20, "macro_y": -10})
        env_b.reset(options={"macro_x": 20, "macro_y": -10})

        np.testing.assert_array_equal(env_a.terrain, env_b.terrain)
        np.testing.assert_allclose(env_a.plant_energy, env_b.plant_energy)
        np.testing.assert_allclose(env_a.animal_energy, env_b.animal_energy)

    def test_generated_patch_only_has_grass_water_plants_and_animals(self):
        env = HunterGathererPatchEnv(PatchEnvConfig(global_seed=6))
        env.reset(seed=7)

        terrain_values = {
            Terrain(int(v)) for v in np.unique(env.terrain)
        }
        self.assertEqual(
            terrain_values,
            {Terrain.GRASSLAND, Terrain.WATER},
        )
        self.assertTrue(np.any(env.plant_energy > 0.0))
        self.assertTrue(np.all(env.plant_energy[env.water > 0.0] == 0.0))
        self.assertTrue(hasattr(env, "plant_capacity"))
        self.assertTrue(np.all(env.plant_energy <= env.plant_capacity))
        self.assertTrue(np.any(env.grass_energy > 0.0))
        self.assertTrue(np.all(env.grass_energy[env.water > 0.0] == 0.0))
        self.assertTrue(np.any(env.animal_energy > 0.0))
        self.assertTrue(np.all(env.animal_energy[env.water > 0.0] == 0.0))
        self.assertEqual(
            np.count_nonzero(env.animal_alive),
            env.config.initial_animals,
        )
        self.assertTrue(np.all(env.danger == 0.0))
        self.assertTrue(np.all(env.depletion == 0.0))

    def test_plants_do_not_spread_to_plain_grass_cells(self):
        env = HunterGathererPatchEnv(PatchEnvConfig(global_seed=16))
        env.reset(seed=17)
        plain_grass = (
            (env.terrain == Terrain.GRASSLAND)
            & (env.plant_capacity == 0.0)
        )
        self.assertTrue(np.any(plain_grass))

        for _ in range(20):
            env.step(Action.STAY)

        self.assertTrue(np.all(env.plant_energy[plain_grass] == 0.0))

    def test_generated_patch_has_west_to_east_river(self):
        env = HunterGathererPatchEnv(PatchEnvConfig(global_seed=6))
        env.reset(seed=7)

        water = env.terrain == Terrain.WATER
        lower_half_water = water[env.grid_size // 2:, :]
        self.assertTrue(np.all(np.any(lower_half_water, axis=0)))

    def test_generated_patch_has_top_center_lake_without_stray_water(self):
        env = HunterGathererPatchEnv(PatchEnvConfig(global_seed=8))
        env.reset(seed=9)

        water = env.terrain == Terrain.WATER
        top = env.grid_size // 3
        left = env.grid_size // 3
        right = 2 * env.grid_size // 3
        self.assertTrue(np.any(water[:top, left:right]))

        component_sizes = self.water_component_sizes(water)
        self.assertEqual(len(component_sizes), 2)
        self.assertGreaterEqual(min(component_sizes), 10)

    def test_season_advances_by_step_count(self):
        env = HunterGathererPatchEnv(PatchEnvConfig(season_length=2))
        env.reset()

        env.step(Action.STAY)
        self.assertEqual(env.season, Season.SPRING)
        env.step(Action.STAY)
        self.assertEqual(env.season, Season.SUMMER)

    def test_episode_terminates_on_starvation(self):
        env = HunterGathererPatchEnv(legacy_config(initial_energy=0.1))
        env.reset()

        _, _, terminated, _, info = env.step(Action.STAY)

        self.assertTrue(terminated)
        self.assertEqual(info["starvation_events"], 1)
        self.assertEqual(info["dehydration_events"], 0)

    def test_episode_terminates_on_dehydration(self):
        env = HunterGathererPatchEnv(
            legacy_config(
                initial_energy=10.0,
                initial_hydration=0.5,
                thirst_per_step=1.0,
                members_per_band=1,
            )
        )
        env.reset()

        _, _, terminated, _, info = env.step(Action.STAY)

        self.assertTrue(terminated)
        self.assertEqual(info["starvation_events"], 0)
        self.assertEqual(info["dehydration_events"], 1)


class BandMemberPatchEnvTest(unittest.TestCase):
    def test_reset_returns_observations_for_controlled_band_members(self):
        env = BandMemberPatchEnv(
            PatchEnvConfig(num_bands=2, members_per_band=4),
        )

        observations, infos = env.reset(seed=21)

        expected_agents = [f"band_0_member_{i}" for i in range(4)]
        self.assertEqual(env.possible_agents, expected_agents)
        self.assertEqual(env.agents, expected_agents)
        self.assertEqual(set(observations), set(expected_agents))
        self.assertEqual(set(infos), set(expected_agents))
        for agent_id in expected_agents:
            self.assertEqual(
                observations[agent_id].shape,
                env.observation_space.shape,
            )
            self.assertEqual(infos[agent_id]["agent_id"], agent_id)
            self.assertEqual(infos[agent_id]["band_id"], 0)
            self.assertIn("camp_potential_energy", infos[agent_id])

    def test_missing_actions_default_to_stay_for_all_active_members(self):
        env = BandMemberPatchEnv(
            legacy_config(
                members_per_band=3,
                initial_energy=10.0,
            ),
        )
        env.reset(seed=22)

        _, rewards, terminations, truncations, infos = env.step({})

        np.testing.assert_allclose(env.member_energy[0], [9.85, 9.85, 9.85])
        np.testing.assert_allclose(env.member_hydration[0], [99.8, 99.8, 99.8])
        self.assertEqual(env.member_last_action[0].tolist(), [0, 0, 0])
        self.assertEqual(set(rewards), set(env.possible_agents))
        self.assertFalse(terminations["__all__"])
        self.assertFalse(truncations["__all__"])
        for agent_id in env.possible_agents:
            self.assertAlmostEqual(
                infos[agent_id]["last_energy_spent"],
                0.15,
            )
            self.assertAlmostEqual(
                infos[agent_id]["last_hydration_spent"],
                0.2,
            )

    def test_starved_members_disappear_from_active_agents(self):
        env = BandMemberPatchEnv(
            legacy_config(
                members_per_band=2,
                initial_energy=0.1,
            ),
        )
        env.reset(seed=23)

        observations, rewards, terminations, truncations, _ = env.step({})

        self.assertEqual(observations, {})
        self.assertEqual(env.agents, [])
        self.assertTrue(terminations["__all__"])
        self.assertFalse(truncations["__all__"])
        self.assertTrue(all(terminations[agent] for agent in rewards))
        self.assertEqual(env.starvation_events, 2)
        self.assertEqual(env.population, 0)

    def test_truncated_members_receive_final_observations(self):
        env = BandMemberPatchEnv(
            PatchEnvConfig(
                members_per_band=2,
                max_steps=1,
            ),
        )
        observations, _ = env.reset(seed=24)
        initial_agents = set(observations)

        observations, _, terminations, truncations, _ = env.step({})

        self.assertEqual(set(observations), initial_agents)
        self.assertEqual(env.agents, [])
        self.assertFalse(terminations["__all__"])
        self.assertTrue(truncations["__all__"])
        for agent_id in initial_agents:
            self.assertFalse(terminations[agent_id])
            self.assertTrue(truncations[agent_id])

    def test_simultaneous_moves_reject_same_target_cell(self):
        env = BandMemberPatchEnv(
            legacy_config(
                grid_size=7,
                obs_range=5,
                members_per_band=2,
                initial_energy=10.0,
                reproduction_adult_age=0,
            ),
        )
        env.reset(seed=24)
        env.terrain[3, 3] = Terrain.GRASSLAND
        env.terrain[3, 4] = Terrain.GRASSLAND
        env.terrain[3, 5] = Terrain.GRASSLAND
        env._set_member_position(0, 0, (3, 3))
        env._set_member_position(0, 1, (3, 5))

        env.step(
            {
                "band_0_member_0": Action.MOVE_EAST,
                "band_0_member_1": Action.MOVE_WEST,
            }
        )

        self.assertEqual(env._member_position(0, 0), (3, 3))
        self.assertEqual(env._member_position(0, 1), (3, 5))
        np.testing.assert_allclose(env.member_energy[0], [9.85, 9.85])

    def test_auto_camp_relocation_uses_multi_agent_return_shape(self):
        env = BandMemberPatchEnv(
            PatchEnvConfig(
                grid_size=7,
                obs_range=5,
                members_per_band=2,
                camp_depletion_radius=1,
                camp_depletion_threshold=0.5,
            ),
        )
        env.reset(seed=25)
        old_camp = env.camp_pos
        env.terrain[:, :] = Terrain.GRASSLAND
        env.water[:, :] = 0.0
        env.plant_capacity[:, :] = env.config.plant_energy_capacity
        env.plant_energy[:, :] = env.config.plant_energy_capacity
        env._update_depletion()
        cy, cx = env.camp_pos
        for y in range(env.grid_size):
            for x in range(env.grid_size):
                if abs(y - cy) + abs(x - cx) <= 1:
                    env.plant_energy[y, x] = 0.0
        env._update_depletion()

        _, rewards, terminations, truncations, infos = env.step(
            {"band_0_member_0": Action.STAY}
        )

        self.assertEqual(env.num_camp_moves, 1)
        camp_distance = (
            abs(env.camp_pos[0] - old_camp[0])
            + abs(env.camp_pos[1] - old_camp[1])
        )
        self.assertGreater(camp_distance, 2 * env.config.camp_depletion_radius)
        self.assertEqual(set(rewards), set(env.possible_agents))
        self.assertIn("__all__", terminations)
        self.assertIn("__all__", truncations)
        self.assertEqual(infos["band_0_member_0"]["camp_id"], 1)
        self.assertAlmostEqual(
            infos["band_0_member_0"]["last_energy_spent"],
            0.15,
        )

    def test_step_survives_death_and_birth_in_same_step(self):
        # Regression: if a member dies and a newborn immediately takes its
        # slot, member_ids[slot] changes identity mid-step.  The step() used
        # to look up rewards/terminations via _agent_to_member AFTER
        # _refresh_agent_ids, which no longer contained the old identity,
        # raising a KeyError.
        env = BandMemberPatchEnv(
            PatchEnvConfig(
                members_per_band=1,
                max_members_per_band=2,
                initial_energy=0.01,   # member dies this step
                initial_hydration=100.0,
                reproduction_enabled=True,
                birth_rate=1.0,        # guaranteed birth if conditions met
                birth_food_cost=0.0,
                birth_water_cost=0.0,
                max_steps=500,
            )
        )
        env.reset(seed=42)
        # Pre-load camp with food/water so birth conditions are met
        env.camp_stored_food[0] = 9999.0
        env.camp_stored_water[0] = 9999.0

        # Should not raise KeyError
        try:
            env.step({"band_0_member_0": int(Action.STAY)})
        except KeyError as exc:
            self.fail(f"step() raised KeyError: {exc}")

    def test_terminated_true_when_slot_reused_by_newborn(self):
        # Regression: when a member dies AND a newborn immediately occupies
        # that exact slot (because it's the lowest-index empty slot),
        # member_alive[slot] is True again for the newborn.  Without the
        # pre_sim_identity check, terminated[old_agent_id] = False even
        # though the old agent is gone — breaking RLlib's episode tracker.
        #
        # Setup: 3 members fill all slots (max_members_per_band=3).
        # Slot 0 (female) is given -999 energy → dies this step.
        # Slots 1 (male) and 2 (female) survive → valid reproducing pair.
        # The only empty slot after death is slot 0, so the newborn goes there.
        env = BandMemberPatchEnv(
            PatchEnvConfig(
                num_bands=1,
                members_per_band=3,
                max_members_per_band=3,  # no spare slots; death = only vacancy
                reproduction_enabled=True,
                birth_rate=1.0,
                birth_food_cost=0.0,
                birth_water_cost=0.0,
                max_steps=500,
            )
        )
        env.reset(seed=0)
        # Force slot-0 member to die from starvation (set deeply negative so
        # camp food-withdraw can't rescue it).
        env.member_energy[0, 0] = -999.0
        env.member_hydration[0, 0] = 100.0
        env.member_energy[0, 1] = 100.0
        env.member_energy[0, 2] = 100.0
        # Initialisation sets sex as (band+slot)%2 → slot0=F, slot1=M, slot2=F.
        # That's already a valid female+male pair for slots 1 & 2.
        env.camp_stored_food[0] = 9999.0
        env.camp_stored_water[0] = 9999.0

        actions = {
            "band_0_member_0": int(Action.STAY),
            "band_0_member_1": int(Action.STAY),
            "band_0_member_2": int(Action.STAY),
        }
        try:
            _, _, terminations, _, _ = env.step(actions)
        except KeyError as exc:
            self.fail(f"step() raised KeyError: {exc}")

        # Slot 0 was reused → original agent must be terminated
        self.assertTrue(
            terminations.get("band_0_member_0", False),
            "band_0_member_0 should be terminated when its slot was "
            "reused by a newborn in the same step",
        )
        # Surviving agents must not be terminated
        self.assertFalse(terminations.get("band_0_member_1", True))
        self.assertFalse(terminations.get("band_0_member_2", True))


if __name__ == "__main__":
    unittest.main()
