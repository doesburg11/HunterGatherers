import unittest

import numpy as np

from hunter_gatherers import (
    BandMemberPatchEnv,
    HunterGathererPatchEnv,
    PatchEnvConfig,
)
from hunter_gatherers.envs.patch_env import Action, MemberSex, Season, Terrain


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
        self.assertEqual(env.agent_pos, env.camp_pos)

    def test_controlled_member_cannot_move_into_occupied_cell(self):
        env = HunterGathererPatchEnv(
            PatchEnvConfig(global_seed=14, num_bands=2)
        )
        env.reset(seed=15)

        start = env.agent_pos
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

        self.assertEqual(env.agent_pos, start)
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
        self.assertTrue(np.all(env.member_age == 0))
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
        self.assertEqual(info["controlled_agent_id"], "band_0_member_0")

    def test_energy_alias_updates_controlled_member_energy(self):
        env = HunterGathererPatchEnv(PatchEnvConfig(members_per_band=3))
        env.reset(seed=17)

        env.energy = 42.0

        self.assertEqual(env.member_energy[0, 0], 42.0)
        self.assertEqual(env.energy, 42.0)

    def test_camp_potential_energy_sums_unreaped_energy_in_camp_radius(self):
        env = HunterGathererPatchEnv(
            PatchEnvConfig(
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

    def test_hydration_alias_updates_controlled_member_hydration(self):
        env = HunterGathererPatchEnv(PatchEnvConfig(members_per_band=3))
        env.reset(seed=19)

        env.hydration = 37.0

        self.assertEqual(env.member_hydration[0, 0], 37.0)
        self.assertEqual(env.hydration, 37.0)

    def test_member_state_reports_lifecycle_fields(self):
        env = HunterGathererPatchEnv(PatchEnvConfig(members_per_band=3))
        env.reset(seed=18)

        state = env.member_state(0, 1)

        self.assertEqual(state.agent_id, "band_0_member_1")
        self.assertEqual(state.band_id, 0)
        self.assertEqual(state.member_slot, 1)
        self.assertEqual(state.member_id, 1)
        self.assertTrue(state.alive)
        self.assertEqual(state.age, 0)
        self.assertEqual(state.sex, MemberSex.MALE)
        self.assertEqual(state.energy, env.config.initial_energy)
        self.assertEqual(state.hydration, env.config.initial_hydration)
        self.assertEqual(state.position, env._member_position(0, 1))

    def test_entering_plant_cell_eats_and_adds_energy(self):
        env = HunterGathererPatchEnv(
            PatchEnvConfig(
                grid_size=7,
                obs_range=5,
                members_per_band=1,
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
        env.energy = 50.0

        _, reward, terminated, truncated, info = env.step(Action.MOVE_EAST)

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        expected_plant_energy = (
            8.0
            - env.config.plant_eat_amount
            + env.config.plant_regrowth_base * 1.35
        )
        self.assertAlmostEqual(env.plant_energy[target], expected_plant_energy)
        expected_depletion = 1.0 - (
            expected_plant_energy / env.config.plant_energy_capacity
        )
        self.assertAlmostEqual(env.depletion[target], expected_depletion)
        self.assertGreater(env.energy, 50.0)
        self.assertGreater(reward, 0.0)
        self.assertEqual(info["plant_eating_events"], 1)

    def test_entering_water_cell_restores_hydration(self):
        env = HunterGathererPatchEnv(
            PatchEnvConfig(
                grid_size=7,
                obs_range=5,
                members_per_band=1,
                initial_hydration=50.0,
                drink_amount=25.0,
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
        self.assertAlmostEqual(env.hydration, 74.0)
        self.assertEqual(info["water_drinking_events"], 1)
        self.assertAlmostEqual(info["last_water_gained"], 25.0)
        self.assertGreater(reward, 0.0)

    def test_depletion_relocates_camp_without_regenerating_patch(self):
        env = HunterGathererPatchEnv(
            PatchEnvConfig(
                grid_size=7,
                obs_range=5,
                members_per_band=3,
                global_seed=3,
                camp_depletion_radius=1,
                camp_depletion_threshold=0.5,
            )
        )
        env.reset(seed=4)
        self.fill_land_with_plants(env)
        old_camp = env.camp_pos
        old_terrain = env.terrain.copy()
        old_water = env.water.copy()
        old_animal_density = env.animal_density.copy()
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
        self.assertEqual(env.agent_pos, env.camp_pos)
        np.testing.assert_array_equal(env.terrain, old_terrain)
        np.testing.assert_array_equal(env.water, old_water)
        np.testing.assert_allclose(env.animal_density, old_animal_density)
        np.testing.assert_allclose(env.danger, old_danger)

    def test_patch_generation_is_deterministic_by_macro_coordinate(self):
        config = PatchEnvConfig(global_seed=5)
        env_a = HunterGathererPatchEnv(config)
        env_b = HunterGathererPatchEnv(config)

        env_a.reset(options={"macro_x": 20, "macro_y": -10})
        env_b.reset(options={"macro_x": 20, "macro_y": -10})

        np.testing.assert_array_equal(env_a.terrain, env_b.terrain)
        np.testing.assert_allclose(env_a.plant_energy, env_b.plant_energy)
        np.testing.assert_allclose(env_a.animal_density, env_b.animal_density)

    def test_generated_patch_only_has_grass_water_and_plants(self):
        env = HunterGathererPatchEnv(PatchEnvConfig(global_seed=6))
        env.reset(seed=7)

        terrain_values = {Terrain(int(value)) for value in np.unique(env.terrain)}
        self.assertEqual(
            terrain_values,
            {Terrain.GRASSLAND, Terrain.WATER},
        )
        self.assertTrue(np.any(env.plant_energy > 0.0))
        self.assertTrue(np.all(env.plant_energy[env.water > 0.0] == 0.0))
        self.assertTrue(hasattr(env, "plant_capacity"))
        self.assertTrue(np.all(env.plant_energy <= env.plant_capacity))
        self.assertTrue(np.all(env.animal_density == 0.0))
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
        env = HunterGathererPatchEnv(PatchEnvConfig(initial_energy=0.1))
        env.reset()

        _, _, terminated, _, info = env.step(Action.STAY)

        self.assertTrue(terminated)
        self.assertEqual(info["starvation_events"], 1)
        self.assertEqual(info["dehydration_events"], 0)

    def test_episode_terminates_on_dehydration(self):
        env = HunterGathererPatchEnv(
            PatchEnvConfig(
                initial_energy=10.0,
                initial_hydration=0.5,
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
            PatchEnvConfig(
                members_per_band=3,
                initial_energy=10.0,
            ),
        )
        env.reset(seed=22)

        _, rewards, terminations, truncations, infos = env.step({})

        np.testing.assert_allclose(env.member_energy[0], [9.85, 9.85, 9.85])
        np.testing.assert_allclose(env.member_hydration[0], [99.0, 99.0, 99.0])
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
                1.0,
            )

    def test_starved_members_disappear_from_active_agents(self):
        env = BandMemberPatchEnv(
            PatchEnvConfig(
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

    def test_simultaneous_moves_reject_same_target_cell(self):
        env = BandMemberPatchEnv(
            PatchEnvConfig(
                grid_size=7,
                obs_range=5,
                members_per_band=2,
                initial_energy=10.0,
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


if __name__ == "__main__":
    unittest.main()
