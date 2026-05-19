import unittest

import numpy as np

from hunter_gatherers import HunterGathererPatchEnv, PatchEnvConfig
from hunter_gatherers.envs.patch_env import Action, Season, Terrain


class HunterGathererPatchEnvTest(unittest.TestCase):
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

    def terrain_neighbor_agreement(self, terrain):
        agreements = []
        for dy, dx in ((1, 0), (0, 1)):
            current = terrain[:-dy or None, :-dx or None]
            neighbor = terrain[dy:, dx:]
            mask = (current != Terrain.WATER) & (neighbor != Terrain.WATER)
            agreements.append(np.mean(current[mask] == neighbor[mask]))
        return float(np.mean(agreements))

    def test_reset_returns_expected_observation(self):
        env = HunterGathererPatchEnv(
            PatchEnvConfig(grid_size=31, obs_range=11)
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

    def test_members_remain_unique_after_camp_move(self):
        env = HunterGathererPatchEnv(PatchEnvConfig(global_seed=12))
        env.reset(seed=13)

        env.step(Action.MOVE_CAMP_EAST)

        self.assertEqual(env.member_positions.shape, (2, 15, 2))
        self.assertEqual(self.unique_member_count(env), 30)
        self.assertEqual(env.agent_pos, env.camp_pos)

    def test_controlled_member_cannot_move_into_occupied_cell(self):
        env = HunterGathererPatchEnv(PatchEnvConfig(global_seed=14))
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

    def test_gather_depletes_cell_and_adds_energy(self):
        env = HunterGathererPatchEnv(PatchEnvConfig(global_seed=1))
        env.reset(seed=2)
        y, x = env.agent_pos
        env.plant_food[y, x] = 1.0
        env.energy = 50.0

        _, reward, terminated, truncated, info = env.step(Action.GATHER)

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertLess(env.plant_food[y, x], 1.0)
        self.assertGreater(env.depletion[y, x], 0.0)
        self.assertGreater(env.energy, 50.0)
        self.assertGreater(reward, 0.0)
        self.assertEqual(info["gathering_events"], 1)

    def test_camp_move_changes_macro_position_and_regenerates_patch(self):
        env = HunterGathererPatchEnv(
            PatchEnvConfig(global_seed=3, camp_move_distance=10)
        )
        env.reset(seed=4)
        old_terrain = env.terrain.copy()

        env.step(Action.MOVE_CAMP_EAST)

        self.assertEqual(env.macro_x, 10)
        self.assertEqual(env.macro_y, 0)
        self.assertEqual(env.camp_id, 1)
        self.assertEqual(env.agent_pos, env.camp_pos)
        self.assertFalse(np.array_equal(old_terrain, env.terrain))

    def test_patch_generation_is_deterministic_by_macro_coordinate(self):
        config = PatchEnvConfig(global_seed=5)
        env_a = HunterGathererPatchEnv(config)
        env_b = HunterGathererPatchEnv(config)

        env_a.reset(options={"macro_x": 20, "macro_y": -10})
        env_b.reset(options={"macro_x": 20, "macro_y": -10})

        np.testing.assert_array_equal(env_a.terrain, env_b.terrain)
        np.testing.assert_allclose(env_a.plant_food, env_b.plant_food)
        np.testing.assert_allclose(env_a.animal_density, env_b.animal_density)

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

    def test_non_water_terrain_is_clustered(self):
        env = HunterGathererPatchEnv(PatchEnvConfig(global_seed=10))
        env.reset(seed=11)

        agreement = self.terrain_neighbor_agreement(env.terrain)

        self.assertGreater(agreement, 0.65)

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


if __name__ == "__main__":
    unittest.main()
