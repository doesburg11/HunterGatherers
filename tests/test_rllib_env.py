import importlib.util
import unittest

from hunter_gatherers import PatchEnvConfig, RllibBandMemberPatchEnv
from hunter_gatherers.envs.patch_env import Action


HAS_RAY = importlib.util.find_spec("ray") is not None


class RllibBandMemberPatchEnvTest(unittest.TestCase):
    @unittest.skipIf(HAS_RAY, "Ray is installed; missing dependency path unused")
    def test_constructor_explains_missing_rllib_dependency(self):
        with self.assertRaisesRegex(RuntimeError, "hunter-gatherers\\[rllib\\]"):
            RllibBandMemberPatchEnv(PatchEnvConfig(members_per_band=2))

    @unittest.skipUnless(HAS_RAY, "Ray is not installed")
    def test_reset_and_step_match_multi_agent_shape(self):
        from ray.rllib.env.multi_agent_env import MultiAgentEnv

        env = RllibBandMemberPatchEnv(
            {
                "patch_env_config": {
                    "members_per_band": 2,
                }
            }
        )

        self.assertIsInstance(env, MultiAgentEnv)
        observations, infos = env.reset(seed=31)
        self.assertEqual(set(observations), set(env.agents))
        self.assertEqual(set(infos), set(env.agents))
        self.assertTrue(set(env.agents).issubset(set(env.possible_agents)))
        for observation in observations.values():
            self.assertEqual(observation.shape, (15 * 11 * 11,))

        observations, rewards, terminateds, truncateds, infos = env.step(
            {"band_0_member_0": Action.STAY}
        )

        self.assertTrue(set(rewards).issubset(set(env.possible_agents)))
        self.assertIn("__all__", terminateds)
        self.assertIn("__all__", truncateds)
        self.assertEqual(set(observations), set(env.agents))
        self.assertTrue(set(infos).issubset(set(env.possible_agents)))


if __name__ == "__main__":
    unittest.main()
