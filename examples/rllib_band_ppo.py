from __future__ import annotations

import argparse

from hunter_gatherers import RllibBandMemberPatchEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()

    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.tune.registry import register_env
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "RLlib support requires Ray. Install it with: "
            "pip install 'hunter-gatherers[rllib]'"
        ) from exc

    register_env(
        "hunter_gatherers_band",
        lambda config: RllibBandMemberPatchEnv(config),
    )

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    config = (
        PPOConfig()
        .environment(
            env="hunter_gatherers_band",
            env_config={
                "patch_env_config": {
                    "members_per_band": 3,
                    "max_steps": 100,
                },
            },
        )
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: (
                "shared_policy"
            ),
        )
        .env_runners(num_env_runners=0)
        .training(train_batch_size=200)
    )

    algo = config.build()
    try:
        for iteration in range(args.iterations):
            result = algo.train()
            print(
                {
                    "iteration": iteration + 1,
                    "episode_return_mean": result.get(
                        "episode_return_mean",
                    ),
                }
            )
    finally:
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
