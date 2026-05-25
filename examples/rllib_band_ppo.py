from __future__ import annotations

import argparse
from pathlib import Path

from hunter_gatherers import RllibBandMemberPatchEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints during and after training.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=0,
        help=(
            "Save a checkpoint every N iterations. "
            "0 means only save at the end (requires --checkpoint-dir)."
        ),
    )
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
                    "members_per_band": 5,
                    "max_members_per_band": 10,
                    "max_steps": 500,
                    "use_physical_energetics": True,
                    "adult_body_mass_kg": 70.0,
                    "newborn_body_mass_kg": 12.0,
                    "initial_energy_per_kg": 30.0,
                    "max_energy_per_kg": 35.0,
                    "newborn_energy_per_kg": 30.0,
                    "cell_distance_km": 0.25,
                    "walking_kcal_per_kg_km": 1.0,
                    "food_energy_density_kcal_per_kg": 2500.0,
                    "water_liters_per_unit": 1.0,
                    "drink_amount": 1.0,
                    "plant_energy_capacity": 1000.0,
                    "plant_eat_amount": 400.0,
                    "plant_carry_share": 1.0,
                    "grass_energy_capacity": 1000.0,
                    "grass_regrowth_base": 0.25,
                    "animal_energy_capacity": 2500.0,
                    "initial_animals": 35,
                    "max_animals": 120,
                    "initial_animal_energy": 1200.0,
                    "max_animal_energy": 2500.0,
                    "animal_move_cost": 8.0,
                    "animal_eat_amount": 120.0,
                    "animal_reproduction_energy_threshold": 2000.0,
                    "animal_reproduction_cost": 900.0,
                    "newborn_animal_energy": 800.0,
                    "animal_hunt_amount": 800.0,
                    "animal_carry_share": 1.0,
                    "food_carry_capacity": 3000.0,
                    "water_carry_capacity": 5.0,
                    "water_collect_amount": 2.0,
                    "personal_energy_reserve": 2100.0,
                    "camp_food_withdraw_threshold": 1800.0,
                    "camp_food_withdraw_amount": 400.0,
                    "camp_water_withdraw_amount": 1.0,
                    "camp_storage_radius": 3,
                    "birth_rate": 0.1,
                    "birth_food_cost": 2500.0,
                    "birth_water_cost": 10.0,
                    "food_deposit_reward": 0.03,
                    "water_deposit_reward": 0.01,
                    "birth_reward": 2.0,
                    "population_reward": 0.001,
                    "juvenile_survival_reward": 0.002,
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
        .training(train_batch_size=1000)
    )

    algo = config.build()
    try:
        for iteration in range(args.iterations):
            result = algo.train()
            env_runner_results = result.get("env_runners", {})
            episode_return_mean = result.get(
                "episode_return_mean",
                env_runner_results.get("episode_return_mean"),
            )
            print(
                {
                    "iteration": iteration + 1,
                    "episode_return_mean": episode_return_mean,
                }
            )
            if (
                args.checkpoint_dir is not None
                and args.checkpoint_freq > 0
                and (iteration + 1) % args.checkpoint_freq == 0
            ):
                path = algo.save(
                    str(Path(args.checkpoint_dir).resolve())
                )
                print(f"Checkpoint saved to {path}")
        if args.checkpoint_dir is not None:
            path = algo.save(str(Path(args.checkpoint_dir).resolve()))
            print(f"Final checkpoint saved to {path}")
    finally:
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
