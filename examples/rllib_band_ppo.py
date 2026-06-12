from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

from hunter_gatherers import RllibBandMemberPatchEnv

CONFIG_PATH = Path(__file__).with_name("pygame_viewer_config.toml")


def load_training_config(path: Path = CONFIG_PATH) -> dict:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return raw.get("training", {})


def main() -> None:
    training_cfg = load_training_config()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iterations",
        type=int,
        default=training_cfg.get("iterations", 1),
        help="Number of PPO training iterations.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=training_cfg.get("checkpoint_dir", None),
        help="Directory to save checkpoints during and after training.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=training_cfg.get("checkpoint_freq", 0),
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

    # Training-specific env overrides from config.
    # Everything else comes from patch_env_config.toml via PatchEnvConfig().
    env_overrides = {
        k: training_cfg[k]
        for k in (
            "members_per_band",
            "max_members_per_band",
            "max_steps",
        )
        if k in training_cfg
    }

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    config = (
        PPOConfig()
        .environment(
            env="hunter_gatherers_band",
            env_config={"patch_env_config": env_overrides},
        )
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: (
                "shared_policy"
            ),
        )
        .env_runners(
            num_env_runners=0,
            # Avoid RLlib cutting live multi-agent episodes at train-batch
            # boundaries. The new API stack may cache actions for terminal
            # agents just before a cut, then drop their zero-length
            # continuation episodes, which can surface as KeyError during
            # module-to-env unbatching.
            batch_mode="complete_episodes",
        )
        .training(
            train_batch_size=4000,  # ~4 episodes/update → stable gradients
            num_epochs=10,          # fewer passes/batch → less overfitting
            vf_clip_param=500.0,    # match reward scale (~5k-20k returns)
            entropy_coeff=0.01,     # keep some exploration pressure
        )
    )

    algo = config.build_algo()
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
