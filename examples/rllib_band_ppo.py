from __future__ import annotations

import argparse
from collections.abc import Hashable
from pathlib import Path
import tomllib

from hunter_gatherers import RllibBandMemberPatchEnv

CONFIG_PATH = Path(__file__).with_name("pygame_viewer_config.toml")


def load_raw_config(path: Path = CONFIG_PATH) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_training_config(path: Path = CONFIG_PATH) -> dict:
    raw = load_raw_config(path)
    return raw.get("training", {})


def load_environment_config(path: Path = CONFIG_PATH) -> dict:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return raw.get("environment", {})


def _sex_policy_for_agent_id(agent_id: Hashable) -> str:
    # Agent ids are formatted as: band_<band_id>_member_<member_identity>
    if not isinstance(agent_id, str):
        return "female_policy"
    parts = agent_id.split("_")
    if len(parts) != 4:
        return "female_policy"
    try:
        band_id = int(parts[1])
        member_identity = int(parts[3])
    except ValueError:
        return "female_policy"
    sex = (band_id + member_identity) % 2
    return "female_policy" if sex == 0 else "male_policy"


def main() -> None:
    training_cfg = load_training_config()
    environment_cfg = load_environment_config()

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
    parser.add_argument(
        "--print-dol-stats",
        action=argparse.BooleanOptionalAction,
        default=training_cfg.get("print_dol_stats", True),
        help=(
            "Print division-of-labor stats during training. "
            "Enabled by default; use --no-print-dol-stats to disable."
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

    # Base environment comes from [environment] in pygame_viewer_config.toml
    # so experiment settings (e.g., sexual reproduction) match viewer/eval.
    env_overrides = dict(environment_cfg)

    # Training-only overrides can still be provided in [training].
    for key in ("members_per_band", "max_members_per_band", "max_steps"):
        if key in training_cfg:
            env_overrides[key] = training_cfg[key]

    # Pass print_dol_stats flag to environment.
    env_overrides["print_dol_stats"] = args.print_dol_stats

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    config = (
        PPOConfig()
        .environment(
            env="hunter_gatherers_band",
            env_config={"patch_env_config": env_overrides},
        )
        .multi_agent(
            policies={"female_policy", "male_policy"},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: (
                _sex_policy_for_agent_id(agent_id)
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
