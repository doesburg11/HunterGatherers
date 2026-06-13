from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hunter_gatherers import BandMemberPatchEnv

try:
    from examples.pygame_viewer import (
        PygamePatchViewer,
        load_env_config,
        load_viewer_config,
    )
except ModuleNotFoundError:
    from pygame_viewer import (  # type: ignore[no-redef]
        PygamePatchViewer,
        load_env_config,
        load_viewer_config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO policy in the pygame viewer.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to the RLlib checkpoint directory. "
            "Defaults to checkpoint_dir in pygame_viewer_config.toml."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override for the environment reset.",
    )
    args = parser.parse_args()

    viewer_config = load_viewer_config()
    if args.seed is not None:
        from dataclasses import replace
        viewer_config = replace(viewer_config, seed=args.seed)

    try:
        import ray
        from ray.rllib.algorithms.algorithm import Algorithm
        from ray.tune.registry import register_env
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "RLlib support requires Ray. Install it with: "
            "pip install 'hunter-gatherers[rllib]'"
        ) from exc

    from hunter_gatherers import RllibBandMemberPatchEnv

    try:
        import torch
        from ray.rllib.core.columns import Columns
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Evaluation requires PyTorch. Install it with: pip install torch"
        ) from exc

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    register_env(
        "hunter_gatherers_band",
        lambda config: RllibBandMemberPatchEnv(config),
    )
    checkpoint_dir = args.checkpoint or viewer_config.checkpoint_dir
    checkpoint_path = str(Path(checkpoint_dir).resolve())
    algo = Algorithm.from_checkpoint(checkpoint_path)
    female_module = algo.get_module("female_policy")
    male_module = algo.get_module("male_policy")
    shared_module = algo.get_module("shared_policy")
    default_module = algo.get_module("default_policy")
    single_policy_module = shared_module or default_module
    has_two_policy_checkpoint = (
        female_module is not None and male_module is not None
    )
    if not has_two_policy_checkpoint and single_policy_module is None:
        raise SystemExit(
            "Checkpoint is incompatible. Expected either both "
            "'female_policy'/'male_policy' or one of "
            "'shared_policy'/'default_policy'."
        )

    def _sex_policy_for_agent_id(agent_id: str) -> str:
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

    def policy_fn(
        obs_dict: dict[str, np.ndarray],
    ) -> dict[str, int]:
        if not obs_dict:
            return {}
        if not has_two_policy_checkpoint:
            if single_policy_module is None:
                raise RuntimeError(
                    "Missing single-policy module for legacy checkpoint."
                )
            agent_ids = list(obs_dict.keys())
            obs_batch = np.stack(
                [
                    obs_dict[aid].reshape(-1).astype(np.float32)
                    for aid in agent_ids
                ]
            )
            obs_tensor = torch.tensor(obs_batch)
            with torch.no_grad():
                result = single_policy_module.forward_inference(
                    {Columns.OBS: obs_tensor}
                )
            logits = result[Columns.ACTION_DIST_INPUTS].numpy()
            actions = np.argmax(logits, axis=-1)
            return {aid: int(actions[i]) for i, aid in enumerate(agent_ids)}

        if female_module is None or male_module is None:
            raise RuntimeError(
                "Missing female or male policy module "
                "for two-policy checkpoint."
            )

        actions_by_agent: dict[str, int] = {}
        for policy_name, module in (
            ("female_policy", female_module),
            ("male_policy", male_module),
        ):
            agent_ids = [
                aid
                for aid in obs_dict.keys()
                if _sex_policy_for_agent_id(aid) == policy_name
            ]
            if not agent_ids:
                continue
            obs_batch = np.stack(
                [
                    obs_dict[aid].reshape(-1).astype(np.float32)
                    for aid in agent_ids
                ]
            )
            obs_tensor = torch.tensor(obs_batch)
            with torch.no_grad():
                result = module.forward_inference({Columns.OBS: obs_tensor})
            logits = result[Columns.ACTION_DIST_INPUTS].numpy()
            policy_actions = np.argmax(logits, axis=-1)
            actions_by_agent.update(
                {
                    aid: int(policy_actions[i])
                    for i, aid in enumerate(agent_ids)
                }
            )
        return actions_by_agent

    env = BandMemberPatchEnv(load_env_config())
    viewer = PygamePatchViewer(
        env,
        viewer_config,
        autoplay=True,
        random_policy_all_members=True,
        policy_fn=policy_fn,
    )
    viewer.run()
    ray.shutdown()


if __name__ == "__main__":
    main()
