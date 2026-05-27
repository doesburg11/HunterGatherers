from __future__ import annotations

from typing import Any

from gymnasium import spaces
import numpy as np

from hunter_gatherers.envs.patch_env import BandMemberPatchEnv, PatchEnvConfig


try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ModuleNotFoundError:
    MultiAgentEnv = None  # type: ignore[assignment]


def _missing_rllib_error() -> RuntimeError:
    return RuntimeError(
        "RLlib support requires Ray. Install it with: "
        "pip install 'hunter-gatherers[rllib]'"
    )


def _parse_env_config(
    config: PatchEnvConfig | dict[str, Any] | None,
) -> tuple[PatchEnvConfig | dict[str, Any] | None, int]:
    if config is None or isinstance(config, PatchEnvConfig):
        return config, 0

    raw_config = dict(config)
    controlled_band_id = int(raw_config.pop("controlled_band_id", 0))
    patch_config = raw_config.pop("patch_env_config", raw_config)
    return patch_config, controlled_band_id


if MultiAgentEnv is None:
    _RllibBase = object
else:
    _RllibBase = MultiAgentEnv


class RllibBandMemberPatchEnv(_RllibBase):  # type: ignore[misc, valid-type]
    """RLlib adapter for the shared-patch band member environment."""

    def __init__(
        self,
        config: PatchEnvConfig | dict[str, Any] | None = None,
    ):
        if MultiAgentEnv is None:
            raise _missing_rllib_error()

        super().__init__()
        patch_config, controlled_band_id = _parse_env_config(config)
        self.controlled_band_id = controlled_band_id
        self.env = BandMemberPatchEnv(
            patch_config,
            controlled_band_id=controlled_band_id,
        )
        self._sync_from_wrapped_env()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        observations, infos = self.env.reset(seed=seed, options=options)
        self._sync_from_wrapped_env()
        return self._flatten_observations(observations), infos

    def step(
        self,
        action_dict: dict[str, int],
    ) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        observations, rewards, terminateds, truncateds, infos = self.env.step(
            action_dict
        )
        self._sync_from_wrapped_env()
        return (
            self._flatten_observations(observations),
            rewards,
            terminateds,
            truncateds,
            infos,
        )

    def render(self) -> str:
        return self.env.render()

    def get_observation_space(self, agent_id: str) -> spaces.Space:
        return self.observation_spaces[agent_id]

    def get_action_space(self, agent_id: str) -> spaces.Space:
        return self.action_spaces[agent_id]

    def _sync_from_wrapped_env(self) -> None:
        self.possible_agents = self._possible_agent_ids()
        self.agents = list(self.env.agents)
        observation_space = self._flatten_space(self.env.observation_space)
        self.observation_spaces = {
            agent_id: observation_space for agent_id in self.possible_agents
        }
        self.action_spaces = {
            agent_id: self.env.action_space
            for agent_id in self.possible_agents
        }
        self.observation_space = spaces.Dict(self.observation_spaces)
        self.action_space = spaces.Dict(self.action_spaces)

    def _possible_agent_ids(self) -> list[str]:
        max_births = self.env.config.max_steps
        max_member_identity = self.env.member_capacity_per_band + max_births
        return [
            f"band_{self.controlled_band_id}_member_{member_id}"
            for member_id in range(max_member_identity)
        ]

    @staticmethod
    def _flatten_space(observation_space: spaces.Box) -> spaces.Box:
        return spaces.Box(
            low=observation_space.low.reshape(-1),
            high=observation_space.high.reshape(-1),
            dtype=observation_space.dtype,
        )

    @staticmethod
    def _flatten_observations(
        observations: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        return {
            agent_id: np.asarray(observation, dtype=np.float32).reshape(-1)
            for agent_id, observation in observations.items()
        }
