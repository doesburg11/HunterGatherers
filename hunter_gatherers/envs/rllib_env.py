from __future__ import annotations

from typing import Any

from gymnasium import spaces

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
        return observations, infos

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
        result = self.env.step(action_dict)
        self._sync_from_wrapped_env()
        return result

    def render(self) -> str:
        return self.env.render()

    def get_observation_space(self, agent_id: str) -> spaces.Space:
        return self.observation_spaces[agent_id]

    def get_action_space(self, agent_id: str) -> spaces.Space:
        return self.action_spaces[agent_id]

    def _sync_from_wrapped_env(self) -> None:
        self.possible_agents = list(self.env.possible_agents)
        self.agents = list(self.env.agents)
        self.observation_spaces = dict(self.env.observation_spaces)
        self.action_spaces = dict(self.env.action_spaces)
        self.observation_space = spaces.Dict(self.observation_spaces)
        self.action_space = spaces.Dict(self.action_spaces)
