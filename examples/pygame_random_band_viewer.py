from __future__ import annotations

from hunter_gatherers import BandMemberPatchEnv, PatchEnvConfig

try:
    from examples.pygame_viewer import PygamePatchViewer, load_viewer_config
except ModuleNotFoundError:
    from pygame_viewer import PygamePatchViewer, load_viewer_config


def main() -> None:
    config = load_viewer_config()
    env = BandMemberPatchEnv(PatchEnvConfig())
    viewer = PygamePatchViewer(
        env,
        config,
        autoplay=True,
        random_policy_all_members=True,
    )
    viewer.run()


if __name__ == "__main__":
    main()
