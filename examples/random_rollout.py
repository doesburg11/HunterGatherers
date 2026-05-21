from hunter_gatherers import HunterGathererPatchEnv


def main() -> None:
    env = HunterGathererPatchEnv()
    obs, info = env.reset(seed=123)
    total_reward = 0.0

    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(
        {
            "steps": info["step_count"],
            "total_reward": round(total_reward, 3),
            "energy": round(info["energy"], 3),
            "camp_potential_energy": round(info["camp_potential_energy"], 3),
            "season": info["season"],
            "camp_moves": info["num_camp_moves"],
            "macro": (info["macro_x"], info["macro_y"]),
        }
    )


if __name__ == "__main__":
    main()
