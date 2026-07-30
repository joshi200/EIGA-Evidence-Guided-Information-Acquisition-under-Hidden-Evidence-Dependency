import numpy as np

from eiga.environments import EnvironmentConfig, HiddenDependencyEnv, ScenarioType
from eiga.environments.agents import RandomAgent


def main() -> None:
    env = HiddenDependencyEnv(EnvironmentConfig())
    agent = RandomAgent(np.random.default_rng(7))
    completed = 0
    for seed in range(1000):
        obs, _ = env.reset(seed=seed, options={"scenario": ScenarioType.MIXED_DEPENDENCY})
        done = False
        while not done:
            action = agent.act(obs)
            obs, _, done, _, _ = env.step(action)
        completed += 1
    print(f"Completed {completed} random episodes without errors.")


if __name__ == "__main__":
    main()
