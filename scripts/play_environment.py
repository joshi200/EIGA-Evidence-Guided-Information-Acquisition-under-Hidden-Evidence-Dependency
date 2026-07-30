from eiga.environments import EnvironmentConfig, HiddenDependencyEnv, ScenarioType


def main() -> None:
    env = HiddenDependencyEnv(EnvironmentConfig())
    observation, info = env.reset(
        seed=42, options={"scenario": ScenarioType.MINORITY_INDEPENDENT_TRUTH}
    )
    terminated = False
    print("Action layout: queries 0..N-1, diagnostics N..2N-1, answers 2N..2N+1, abstain last")
    while not terminated:
        print("\nInfo:", info)
        print("Claims:", observation["tool_claims"])
        print("Metadata:", observation["metadata_signatures"])
        print("Valid actions:", list(observation["action_mask"].nonzero()[0]))
        action = int(input("Action: "))
        observation, reward, terminated, _, info = env.step(action)
        print("Reward:", reward)
    print("\nOracle terminal summary:", info["oracle"])


if __name__ == "__main__":
    main()
