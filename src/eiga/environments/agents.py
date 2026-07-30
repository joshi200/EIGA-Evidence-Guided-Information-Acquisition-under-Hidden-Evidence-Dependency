from dataclasses import dataclass

import numpy as np

from .actions import ActionType, AgentAction, encode_action
from .hidden_dependency_env import HiddenDependencyEnv


@dataclass
class RandomAgent:
    rng: np.random.Generator

    def act(self, observation: dict) -> int:
        valid = np.flatnonzero(observation["action_mask"])
        return int(self.rng.choice(valid))


class OracleLineageAgent:
    """Simple benchmark agent; it intentionally uses privileged state."""

    def act(self, env: HiddenDependencyEnv, observation: dict) -> int:
        oracle = env.get_oracle_state()
        episode = env._require_episode()  # oracle-only baseline
        cfg = env.config
        queried = observation["queried_mask"]

        # Query one representative from every not-yet-seen lineage while affordable.
        seen = set(episode.tool_lineages[queried].tolist())
        for i, lineage in enumerate(episode.tool_lineages):
            if not queried[i] and int(lineage) not in seen and observation["action_mask"][i]:
                return encode_action(AgentAction(ActionType.QUERY, i), cfg.num_tools, 2)

        if queried.any():
            lineage_votes = {}
            for lineage in np.unique(episode.tool_lineages[queried]):
                idx = np.flatnonzero(queried & (episode.tool_lineages == lineage))[0]
                lineage_votes[int(lineage)] = int(episode.tool_claims[idx])
            counts = np.bincount(list(lineage_votes.values()), minlength=2)
            answer = int(np.argmax(counts))
        else:
            answer = int(oracle["true_answer"])
        return encode_action(AgentAction(ActionType.ANSWER, answer), cfg.num_tools, 2)
