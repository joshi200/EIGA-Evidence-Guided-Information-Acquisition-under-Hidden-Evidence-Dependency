from eiga.environments.actions import ActionType, AgentAction, decode_action, encode_action


def test_action_roundtrip():
    n, c = 6, 2
    actions = [
        AgentAction(ActionType.QUERY, 3),
        AgentAction(ActionType.DIAGNOSE, 2),
        AgentAction(ActionType.ANSWER, 1),
        AgentAction(ActionType.ABSTAIN),
    ]
    for action in actions:
        assert decode_action(encode_action(action, n, c), n, c) == action
