from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    QUERY = "query"
    DIAGNOSE = "diagnose"
    ANSWER = "answer"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class AgentAction:
    action_type: ActionType
    target: int | None = None


def decode_action(action_index: int, num_tools: int, num_answers: int) -> AgentAction:
    if 0 <= action_index < num_tools:
        return AgentAction(ActionType.QUERY, action_index)
    if num_tools <= action_index < 2 * num_tools:
        return AgentAction(ActionType.DIAGNOSE, action_index - num_tools)
    answer_start = 2 * num_tools
    answer_end = answer_start + num_answers
    if answer_start <= action_index < answer_end:
        return AgentAction(ActionType.ANSWER, action_index - answer_start)
    if action_index == answer_end:
        return AgentAction(ActionType.ABSTAIN)
    raise ValueError(f"invalid action index: {action_index}")


def encode_action(action: AgentAction, num_tools: int, num_answers: int) -> int:
    if action.action_type is ActionType.QUERY:
        if action.target is None or not 0 <= action.target < num_tools:
            raise ValueError("query target is invalid")
        return action.target
    if action.action_type is ActionType.DIAGNOSE:
        if action.target is None or not 0 <= action.target < num_tools:
            raise ValueError("diagnostic target is invalid")
        return num_tools + action.target
    if action.action_type is ActionType.ANSWER:
        if action.target is None or not 0 <= action.target < num_answers:
            raise ValueError("answer target is invalid")
        return 2 * num_tools + action.target
    if action.action_type is ActionType.ABSTAIN:
        return 2 * num_tools + num_answers
    raise ValueError(f"unsupported action: {action}")
