

from typing import Annotated, TypedDict, Sequence, Literal, Union, List
from langgraph.graph.message import BaseMessage, add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    summary: str
    route: Literal["direct", "normal", "planner"]
    user_id: Union[str, int]
    thread_id: str
    hitl_collected: List[str]
    hitl_needed: str
    hitl_round: int

class Planner:

    def __init__(self, state: AgentState):
        self.state = state

    def handle(query: str) -> dict:
        """take query, activate comments analysis, return dict (include timeline + content)"""
        return {}