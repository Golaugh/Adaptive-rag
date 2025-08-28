"""
main function flow
"""

import os, dotenv, pathlib
from typing import TypedDict, Sequence, Literal, List

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, BaseMessage
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages: Sequence[BaseMessage, add_messages]
    status: Literal["simple", "mid", "planner"]
    

def query_analyzer(state: AgentState) -> AgentState:
    """
    Analyze the query then route to: simple; mid; planner
    """




def run_app():
    return