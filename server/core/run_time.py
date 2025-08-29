"""
main function flow
"""

import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Sequence, Literal, List

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
    Use LLM to analyze the query then route to: simple; mid; planner
    """
    category_prompt = f"""Quickly categoize the message into one of three options below:
                    Simple: This is the meesage that is straightforward and no-database-related;
                    Mid: This is the message that may need to consider several factors, and maybe database-related;
                    Planner: This is the meesage that requires detailed plan to achieve involve multiple aspects of consideration.
                    (Note: The output can be single word choosen from above)
                    """
llm = ChatOpenAI


def run_app():
    return