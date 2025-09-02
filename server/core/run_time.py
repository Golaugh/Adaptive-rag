#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agentic Planner System runtime
"""

import logging
from pathlib import Path
from typing import Annotated, Sequence, TypedDict, Literal, Union, List

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

# Setup & Config
load_dotenv()
logger = logging.getLogger(__name__)

# Project root: .../server/
SERVER_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = SERVER_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# SQLite file for both app tables + LangGraph checkpoint
DB_PATH = str(DATA_DIR / "planner.db")

from db import DBManager
from core.nodes import (
    router_node, rewrite_node, analyze_node, planner_sys_node, 
    record_node, summarize_node, llm_node, tool_node
)

DB = DBManager(DB_PATH)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    summary: str
    route: Literal["direct", "normal", "planner"]
    user_id: Union[str, int]
    thread_id: str
    hitl_collected: List[str]
    hitl_needed: str
    hitl_round: int

graph = StateGraph(AgentState)
graph.add_node("router", router_node)
graph.add_node("rewrite", rewrite_node)
graph.add_node("analizer", analyze_node)
graph.add_node("planner_sys", planner_sys_node)
graph.add_node("tools", tool_node)
graph.add_node("recording", record_node) 
graph.add_node("llm", llm_node)
graph.add_node("summarize", summarize_node)

graph.add_edge(START, "router")
graph.add_edge(START, "db_related")
graph.add_conditional_edges("router", lambda s: s["route"], {
    "direct": "llm",
    "normal": "rewrite", 
    "planner": "advanced_search"
})
graph.add_edge("llm", "recording")
graph.add_edge("recording", "summarize")
graph.add_edge("rewrite", "analizer")
graph.add_edge("analizer", "planner_sys")
graph.add_edge("planner_sys", "recording")
graph.add_edge("advanced_search", "analizer")
graph.add_edge("summarize", END)

checkpointer = SqliteSaver.from_conn_string(DB_PATH)
app = graph.compile(checkpointer=checkpointer)


# Runtime (CLI)
def run_app():
    print("Agentic Planner System. Type 'exit' to quit.\n")

    # User selection
    while True:
        users = DB.list_users()
        if users:
            print("Available users:")
            for uid, uname in users:
                print(f"{uid}: {uname}")
        else:
            print("No users found.")

        choice = input("Enter user id, 'new <name>', 'del <id>': ").strip()
        if choice.startswith("new "):
            DB.create_user(choice[4:].strip())
            continue
        elif choice.startswith("del "):
            DB.delete_user(int(choice[4:].strip()))
            continue
        elif choice.isdigit():
            user_id = int(choice)
            break

    # Thread selection
    while True:
        threads = DB.list_threads(user_id)
        if threads:
            print("Available threads:")
            for tid, tname in threads:
                print(f"{tid}: {tname}")
        else:
            print("No threads found.")
        thread_id = input("Enter thread id (or new name): ").strip()
        if thread_id:
            break

    cfg = {"configurable": {"thread_id": thread_id}}
    while True:
        user_input = input("> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        # Initialize state with user_id and thread_id
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_id": user_id,
            "thread_id": thread_id
        }

        for event in app.stream(initial_state, cfg, stream_mode="updates"):
            if isinstance(event, dict):
                for node_name, payload in event.items():
                    if isinstance(payload, dict) and "messages" in payload:
                        for m in payload["messages"]:
                            if isinstance(m, AIMessage):
                                print(m.content)
            if "__interrupt__" in event:
                prompt = event["__interrupt__"][0].value
                print("\n[MORE INFO NEEDED]\n" + str(prompt))
                ans = input("(reply) > ")
                for _ in app.stream(Command(resume=ans), cfg, stream_mode="updates"):
                    pass
