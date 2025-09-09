#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agentic Planner System runtime
"""
import sys
import logging
from pathlib import Path
from typing import Annotated, Sequence, TypedDict, Literal, List

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DB_PATH, DB_CHECKPOINTER_PATH
from .db import DBManager
from .nodes import (
    router_node, router, rewrite_node, analyze_node, planner_sys_node, 
    record_node, should_summarize, summarize_node, llm_node, tool_node
)

load_dotenv()
DB = DBManager(DB_PATH)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    summary: str
    route: Literal["direct", "normal", "planner"]
    user_id: int
    thread_id: str
    hitl_collected: List[str]
    hitl_needed: str
    hitl_rounds: int


graph = StateGraph(AgentState)
graph.add_node("router_node", router_node)
graph.add_node("rewrite", rewrite_node)
graph.add_node("analyzer", analyze_node)
graph.add_node("planner_sys", planner_sys_node)
graph.add_node("record", record_node) 
graph.add_node("summarize", summarize_node)
graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)

graph.add_conditional_edges(START, should_summarize, {"summarize": "summarize", "router_node": "router_node"})
graph.add_edge("summarize", "record")
graph.add_edge("record", "router_node")
graph.add_conditional_edges("router_node", router, {
    "direct": "llm",
    "normal": "rewrite", 
    "planner": "analyzer"
})
graph.add_edge("rewrite", "llm")
graph.add_edge("tools", "llm")
graph.add_edge("llm", END)

# Analyzer will use HITL, head to analizer again or planner_sys(when info is enough)
graph.add_edge("planner_sys", END)


def run_app():
    print("\nAgentic Planner System activated successfully. [Type 'exit' or 'quit' to quit]\n")

    # user selection
    while True:
        users = DB.list_users()
        if users:
            print("Available users:")
            for uid, uname in users:
                print(f"  {uid}: {uname}")
        else:
            print("No users found.")

        choice = input("Enter username | 'new <name>' | 'del <id>' | '<id>': ").strip()
        if not choice:
            continue
        if choice.lower() in {"exit", "quit"}:
            return

        if choice.lower().startswith("new "):
            name = choice[4:].strip()
            if not name:
                print("Username cannot be empty.")
                continue
            user_id = DB.ensure_user(name)
            print(f"User ready: {name} (id={user_id})")
            break

        if choice.lower().startswith("del "):
            try:
                del_id = int(choice[4:].strip())
                DB.delete_user(del_id)
                print(f"Deleted user id={del_id}")
            except ValueError:
                print("Usage: del <id>")
            continue

        if choice.isdigit():
            user_id = int(choice)
            print(f"Selected user id={user_id}")
            break

        user_id = DB.ensure_user(choice)
        print(f"Selected user '{choice}' (id={user_id})")
        break

    # thread selection
    while True:
        threads = DB.list_threads(user_id)
        if threads:
            print("Available threads:")
            for pk, tname in threads:
                print(f"  {tname}")
        else:
            print("No threads found for this user.")

        t_choice = input("Enter existing thread_id | 'new <thread_id>' | 'del <thread_id>': ").strip()
        if not t_choice:
            continue
        if t_choice.lower() in {"exit", "quit"}:
            return

        if t_choice.lower().startswith("del "):
            t_del = t_choice[4:].strip()
            if not t_del:
                print("Usage: del <thread_id>")
                continue
            DB.delete_thread(user_id, t_del)
            print(f"Deleted thread '{t_del}'")
            continue

        if t_choice.lower().startswith("new "):
            thread_id = t_choice[4:].strip()
            if not thread_id:
                print("Usage: new <thread_id>")
                continue
            DB.create_thread(user_id, thread_id)
            print(f"Thread ready: '{thread_id}'")
            break

        thread_id = t_choice
        DB.create_thread(user_id, thread_id)
        print(f"Using thread: '{thread_id}'")
        break

    maybe_cm = SqliteSaver.from_conn_string(str(DB_CHECKPOINTER_PATH))
    if hasattr(maybe_cm, "__enter__") and hasattr(maybe_cm, "__exit__"):
        with maybe_cm as checkpointer:
            app = graph.compile(checkpointer=checkpointer)
            _run_cli_loop(app, user_id, thread_id)
    else:
        checkpointer = maybe_cm
        app = graph.compile(checkpointer=checkpointer)
        _run_cli_loop(app, user_id, thread_id)


def _run_cli_loop(app, user_id: int, thread_id: str):
    cfg = {"configurable": {"thread_id": f"{user_id}:{thread_id}"}}
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if user_input.lower() in {"exit", "quit"}:
            print("Bye.")
            return
        if not user_input:
            continue

        initial_state = {
                    "messages": [HumanMessage(content=user_input)],
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "summary": "",
                    "route": "",
                    "hitl_collected": [],
                    "hitl_needed": "",
                    "hitl_rounds": 0,
                }

        while True:
            interrupted = False
            for event in app.stream(initial_state, cfg, stream_mode="updates"):
                if isinstance(event, dict):
                    for node_name, payload in event.items():
                        if isinstance(payload, dict) and "messages" in payload:
                            for m in payload["messages"]:
                                if isinstance(m, AIMessage):
                                    print(m.content)

                    if "__interrupt__" in event:
                        interrupted = True
                        prompt = event["__interrupt__"][0].value
                        print("\n[MORE INFO NEEDED]\n" + str(prompt))
                        ans = input("(reply) > ")
                        initial_state = Command(resume=ans)
                        break

            if not interrupted:
                break
