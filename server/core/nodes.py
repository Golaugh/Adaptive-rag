

import os
import json
import logging
from typing import Annotated, TypedDict, Sequence, Literal, Union, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage
from langgraph.graph.message import BaseMessage, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command

logger = logging.getLogger(__name__)

# Get DB instance
from pathlib import Path
from db import DBManager
SERVER_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = SERVER_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = str(DATA_DIR / "planner.db")
DB = DBManager(DB_PATH)

SYSTEM_DIRECT = SystemMessage(content="You are a consice, helpful assistant. Use the conversation summary to maintain continuity.")
LLM = ChatOpenAI(model=os.environ("MODEL"))
SIDE_LLM = ChatOpenAI(model=os.environ("SIDE_MODEL"), temperature=0)
SUMMARIZER = ChatOpenAI(model=os.environ("MODEL"))
status = ("direct", "normal", "planner")
MAX_ROUNDS = 3
RECENT_K = 6
KEEP_RECENT = 4
SUMMARIZE_AFTER = 8

from utils.search import web_search
TOOLS = [web_search]
tool_node = ToolNode(TOOLS)
LLM_DIRECT = LLM.bind_tools(TOOLS)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    summary: str
    route: Literal["direct", "normal", "planner"]
    user_id: Union[str, int]
    thread_id: str
    hitl_collected: List[str]
    hitl_needed: str
    hitl_round: int


def router_node(state: AgentState) -> dict:
    """change state['route'] attribute for the current state of messages"""

    last_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    sys = SystemMessage(content=("""Quickly categoize the latest message into one of three options below:
                    direct: This is the meesage that is straightforward;
                    normal: This is the message that may need to consider several factors or the message itself is too vague;
                    planner: This is the meesage that requires detailed plan to achieve, and it involves multiple aspects of consideration.
                    (Note: The output can only be one single word choosen from above)
                    """))
    res = SIDE_LLM.invoke(sys + [last_user.content])

    while res.content not in status:
        logger.error(f"Wrong router response: {res}, trying again...")
        res = SIDE_LLM.invoke(sys + [last_user.content])

    logger.info(f"Current query route to: {res.content}")
    return {"route": res.content}


def router(state: AgentState) -> str:
    """route the query to different paths"""
    return state["route"]


def rewrite_node(state: AgentState) -> dict:
    """Rewrite query with context from local_info if available"""

    last_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    local_info = DB.get_local_info(user_id=state["user_id"], thread_id=state["thread_id"])
    sys = SystemMessage(content=(f"""Expand this query to make it more reasonable and understandable base on the personal_info
                                 [if no info provided then use normal logic]: {local_info}\n And here's the query: {last_user.content}"""))
    
    res = SIDE_LLM.invoke([sys])
    logger.info(f"Current query rewrited from: {last_user.content} \n\nTo: {res}")
    return {"messages": [RemoveMessage(id=last_user.id), HumanMessage(content=res)]}


def analyze_node(state: AgentState) -> dict:
    """Analyze query sufficiency with Human-in-the-Loop"""

    if state.get("route") != "planner":
        logger.error(f"wrong route to this node, current state['route']={state['route']}")
        return {}

    last_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    query_text = (last_user.content if last_user else "").strip()
    factors = state.get("hitl_needed") or ""
    if not factors:
        from utils.search import advan_web_search
        factors = advan_web_search(query_text)
        return Command(goto="analizer", update={"hitl_needed": factors})
    
    collected = list(state.get("hitl_collected") or [])
    rounds = int(state.get("hitl_rounds") or 0)
    sys = SystemMessage(content=(
        "You are an analyst for a planning agent. "
        "Given the user's query, internet factors, and previously provided info, "
        "ASK ONLY for the STILL-MISSING critical fields needed to plan "
        "(e.g., deadline YYYY-MM-DD, budget, team_size, scope, risks). "
        "Output only the request text."
        f"\n\nUser Query:\n{query_text}"
        f"\n\nAdditional Factors (from web):\n{factors}"
        f"\n\nAlready Provided:\n{chr(10).join(map(str, collected)) if collected else '(none)'}\n"
    ))

    request_text = LLM.invoke([sys]).content.strip()
    user_feedback = interrupt(request_text)
    collected.append(str(user_feedback))
    rounds += 1

    def quick_judge(user_input: str, system_request: str) -> bool:
        judge_sys = SystemMessage(content=(
            "Return strictly 'True' or 'False' (no punctuation). "
            "Answer True iff the user_input completely satisfies the fields requested by system_request."
            f"\n\nuser_input:\n{user_input}\n\nsystem_request:\n{system_request}\n"
        ))
        for _ in range(3):
            try:
                resp = SIDE_LLM.invoke([judge_sys]).content.strip()
                if resp in ("True", "False"):
                    return resp == "True"
            except Exception as e:
                logger.warning(f"quick_judge LLM error: {e}")
        return False
    
    ok = quick_judge("\n".join(collected), request_text)
    if ok or rounds >= MAX_ROUNDS:
        updated_last_user = [last_user.content] + [*collected]
        return Command(
            goto="planner_sys",
            update={
                "hitl_collected": collected,
                "hitl_needed": factors,
                "hitl_rounds": rounds,
                "messages": [RemoveMessage(id=last_user.id), 
                             HumanMessage(content=updated_last_user),
                             AIMessage(content=("Info is enough, now proceed to planner node..."))],
            },
        )
    
    return Command(
        goto="analizer",
        update={
            "hitl_collected": collected,
            "hitl_needed": factors,
            "hitl_rounds": rounds,
            "messages": [AIMessage(content=("Info is still not enough, now entering analizer node again..."))]
        },
    )


def planner_sys_node(state: AgentState) -> dict:
    """Planner system node"""
    from core.planner import Planner
    planner = Planner(state)
    res = planner.handle
    return {"messages": [AIMessage(content=res.content)]}


def record_node(state: AgentState) -> dict:
    """Record objective and emotional info to local_info table"""
    
    recent_msgs = list(state["messages"])[-RECENT_K:]
    joined = "\n".join([getattr(m, "content", "") for m in recent_msgs])
    
    # Extract objective and emotional info (simplified)
    obj_info = f"Objective: {joined[:100]}..."
    emo_info = f"Emotional: neutral tone detected"
    
    # Get user_id and thread_id from state
    user_id = state.get("user_id", 1)
    thread_id = state.get("thread_id", "default")
    
    # Update database
    DB.update_local_info(user_id, thread_id, obj_info, emo_info)
    
    return {"messages": [HumanMessage(content="[Recorded] Information saved to database")]}


def should_summarize(state: AgentState) -> str:
    if len(state["messages"]) >= SUMMARIZE_AFTER:
        logger.info("Detect long context, summarizing...")
        return "summarize"
    return "router_node"


def summarize_node(state: AgentState) -> dict:
    """produce updated summary"""
    previous_summary = state.get("summary", "")
    recent = list(state["messages"])[-RECENT_K:]

    sys = [SystemMessage(content=("You are a memory condenser. Given the prior running summary and the latest conversation slice,"
                                    "produce a concise, factual summary that preserves names, goals, decisions and open todos."))]

    if previous_summary:
        sys.append(SystemMessage(content=f"[previous summary]\n{previous_summary}"))

    joined_recent = [m.content for m in recent]
    sys.append(SystemMessage(content="[Recent slice]\n" + "\n".join(joined_recent)))

    new_summary = SUMMARIZER.invoke(sys).content.strip()
    return {"summary": new_summary, "messages": list(state["messages"])[-KEEP_RECENT:]}


def llm_node(state: AgentState) -> dict:
    """direct llm response"""

    sys = SystemMessage(content=("You are a consice, helpful assistant. Use the conversation summary to maintain continuity. "
                                 "Feel free to use web_search tool if you think the information you search is time-sensitive."
                                    "If user asking about past context that isn't in RECENT messages, use the summary to recall it. "))
    context: list[BaseMessage] = [sys]
    if state.get("summary"):
        context.append(SystemMessage(content=f"[Conversation summary]\n{state['summary']}"))

    recent = list(state["messages"])[-RECENT_K:]
    context.extend([m for m in recent])
    
    res = LLM.invoke(context)
    return {"messages": [res]}