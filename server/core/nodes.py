

import sys
import json
from uuid import uuid4
from pathlib import Path
import logging
from typing import Annotated, TypedDict, Sequence, Literal, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, ToolMessage, AIMessage
from langgraph.graph.message import BaseMessage, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DB_PATH, MODEL, SIDE_MODEL, MAX_ROUNDS, RECENT_K, KEEP_RECENT, SUMMARIZE_AFTER
from utils.search import web_search
from utils.retrieve import db_retrieve
from .db import DBManager

TOOLS = [web_search, db_retrieve]
tool_node = ToolNode(TOOLS)
DB = DBManager(str(DB_PATH))
LLM = ChatOpenAI(model=MODEL)
SIDE_LLM = ChatOpenAI(model=SIDE_MODEL, temperature=0)
SUMMARIZER = ChatOpenAI(model=SIDE_MODEL)
STATUS = ("direct", "normal", "planner")

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


def router_node(state: AgentState) -> dict:
    """change state['route'] attribute for the current state of messages"""

    last_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not last_user:
        logger.info("No previous user input detected! Route to direct...")
        return Command(goto="llm")

    sys = SystemMessage(content=("""Quickly categorize the latest message into one of three options below:
                    direct: This is the message that is straightforward and easy to answer;
                    normal: This is the message that may need to consider several factors or the message itself is too vague;
                    planner: This is the message that requires detailed plan to achieve, and it involves multiple aspects of consideration.
                    (Note: The output can only be one single word choosen from above)
                    """))
    res = SIDE_LLM.invoke([sys, HumanMessage(content=last_user.content)])

    while res.content not in STATUS:
        logger.error(f"Wrong router response: {res}, trying again...")
        res = SIDE_LLM.invoke([sys, HumanMessage(content=last_user.content)])

    logger.info(f"Current query route to: {res.content}")
    return {"route": res.content}


def router(state: AgentState) -> str:
    """directly return message category"""
    return state["route"]


def rewrite_node(state: AgentState) -> dict:
    """Rewrite query with context from local_info if available"""

    last_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not last_user:
        logger.info("No previous user input detected! Route to direct...")
        return Command(goto="llm")

    local_info = DB.get_local_info(user_id=state["user_id"], thread_id=state["thread_id"])
    if local_info:
        obj_info, emo_info = local_info.get("obj_info", ""), local_info.get("emo_info", "")
        info_text = f"\n[objective]\n{obj_info}\n\n[emotional]\n{emo_info}\n"
    else:
        info_text = ""

    sys = SystemMessage(content=(
        "Rewrite the user's last message into a clearer, actionable, stand-alone query for retrieval/answering.\n"
        "Rules:\n"
        "1) Preserve the original intent and the original language; do not invent new facts.\n"
        "2) Use personal_info only to disambiguate; do not change the user's intent.\n"
        "3) Do NOT address the assistant (no '你/您/请问'); do NOT use first-person ('我/我们').\n"
        "4) Prefer declarative/imperative form (e.g., '检索…/提供…/总结…') rather than conversational questions.\n"
        "5) No greetings, no follow-up questions, no politeness, no explanations, no summaries.\n"
        "6) OUTPUT MUST BE ONLY the rewritten query text, one line, without quotes or extra words.\n"
        "7) If no rewrite is needed, output the original text exactly.\n\n"
        f"personal_info:\n{info_text}"
        f"user_query:\n{last_user.content}"
    ))

    res = SIDE_LLM.invoke([sys])
    rewritten = (res.content or "").strip().strip('"').strip("“”").strip()

    logger.info(f"Current query rewritten to: {rewritten}")

    if getattr(last_user, "id", None):
        return {"messages": [HumanMessage(id=last_user.id, content=rewritten)]}
    else:
        return {"messages": [RemoveMessage(id=getattr(last_user, "id", None)),
                             HumanMessage(content=rewritten, id=str(uuid4()))]}



def analyze_node(state: AgentState) -> Command:
    """Analyze query sufficiency with Human-in-the-Loop"""

    if state.get("route") != "planner":
        logger.error(f"wrong route to this node, current state['route']={state.get('route')}")
        return Command(goto="llm",)

    last_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    query_text = (last_user.content if last_user else "").strip()

    factors = state.get("hitl_needed") or ""
    if not factors:
        sys = SystemMessage(content=("You are a bilingual translator; input is a user query; output EXACTLY '<Chinese translation> ### <English translation>'; "
                                     f"preserve meaning; no extra text/explanations. \n\nThe query: \n{query_text}"))
        res = SIDE_LLM.invoke([sys])
        parts = [p.strip() for p in res.content.split("###", 1)]
        if len(parts) == 2:
            cn_query, en_query = parts[0], parts[1]
        else:
            cn_query = en_query = query_text

        from utils.search import advan_web_search
        factors_json = advan_web_search.invoke({"cn_query": cn_query, "en_query": en_query})
        extract_sys = SystemMessage(content=("You are a rigorous cross-source summarizer. "
                                            "Your task is to extract ONLY the key, non-duplicated factors related to the user query "
                                            "from two sources in the JSON below: `factors_from_zhihu` and `factors_from_reddit`.\n\n"

                                            "Data details:\n"
                                            "- factors_from_zhihu: Each key is the text of an answer. Its value is a list of comments "
                                            "(with `content`, `created_time`, `vote_count`). Higher `vote_count` = more credible.\n"
                                            "- factors_from_reddit: Each item is a dict. Items with the same `title` belong to one answer. "
                                            "  • type=`post`: the main answer text\n"
                                            "  • type=`comment`: a comment to the post\n"
                                            "  • type=`reply`: a reply to a comment\n"
                                            "Higher `score` = more credible.\n\n"

                                            "Guidelines:\n"
                                            "- Focus strictly on factors relevant to the user query.\n"
                                            "- Remove duplicates across and within sources.\n"
                                            "- Favor information with higher `vote_count` (Zhihu) or higher `score` (Reddit).\n"
                                            "- Summarize clearly and concisely.\n\n"

                                            f"User query:\n{query_text}\n\n"
                                            f"JSON to analyze:\n{factors_json}\n"
                                        ))
        res = SUMMARIZER.invoke([extract_sys])
        return Command(goto="analyzer", update={"hitl_needed": res.content})
    
    collected = list(state.get("hitl_collected") or [])
    rounds = int(state.get("hitl_rounds") or 0)
    sys = SystemMessage(content=("You are an analyst for a planning agent. "
                                "Given the user's query, internet factors, and previously provided info, "
                                "ASK ONLY for the STILL-MISSING critical fields needed to plan "
                                "(e.g., deadline YYYY-MM-DD, budget, team_size, scope, risks). "
                                "Output only the request text.(use the same language as query_text)"
                                f"\n\nUser Query:\n{query_text}"
                                f"\n\nAdditional Factors (from web):\n{factors}"
                                f"\n\nAlready Provided:\n{chr(10).join(map(str, collected)) if collected else '(none)'}\n"
                            ))

    request_text = LLM.invoke([sys]).content.strip()
    user_feedback = interrupt(request_text)
    collected.append(str(user_feedback))
    rounds += 1

    def quick_judge(user_input: str, system_request: str) -> bool:
        judge_sys = SystemMessage(content=("Return strictly 'True' or 'False' (no punctuation)(use the same language as user_input). "
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
                "hitl_collected": "",
                "hitl_needed": "",
                "hitl_rounds": 0,
                "messages": [RemoveMessage(id=last_user.id), 
                             HumanMessage(content="\n".join(updated_last_user), id=str(uuid4())),
                             SystemMessage(content=("Info is enough, now proceed to planner node..."))],
            },
        )
    
    return Command(
        goto="analyzer",
        update={
            "hitl_collected": collected,
            "hitl_needed": factors,
            "hitl_rounds": rounds,
        },
    )


def planner_sys_node(state: AgentState) -> dict:
    """Planner system node"""
    from .planner import PlannerHandle
    planner = PlannerHandle(state)
    res = planner.handle()
    text = res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)
    return {"messages": [AIMessage(content=text)]}


def record_node(state: AgentState) -> dict:
    """Record objective and emotional info to local_info table"""
    
    prev_summary = state["summary"]
    recent_msgs = list(m.content for m in state["messages"] if isinstance(m, HumanMessage) or isinstance(m, AIMessage))
    joined = "\n".join(recent_msgs)
    sys = SystemMessage(content=("Analyze the conversation and return ONLY a JSON object with two keys:\n"
                                "  - objective: facts/preferences/constraints\n"
                                "  - emotional: mindset/tone/concerns\n"
                                "No prose outside JSON.\n\n"
                                f"Conversation:\n{joined}\n\n"
                                f"Previous summary:\n{prev_summary}"
                            ))
    res = SUMMARIZER.invoke([sys])
    user_id = state.get("user_id", 1)
    thread_id = state.get("thread_id", "default")

    obj_info = emo_info = ""
    try:
        data = json.loads(res.content)
        obj_info = (data.get("objective") or "").strip()
        emo_info = (data.get("emotional") or "").strip()
    except Exception:
        parts = str(res.content).split("###", 1)
        if len(parts) == 2:
            obj_info, emo_info = parts[0].strip(), parts[1].strip()
        else:
            obj_info, emo_info = str(res).strip(), ""

    DB.update_local_info(user_id, thread_id, obj_info, emo_info)
    return {"messages": []}


def should_summarize(state: AgentState) -> str:
    if len(state["messages"]) >= SUMMARIZE_AFTER:
        logger.info("Long context optimizing...")
        return "summarize"
    return "router_node"


def summarize_node(state: AgentState) -> dict:
    """produce updated summary"""
    previous_summary = state.get("summary", "")
    recent = list(state["messages"])[-RECENT_K:]

    sys = [SystemMessage(content=(
        "You are a memory condenser. Given the prior running summary and the latest conversation slice,"
        "produce a concise, factual summary that preserves useful information."
    ))]
    if previous_summary:
        sys.append(SystemMessage(content=f"[previous summary]\n{previous_summary}"))

    joined_recent = [m.content for m in recent]
    sys.append(SystemMessage(content="[Recent slice]\n" + "\n".join(joined_recent)))

    new_summary = SUMMARIZER.invoke(sys).content.strip()
    msgs = list(state["messages"])
    to_remove = msgs[:-KEEP_RECENT]
    removals = [RemoveMessage(id=m.id) for m in to_remove if getattr(m, "id", None)]

    return {"summary": new_summary, "messages": removals}


def llm_node(state: AgentState) -> dict:
    """direct llm response"""

    sys = SystemMessage(content=("You are concise and helpful. "
                                "do NOT repeat, quote, or paraphrase the summary in your reply unless explicitly asked. "
                                "When the user asks for China universities' rankings/majors/admission or needs education information, "
                                "call `db_retrieve(query, top_k)`.\n"
                                "When the user needs broader, recent info across the web, call `web_search(query)`.\n"
                                "If a tool is used, ALWAYS read its ToolMessage and then produce a final answer."
                                ))
    context: list[BaseMessage] = [sys]
    if state.get("summary"):
        context.append(SystemMessage(content=f"[Conversation summary]\n{state['summary']}"))

    recent = list(state["messages"])
    context.extend([m for m in recent])
    llm_with_tools = LLM.bind_tools(TOOLS, tool_choice="auto")
    tools_by_name = {t.name: t for t in TOOLS}

    last_ai: AIMessage | None = None
    max_tool_iters = MAX_ROUNDS

    for _ in range(max_tool_iters):
        res = llm_with_tools.invoke(context)
        ai_msg = AIMessage(content=(getattr(res, "content", "") or ""),
                           tool_calls=getattr(res, "tool_calls", None))
        context.append(ai_msg)
        last_ai = ai_msg

        calls = getattr(res, "tool_calls", None) or []
        if not calls:
            break

        for call in calls:
            name = getattr(call, "name", None) or (call.get("name") if isinstance(call, dict) else None)
            call_id = getattr(call, "id", None) or (call.get("id") if isinstance(call, dict) else "")
            args = getattr(call, "args", None) or getattr(call, "arguments", None) \
                   or (call.get("args") if isinstance(call, dict) else None) \
                   or (call.get("arguments") if isinstance(call, dict) else None) \
                   or {}

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass

            tool = tools_by_name.get(name)
            try:
                if tool is None:
                    tool_result = f"[ToolError] Unknown tool: {name}"
                else:
                    tool_result = tool.invoke(args)
            except Exception as e:
                tool_result = f"[ToolError] {type(e).__name__}: {e}"

            context.append(ToolMessage(content=str(tool_result), tool_call_id=str(call_id)))

    final_text = (last_ai.content if last_ai else "")
    return {"messages": [AIMessage(content=final_text)]}