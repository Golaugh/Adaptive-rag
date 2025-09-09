

import sys
import json
import time
from pathlib import Path
from langchain_openai import ChatOpenAI
from typing import Annotated, TypedDict, Sequence, Literal, List, Dict, Any
from langgraph.graph.message import BaseMessage, add_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PLANNER_MODEL, PLANNER_RECENT
from utils.search import advan_web_search

TOOLS = [advan_web_search]
PLANNER = ChatOpenAI(model=PLANNER_MODEL).bind_tools(TOOLS)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    summary: str
    route: Literal["direct", "normal", "planner"]
    user_id: int
    thread_id: str
    hitl_collected: List[str]
    hitl_needed: str
    hitl_round: int


class PlannerHandle:

    def __init__(self, state: AgentState):
        self.state = state

    def handle(self) -> str:
        """
        Let PLANNER_MODEL freely decide whether to call `advan_web_search` via tool-calls.
        The model may call the tool up to TWO times; after that, we force it to stop and produce
        the final structured output.

        Final output contract (MUST be JSON):
        {
          "summary": str,
          "timeline": [{ "milestone": str, "by": str, "notes": str }, ...],
          "plan": [{ "step": str, "rationale": str }, ...],
          "risks": [{ "risk": str, "mitigation": str }, ...],
          "citations": [str, ...]
        }

        We enrich the return with meta:
        {
          ...above fields...
          "tool_calls": int,
          "elapsed_sec": float,
          "sources": { "zhihu": int, "reddit": int },
          "raw_factors": {...}  # last tool result if available
        }
        """
        messages_seq: List[BaseMessage] = list(self.state["messages"])
        recent_msgs: List[BaseMessage] = messages_seq[-PLANNER_RECENT:]
        last_user = next((m for m in reversed(messages_seq) if isinstance(m, HumanMessage)), None)
        query: str = (last_user.content if last_user else "")
            
        t0 = time.perf_counter()
        sys_prompt = ("You are a senior planning agent with tool access.\n"
                    "- You may CALL the tool `advan_web_search` to gather cross-source factors (Zhihu/Reddit) "
                    "using {cn_query, en_query}. You decide whether to call it, and what queries to use.\n"
                    "- Tool-call BUDGET: at most TWO calls total.\n"
                    "- After you finish zero, one, or two calls, you MUST return the FINAL answer as valid JSON with keys:\n"
                    "  summary, timeline(list of {milestone, by, notes}), plan(list of {step, rationale}),\n"
                    "  risks(list of {risk, mitigation}), citations(list of str).\n"
                    "- Be concise, remove duplicates, prefer higher vote_count (Zhihu) and higher score (Reddit).\n"
                    "- Do not include any extra prose outside JSON."
                )
        
        messages: List[BaseMessage] = [SystemMessage(content=sys_prompt)]
        if self.state.get("summary"):
            messages.append(SystemMessage(content=f"[Conversation summary]\n{self.state['summary']}"))
        messages.extend(recent_msgs)
        messages.append(AIMessage(content="{"))

        tool_calls = 0
        last_factors: Dict[str, Any] = {}
        while True:
            res: AIMessage = PLANNER.invoke(messages)
            messages.append(res)

            if getattr(res, "tool_calls", None):
                if tool_calls >= 2:
                    messages.append(HumanMessage(
                        content=("You have reached the tool-call budget (2). "
                                 "Now STOP calling tools and produce the FINAL JSON answer as specified.")
                    ))
                    final_res: AIMessage = PLANNER.invoke(messages)
                    final_text = getattr(final_res, "content", "") or "{}"
                    final_core = json.loads(final_text)

                    wrap = {
                        **final_core,
                        "tool_calls": tool_calls,
                        "elapsed_sec": round(time.perf_counter() - t0, 2),
                        "sources": {
                            "zhihu": len((last_factors or {}).get("factors_from_zhihu", []) or []),
                            "reddit": len((last_factors or {}).get("factors_from_reddit", []) or []),
                        },
                        "raw_factors": last_factors or {}
                    }
                    return self._safe_json(wrap)

                for call in res.tool_calls:
                    if call["name"] == "advan_web_search":
                        args = call.get("args") or {}
                        cn_q = args.get("cn_query") or query
                        en_q = args.get("en_query") or query

                        raw = advan_web_search.invoke({"cn_query": cn_q, "en_query": en_q}) \
                              if hasattr(advan_web_search, "invoke") else advan_web_search(cn_q, en_q)
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8", errors="ignore")
                        if not isinstance(raw, str):
                            raw = str(raw)
                        try:
                            parsed = json.loads(raw)
                            parsed.setdefault("factors_from_zhihu", [])
                            parsed.setdefault("factors_from_reddit", [])
                            last_factors = parsed
                        except Exception:
                            last_factors = {"factors_from_zhihu": [], "factors_from_reddit": []}

                        messages.append(ToolMessage(content=raw, tool_call_id=call["id"]))
                        tool_calls += 1

                if tool_calls >= 2:
                    messages.append(HumanMessage(
                        content=("You have now called tools twice (max). "
                                 "STOP calling tools and produce the FINAL JSON answer as specified.")
                    ))
                    final_ai2: AIMessage = PLANNER.invoke(messages)
                    final_text2 = getattr(final_ai2, "content", "") or "{}"
                    try:
                        final_core2 = json.loads(final_text2)
                    except Exception:
                        final_core2 = {"summary": final_text2.strip(), "timeline": [], "plan": [], "risks": [], "citations": []}
                    wrap2 = {
                        **final_core2,
                        "tool_calls": tool_calls,
                        "elapsed_sec": round(time.perf_counter() - t0, 2),
                        "sources": {
                            "zhihu": len((last_factors or {}).get("factors_from_zhihu", []) or []),
                            "reddit": len((last_factors or {}).get("factors_from_reddit", []) or []),
                        },
                        "raw_factors": last_factors or {}
                    }
                    return self._safe_json(wrap2)

                continue

            final_text = getattr(res, "content", "") or "{}"
            final_core = json.loads(final_text)
            wrap = {
                **final_core,
                "tool_calls": tool_calls,
                "elapsed_sec": round(time.perf_counter() - t0, 2),
                "sources": {
                    "zhihu": len((last_factors or {}).get("factors_from_zhihu", []) or []),
                    "reddit": len((last_factors or {}).get("factors_from_reddit", []) or []),
                },
                "raw_factors": last_factors or {}
            }
            return "```json\n" + self._safe_json(wrap) + "\n```"

    def _safe_json(self, obj: Dict[str, Any]) -> str:
        return json.dumps(obj, ensure_ascii=False, indent=2)
