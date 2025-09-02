

import re
import json
from langchain_core.tools import tool

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}")


# Tools
@tool
def web_search(query: str) -> str:
    """this search_node use TAVILY_API_KEY to get info from the internet"""
    return json.dumps({
        "query": query,
        "snippet": f"[Stub] Pretend web results about: {query}",
        "source": "web_stub"
    })


def advan_web_search(query: str) -> str:
    """this search_node use TAVILY_API_KEY to get info from the internet"""
    return json.dumps({
        "query": query,
        "snippet": f"[Stub] Pretend web results about: {query}",
        "source": "web_stub"
    })