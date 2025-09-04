

import os
import json
import logging
from tavily import TavilyClient
from typing import Dict, Any
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Tools
@tool
def web_search(query: str) -> str:
    """Search the web with Tavily and return structured JSON for agents."""

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("TAVILY_API_KEY not detected!")

    client = TavilyClient(api_key=api_key)
    params: Dict[str, Any] = {
        "search_depth": "basic",
        "topic": "general",
        "include_answer": True,
        "include_raw_content": False,
        "max_results": 5,
    }
    res = client.search(query, **params)

    results = [{
        "title": it.get("title"),
        "url": it.get("url"),
        "content": it.get("content"),
        "score": it.get("score"),
        "favicon": it.get("favicon"),
    } for it in res.get("results", [])]

    out = {
        "query": res.get("query", query),
        "answer": res.get("answer"),
        "results": results,
        "citations": [r["url"] for r in results if r.get("url")],
        "source": "tavily",
        "response_time": res.get("response_time"),
        "auto_parameters": res.get("auto_parameters"),
        "request_id": res.get("request_id"),
    }
    return json.dumps(out, ensure_ascii=False)


def advan_web_search(query: str) -> str:
    """using more fine-tuned methods to search necessary and related factors across all platforms for the query"""

    from zhihu_search import Zhihu
    from reddit_search import Reddit

    zhihu = Zhihu(['81964408445','82586149604','82493740255','81348057992','81748398040','81531639383'])
    zhihu_factor = zhihu.search()

    reddit = Reddit('bot1', ['AskEngineers' ,'financialindependence' ,'Entrepreneur' ,'smallbusiness' , 'lifehacks'])
    reddit_factor = reddit.search()

    return json.dumps({
        "query": query,
        "factors_from_zhihu": zhihu_factor,
        "factors_from_reddit": reddit_factor,
    })