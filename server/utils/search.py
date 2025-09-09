

import re
import sys
import json
import praw
import logging
from urllib.parse import urlparse
from pathlib import Path
from tavily import TavilyClient
from typing import Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TAVILY_API_KEY, CLIENT_ID, CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD, USER_AGENT

logger = logging.getLogger(__name__)

class AdvSearchArgs(BaseModel):
    cn_query: str = Field(..., description="Chinese translation to the query")
    en_query: str = Field(..., description="English translation to the query")

# Tools
@tool
def web_search(query: str) -> str:
    """
    Perform a general web search using Tavily API and return the results
    in a structured JSON string for agent consumption.

    Args:
        query (str): The user query (any language). This will be sent to Tavily.

    Process:
        1. Send the query to TavilyClient with basic search parameters.
        2. Retrieve up to 5 results including title, url, content, score, and favicon.
        3. Collect metadata such as response time, auto_parameters, and request_id.
        4. Build a JSON object with all search results and metadata.

    Returns:
        str: A JSON string with the following structure:
            {
                "query": str,                # the original or Tavily-modified query
                "answer": str | null,        # Tavily's direct answer if provided
                "results": [
                    {
                        "title": str | null,
                        "url": str | null,
                        "content": str | null,
                        "score": float | null,    # Tavily relevance score
                        "favicon": str | null     # favicon URL if available
                    },
                    ...
                ],
                "citations": [str, ...],     # list of URLs extracted from results
                "source": "tavily",          # constant identifier of the source
                "response_time": float | null,
                "auto_parameters": dict | null,
                "request_id": str | null
            }

    Notes:
        - This function returns a JSON-serialized string, not a Python dict.
        - `results` may be empty if Tavily finds nothing.
        - Use `ensure_ascii=False` to preserve Unicode characters in the output.
    """

    client = TavilyClient(api_key=TAVILY_API_KEY)
    params: Dict[str, Any] = {
        "search_depth": "basic",
        "topic": "general",
        "include_answer": False,
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


@tool(args_schema=AdvSearchArgs)
def advan_web_search(cn_query: str, en_query: str) -> str:
    """
    Perform a cross-platform web search (Zhihu in Chinese, Reddit in English) for a user query,
    and return the extracted answer factors in a structured JSON string.

    Args:
        cn_query (str): The user query in Chinese.
        en_query (str): The user query in English.

    Returns:
        str: A JSON string with the following structure:
            {
                "factors_from_zhihu": [
                    { "<zhihu_answer_content>": [ { "content": str,
                                                   "created_time": str,
                                                   "vote_count": int, ... }, ... ] },
                    ...
                ],
                "factors_from_reddit": [
                    { "title": str,
                      "type": "post" | "comment" | "reply",
                      "text": str,
                      "score": int, ... },
                    ...
                ]
            }

    Notes:
        - Zhihu credibility is indicated by higher `vote_count`.
        - Reddit credibility is indicated by higher `score`.
    """

    client = TavilyClient(api_key=TAVILY_API_KEY)
    zhihu_params: Dict[str, Any] = {
        "search_depth": "basic",
        "topic": "general",
        "include_answer": False,
        "include_raw_content": False,
        "max_results": 20,
        "include_domains": ["https://www.zhihu.com/question/"],
    }
    reddit_params: Dict[str, Any] = {
        "search_depth": "basic",
        "topic": "general",
        "include_answer": False,
        "include_raw_content": False,
        "max_results": 1,
        "include_domains": ["https://www.reddit.com/"],
    }
    zhihu_res = client.search(cn_query, **zhihu_params)
    reddit_res = client.search(en_query, **reddit_params)


    _zh_urls = [it.get("url", "") for it in zhihu_res.get("results", [])]
    zhihu_list = [u.rstrip("/").split("/")[-1] for u in _zh_urls if "/answer/" in u]
    if not zhihu_list:
        tmp = []
        for u in _zh_urls:
            parts = u.rstrip("/").split("/")
            if "question" in parts:
                try:
                    qid = parts[parts.index("question")+1]
                    tmp.append(qid)
                except Exception:
                    pass
        zhihu_list = list(dict.fromkeys(tmp))

    reddit_list = []
    for it in reddit_res.get("results", []):
        u = it.get("url", "")
        try:
            p = urlparse(u)
            if p.netloc.endswith("reddit.com"):
                path = p.path if p.path.endswith("/") else p.path + "/"
                m = re.search(r"/r/([^/]+)/", path)
                if m:
                    reddit_list.append(m.group(1))
        except Exception:
            pass
    reddit_list = list(dict.fromkeys(reddit_list))

    print(zhihu_list)
    print(reddit_list)
    from .zhihu_search import ZhihuCollector
    zhihu = ZhihuCollector(zhihu_list)
    zhihu_factor = zhihu.search(max_count=10, show_comments=0, return_factor=True)

    from .reddit_search import RedditCollector
    reddit_client = praw.Reddit(
        client_id = CLIENT_ID,
        client_secret = CLIENT_SECRET,
        username = REDDIT_USERNAME,
        password = REDDIT_PASSWORD,
        user_agent = USER_AGENT
    )
    reddit = RedditCollector(client=reddit_client, subr_list=reddit_list)
    reddit_factor = reddit.search(max_count=10, return_factor=True)

    mapped_zhihu = []
    for i, item in enumerate(zhihu_factor):
        if isinstance(item, dict) and i < len(zhihu_res["results"]):
            content = zhihu_res["results"][i].get("content") or zhihu_res["results"][i].get("title")
            mapped_zhihu.append({content: list(item.values())[0]})
        else:
            mapped_zhihu.append(item)

    return json.dumps({
        "factors_from_zhihu": mapped_zhihu,
        "factors_from_reddit": reddit_factor,
    })