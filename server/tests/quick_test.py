# server/tests/quick_test.py

import sys
import json
from pathlib import Path
from typing import Dict, Any
from tavily import TavilyClient
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import search
from config import SIDE_MODEL, TAVILY_API_KEY

SIDE_LLM = ChatOpenAI(model=SIDE_MODEL, temperature=0)


def test_search(query: str):
    trans_prompt = ("You are a bilingual translator; input is a user query; output EXACTLY "
                    "'<Chinese translation> ### <English translation>'; preserve meaning; no extra text/explanations.\n\n"
                    f"The query:\n{query}")
    sys_msg = SystemMessage(content=trans_prompt)
    res = SIDE_LLM.invoke([sys_msg])
    print(f"The res is: {res.content}")

    parts = [p.strip() for p in res.content.split("###", 1)]
    if len(parts) == 2:
        cn_query, en_query = parts[0], parts[1]
    else:
        print(res.content)
        sys.exit(1)

    out = search.advan_web_search(cn_query=cn_query, en_query=en_query)
    print(json.dumps(json.loads(out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    test_search("近视手术安全么")

# ----------------------------------------------------------------------------------------------------------

    # client = TavilyClient(api_key=TAVILY_API_KEY)
    # ext = client.extract(urls=["https://www.zhihu.com/api/v4/answers/1947366657803683630/root_comments"], include_images=False, include_links=False)
    # raw = (ext or {}).get("raw_content") or (ext or {}).get("content")
    # print(ext)

# ----------------------------------------------------------------------------------------------------------

    # answer_id = "1947366657803683630"
    # url = f"https://www.zhihu.com/api/v4/answers/{answer_id}/root_comments"
    # headers = {
    #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    #     "Referer": f"https://www.zhihu.com/question/{answer_id}",
    # }
    # params = {
    #     "order": "normal",
    #     "limit": 20,
    #     "offset": 0,
    #     "status": "open",
    # }
    # try:
    #     import requests
    #     response = requests.get(url, headers=headers, params=params, timeout=10)
    #     response.raise_for_status()
    #     data = response.json()
    #     print(data)
    # except requests.exceptions.RequestException as e:
    #     print(f"Request failed: {e}")

# ----------------------------------------------------------------------------------------------------------

