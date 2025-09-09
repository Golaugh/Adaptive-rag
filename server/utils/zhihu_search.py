

import requests
import logging
import time
from typing import Optional
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)
from config import DATA_DIR


class ZhihuCollector:

    # ['81964408445','82586149604','82493740255','81348057992','81748398040','81531639383']
    def __init__(self, list_ids: list[str], log: Optional[logging.Logger] = logger) -> None:
        self.answer_ids = list_ids
        self.province_dict = {}
        self.logger = log

    def search(self, max_count: int = 20, show_comments: int = 0, return_factor: bool = True) -> Optional[list[dict]]:
        """
        Run gathering sequence over all answer_ids.

        :param max_count: maximum number of comments to fetch per answer_id
        :param show_comments: number of comments to print as examples
        :param return_factor: if True, return gathered comments; otherwise save to file
        :return: list of dicts (comments grouped per answer_id) or None
        """
        factors = []

        for answer_id in self.answer_ids:
            self.logger.debug(f"Start gathering comments from {answer_id}...")
            comments = self._get_all_comments(answer_id, max_count)

            if not comments:
                self.logger.debug(f"No comments found for {answer_id}")
                continue

            # show sample comments
            for i, comment in enumerate(comments[:show_comments], 1):
                print(f"\nExample comment {i}:")
                print(f"ID: {comment['user_id']}")
                print(f"Username: {comment['user_name']}")
                print(f"Province: {comment['province']}")
                print(f"Created_time: {comment['created_time']}")
                print(f"Content: {comment['content']}")
                print(f"Likes: {comment['like_count']}")

            # for comment in comments:
            #     province = comment['province']
            #     self.province_dict[province] = self.province_dict.get(province, 0) + 1

            if return_factor:
                factors.append({answer_id: comments})
            else:
                self._save_to_file(comments, answer_id)

        return factors or None


    def _get_all_comments(self, answer_id: str, max_count: int):
        """
        getting all comments
        :param answer_id: targeted answer_id
        :param max_count: maximum number of comments to get
        :return: all comment's list
        """
        all_comments = []
        offset = 0
        limit = 20  # get 20 comments per query

        while True:
            self.logger.debug(f"Now getting comments on {offset // limit + 1} page...")
            data = self._get_zhihu_answer_comments(answer_id, limit, offset)

            if not data or "data" not in data:
                self.logger.debug("No more comments")
                break

            comments = self._parse_comments(data)
            all_comments.extend(comments)

            # check if there's still more comments
            if data.get("paging", {}).get("is_end", True) and len(comments) == 0:
                self.logger.debug("Reach the bottom / Comments empty.")
                break

            if len(all_comments) >= max_count:
                all_comments = all_comments[:max_count]
                self.logger.debug(f"Getting maximum num of comments: {max_count}")
                break

            offset += limit
            time.sleep(2)
        return all_comments


    def _get_zhihu_answer_comments(self, answer_id: str, limit: int, offset: int):
        """
        get commentes from zhihu answers
        :param answer_id: the targeted answer
        :param limit: maximum number of comments per request
        :param offset: offset (starting from offset.value)
        :return: list of comments (in json)
        """
        url = f"https://www.zhihu.com/api/v4/answers/{answer_id}/root_comments"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": f"https://www.zhihu.com/question/{answer_id}",
        }

        params = {
            "order": "normal",
            "limit": limit,
            "offset": offset,
            "status": "open",
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            return None

    
    def _parse_comments(self, comment_data: list[dict]):
        """
        structurize the comments' info
        :param comment_data: original comment data
        :return: structured output
        """
        comments = []
        if not comment_data or "data" not in comment_data:
            return comments

        for item in comment_data["data"]:
            logger.debug(item)
            try:
                comment = {
                    "content": item.get("content", ""),
                    "created_time": datetime.fromtimestamp(item.get("created_time", 0)).strftime('%Y-%m-%d %H:%M:%S'),
                    "vote_count": item.get("vote_count", 0),
                }
                comments.append(comment)
            except Exception as e:
                self.logger.error(f"Comment structure error: {e}")
                continue
        
        if len(comments) > 0:
            logger.info(f"Collected comments:\n{comments}")
        return comments


    def _save_to_file(self, comments: list[dict], answer_id: str, filename: str = "zhihu", encoding: str = 'utf-8'):
        """
        save the comments into {filename}_{answer_id}.csv
        :param comments: list of comments
        :param filename: saved filename
        """
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            o_file = DATA_DIR / f"{filename}_{answer_id}.csv"
            df = pd.DataFrame(comments)
            df.to_csv(o_file, index=False, encoding=encoding)
                # json.dump(comments, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Comments saved to {o_file}")
        except Exception as e:
            self.logger.error(f"Saved file failed: {e}")