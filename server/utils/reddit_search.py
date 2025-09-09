# praw github
# https://github.com/praw-dev/praw
import praw
import time
import logging
import json
from typing import Optional
from config import DATA_DIR
from collections import deque

logger = logging.getLogger(__name__)


class RedditCollector:

    # ['AskEngineers' ,'financialindependence' ,'Entrepreneur' ,'smallbusiness' , 'lifehacks' ,
    # 'productivity', 'GetMotivated' ,'GetStudying' ,'Cooking' ,'fantasywriters' ,'WritingPrompts' ,'ShortStories','Jokes']
    # subr = "funny"
    def __init__(self, client: praw.Reddit, subr_list: list[str], month_limit: int = 12, log: Optional[logging.Logger] = logger):
        self.reddit = client
        self.subr_list = subr_list
        self.month_limit = month_limit
        self.logger = log


    def search(
        self,
        max_count: int = 5,                
        return_factor: bool = True,         
        max_submissions: int = 5,
        max_comments: int = 10,
        max_seconds: int = 5,
        per_item_max_chars: int = 500,
        max_total_chars: int = 2000,
    ):
        """
        params:
            max_count: how many posts to collect eventually
            return_factor: if return the structured JSON result
            max_submissions: how many posts to look up per subreddit
            max_comments: how many comments to take per subreddit
            max_seconds: time limit per subreddit
            per_item_max_chats: maximum text length for each
            max_total_chats: total budget for each subreddit
        """
        factors = []
        for subr in self.subr_list:
            subreddit = self.reddit.subreddit(subr)
            collect_list = []
            start = time.time()
            deadline = start + max_seconds
            taken_comments = 0
            used_chars = 0
            seen_submissions = 0

            for subm in subreddit.hot(limit=max_submissions):
                if time.time() >= deadline:
                    break
                if not self._has_time_efficiency(getattr(subm, "created_utc", 0)):
                    continue

                # limit unfold times
                try:
                    subm.comments.replace_more(limit=0)
                except Exception as e:
                    self.logger.warning(f"replace_more failed: {e}")

                # clip the content
                if getattr(subm, "selftext", ""):
                    txt = self._clip_text(subm.selftext, per_item_max_chars)
                    if txt:
                        collect_list.append(self._pack_piece(subr, "post", txt, submission=subm))
                        used_chars += len(txt)
                        if used_chars >= max_total_chars:
                            break

                # budget control
                remain_take = max_comments - taken_comments
                remain_chars = max_total_chars - used_chars
                if remain_take <= 0 or remain_chars <= 0:
                    break

                got_items, got_count, got_chars = self._comments_in_a_submission_limited(
                    subm,
                    max_take=remain_take,
                    deadline=deadline,
                    per_item_max_chars=per_item_max_chars,
                    remaining_chars=remain_chars,
                    subreddit_name=subr,
                )
                collect_list.extend(got_items)
                taken_comments += got_count
                used_chars += got_chars
                seen_submissions += 1

                if time.time() >= deadline or used_chars >= max_total_chars or taken_comments >= max_comments:
                    break

                time.sleep(0.2)

            if len(collect_list) > max_count:
                collect_list = collect_list[:max_count]

            if return_factor:
                factors.extend(collect_list)
            else:
                self._dump_a_subreddit(collect_list, subr)

        if return_factor:
            return factors


    def _clip_text(self, text: str, per_item_max_chars: int) -> str:
        if not text:
            return ""
        t = " ".join(text.split())
        t = t.replace("http://", " ").replace("https://", " ")
        return t[:per_item_max_chars].rstrip()


    def _pack_piece(self, subr: str, kind: str, text: str, submission=None, comment=None):
        # "src": "reddit", "subr": subr 
        base = {"type": kind, "text": text}
        if submission is not None:
            # base["url"] = f"https://reddit.com{getattr(submission, 'permalink', '')}"
            base["score"] = getattr(submission, "score", None)
            base["title"] = getattr(submission, "title", None)
        if comment is not None:
            # base["url"] = f"https://reddit.com{getattr(comment, 'permalink', '')}"
            base["score"] = getattr(comment, "score", None)
        return base


    def _dump_a_subreddit(self, list_of_posts, subreddit_name: str):
        """dump one file with one r/subreddit"""

        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            path = DATA_DIR / f"{subreddit_name}.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump(list_of_posts, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Saved subreddit data to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save subreddit {subreddit_name}: {e}")


    def _has_time_efficiency(self, created_utc, dateback_months: int = 3):
        if dateback_months:
            return time.time() - created_utc <= dateback_months*30*24*3600
        else:
            return time.time() - created_utc <= self.month_limit*30*24*3600
    

    def _contents_in_a_submission(self, submission):
        contents=[]
        if submission.selftext != None:
            if self._has_time_efficiency(submission.created_utc):
                contents.append(submission.selftext)
                self.logger.debug(submission.selftext)
        return contents


    def _comments_in_a_submission(self, subm):
        items, _, _ = self._comments_in_a_submission_limited(
            subm,
            max_take=10**4,
            deadline=time.time() + 10**4,
            per_item_max_chars=10**4,
            remaining_chars=10**5,
            subreddit_name=getattr(getattr(subm, "subreddit", None), "display_name", ""),
        )
        return [it["text"] for it in items]


    def _comments_in_a_submission_limited(
        self, submission, *, max_take: int, deadline: float,
        per_item_max_chars: int, remaining_chars: int, subreddit_name: str
    ):
        items = []
        taken = 0
        used_chars = 0

        for comment in getattr(submission, "comments", []):
            if time.time() >= deadline:
                break
            body = getattr(comment, "body", None)
            if body and body != "[deleted]":
                if self._has_time_efficiency(getattr(comment, "created_utc", 0)):
                    piece = self._clip_text(body, per_item_max_chars)
                    if piece:
                        items.append(self._pack_piece(subreddit_name, "comment", piece, submission=submission, comment=comment))
                        taken += 1
                        used_chars += len(piece)
                        if taken >= max_take or used_chars >= remaining_chars:
                            return items, taken, used_chars

            dq = deque(getattr(comment, "replies", []) or [])
            while dq:
                if time.time() >= deadline:
                    break
                reply = dq.popleft()
                rbody = getattr(reply, "body", None)
                if rbody and rbody != "[deleted]":
                    if self._has_time_efficiency(getattr(reply, "created_utc", 0)):
                        piece = self._clip_text(rbody, per_item_max_chars)
                        if piece:
                            items.append(self._pack_piece(subreddit_name, "reply", piece, submission=submission, comment=reply))
                            taken += 1
                            used_chars += len(piece)
                            if taken >= max_take or used_chars >= remaining_chars:
                                return items, taken, used_chars
                more = getattr(reply, "replies", None)
                if more:
                    dq.extend(more)

            if taken >= max_take or used_chars >= remaining_chars:
                break

        return items, taken, used_chars