

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# data path 
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "planner.db"
DB_CHECKPOINTER_PATH = DATA_DIR / "checkpointer.db"

# primary config
MODEL = os.getenv("MODEL", "gpt-4o")
SIDE_MODEL = os.getenv("SIDE_MODEL", "gpt-4o-mini")
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "gpt-5")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# graph config
MAX_ROUNDS = 3
RECENT_K = 18
PLANNER_RECENT = 6
KEEP_RECENT = 4
SUMMARIZE_AFTER = 18

# retrieve config
CHUNK_SIZE = 835
CHUNK_OVERLAP = 120
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-large")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# reddit client param
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
USER_AGENT = os.getenv("USER_AGENT")


def get_config_summary():
    return {
        "db_path": DB_PATH,
        "model": MODEL,
        "side_model": SIDE_MODEL,
        "planner_model": PLANNER_MODEL,

    }
