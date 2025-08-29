import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# data path 
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# planner.db config
DB_PATH = str(DATA_DIR / "planner.db")
THREAD_ID = str(os.getenv("PLANNER_THREAD_ID", "default"))


def get_config_summary():
    return {
        "db_path": DB_PATH,
        "thread_id": THREAD_ID
    }
