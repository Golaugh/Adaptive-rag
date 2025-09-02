import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# data path 
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# primary config
DB_PATH = str(DATA_DIR / "planner.db")
MODEL = os.environ("MODEL", "gpt-4o")
SIDE_MODEL = os.environ("SIDE_MODEL", "gpt-4o-mini")

def get_config_summary():
    return {
        "db_path": DB_PATH,
        "model": MODEL,
        "side_model": SIDE_MODEL
    }
