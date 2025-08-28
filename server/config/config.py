import os
from dotenv import load_dotenv

load_dotenv()

# data path config
DATA_DIR = "data"
GAOKAO_INDEX_DIR = os.path.join(DATA_DIR, "gaokaozixun")

def get_config_summary():
    return {
        "kaogao_index": f"{GAOKAO_INDEX_DIR}"
    }
