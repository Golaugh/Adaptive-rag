

import os
import json
import glob
import logging
from pathlib import Path
from typing import List
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


SERVER_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = SERVER_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR =  str(DATA_DIR / "retriever.db")
DOCS_DIR = str(DATA_DIR / "gaokaozixun")
INGEST_FLAG = os.path.join(CHROMA_DIR, ".ingested")
splitter = RecursiveCharacterTextSplitter(chunk_size=835, chunk_overlap=120)
emb = OpenAIEmbeddings(model="text-embedding-3-small")


@tool
def db_retrieve(query: str, k: int = 3) -> str:
    """only use this tool when the query is Chinese-education-related"""
    try:
        VECTOR = _ensure_chroma()
        hits = VECTOR.similarity_search(query, k=max(1, min(10, int(k))))
        return json.dumps({
            "results": [{"text": d.page_content, "metadata": d.metadata} for d in hits]
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def _ensure_chroma() -> Chroma:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=emb)
    if not os.path.exists(INGEST_FLAG) and os.path.isdir(DOCS_DIR):
        raw_docs = _load_local_docs(DOCS_DIR)
        if raw_docs:
            vs.add_documents(raw_docs)
            vs.persist()
            open(INGEST_FLAG, "w").close()
    return vs


def _load_local_docs(folder: str) -> List[Document]:
    docs = []
    for path in glob.glob(os.path.join(folder, "**", "*"), recursive=True):
        if os.path.isdir(path): 
            continue
        if path.lower().endswith((".txt", ".md")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            for i, chunk in enumerate(splitter.split_text(text)):
                docs.append(Document(page_content=chunk, metadata={"source": os.path.basename(path), "idx": i}))
    return docs
