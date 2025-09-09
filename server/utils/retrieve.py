

import os
import sys
import json
import glob
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore.document import Document
from langchain_chroma.vectorstores import Chroma

from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.retrievers.document_compressors import EmbeddingsFilter, CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

logger = logging.getLogger(__name__)


SERVER_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = SERVER_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR =  str(DATA_DIR / "retriever.db")
DOCS_DIR = str(DATA_DIR / "education")
INGEST_FLAG = os.path.join(DOCS_DIR, ".ingested")


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMB_MODEL, RERANKER_MODEL
EMB = OpenAIEmbeddings(model=EMB_MODEL)

try:
    SEM_SPLITTER = SemanticChunker(EMB, breakpoint_threshold_type="percentile")
    _USE_SEMANTIC = True
except Exception as _e:
    logger.warning("SemanticChunker unavailable, fallback to RecursiveCharacterTextSplitter: %s", _e)
    SEM_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    _USE_SEMANTIC = False

_HF_CE = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
_RERANKER = CrossEncoderReranker(model=_HF_CE, top_n=50)
_VECTOR: Optional[Chroma] = None
_BM25: Optional[BM25Retriever] = None
_EMB_FILTER = EmbeddingsFilter(embeddings=EMB, similarity_threshold=0.3)


@tool
def db_retrieve(query: str, top_k: int = 5) -> str:
    """
    only use this tool when the query is Chinese-education-related.
    Hybrid (BM25 + Dense) -> Reciprocal Rank Fusion -> BGE cross-encoder rerank -> Embedding filter compression.
    """
    try:
        _ensure_indexes()

        dense_docs = _VECTOR.similarity_search(query, k=max(20, top_k))
        if _BM25:
            _BM25.k = max(50, top_k)
            sparse_docs = _BM25.invoke(query)
        else:
            sparse_docs = []
        fused = _rrf_fuse({"dense": dense_docs, "sparse": sparse_docs}, k=max(50, top_k))
        reranked_docs = _RERANKER.compress_documents(fused, query=query)
        final_docs = _EMB_FILTER.compress_documents(reranked_docs, query=query)[:top_k]

        return json.dumps(
            {"results": [{"text": d.page_content, "metadata": d.metadata} for d in final_docs]},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.exception("db_retrieve error")
        return json.dumps({"error": str(e)})


def _ensure_indexes() -> None:
    """construct/reuse semanticSearch & BM25"""
    global _VECTOR, _BM25
    os.makedirs(CHROMA_DIR, exist_ok=True)

    docs = _load_local_docs(DOCS_DIR)

    if _VECTOR is None:
        _VECTOR = Chroma(persist_directory=CHROMA_DIR, embedding_function=EMB)
        if not os.path.exists(INGEST_FLAG) and docs:
            _VECTOR.add_documents(docs)
            open(INGEST_FLAG, "w").close()

    if _BM25 is None:
        if docs:
            _BM25 = BM25Retriever.from_documents(docs)
        else:
            _BM25 = BM25Retriever.from_texts([""])


def _load_local_docs(folder: str) -> List[Document]:
    docs: List[Document] = []
    if not os.path.isdir(folder):
        return docs

    for path in glob.glob(os.path.join(folder, "**", "*"), recursive=True):
        if os.path.isdir(path):
            continue
        if path.lower().endswith((".txt", ".md")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            chunks = (
                SEM_SPLITTER.split_text(text)
                if _USE_SEMANTIC
                else RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_text(text)
            )
            for i, chunk in enumerate(chunks):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": os.path.basename(path),
                            "path": path,
                            "idx": i,
                            "split": "semantic" if _USE_SEMANTIC else "rc",
                        },
                    )
                )
    return docs


def _rrf_fuse(candidates: Dict[str, List[Document]], k: int = 50, c: int = 60) -> List[Document]:
    """
    Reciprocal Rank Fusion: score += 1 / (c + rank)
    """
    def _doc_key(d: Document) -> Tuple[str, int, str]:
        return (d.metadata.get("path", ""), int(d.metadata.get("idx", -1)), d.page_content[:30])

    scores: Dict[Tuple[str, int, str], float] = {}
    first_doc: Dict[Tuple[str, int, str], Document] = {}

    for _, docs in candidates.items():
        for rank, d in enumerate(docs[:k], start=1):
            doc_id = _doc_key(d)
            first_doc.setdefault(doc_id, d)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (c + rank)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [first_doc[doc_id] for doc_id, _ in ranked]
