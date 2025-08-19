import json
import os
from typing import Dict, List, Tuple

import faiss
import numpy as np

from openai_utils import get_client

INDEX_FILE = "faiss.index"
META_FILE = "faiss_meta.json"


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into chunks of roughly ``chunk_size`` words with overlap."""
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _embed_text(text: str) -> List[float]:
    """Return embedding for ``text`` using OpenAI's small embedding model."""
    client = get_client()
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


def build_index(
    sources: List[Dict[str, str]],
    index_file: str = INDEX_FILE,
    meta_file: str = META_FILE,
    chunk_size: int = 500,
    overlap: int = 50,
) -> Tuple[faiss.IndexFlatL2, List[Dict[str, str]]]:
    """Build a FAISS index from sources and persist it along with metadata."""
    embeddings: List[np.ndarray] = []
    metadata: List[Dict[str, str]] = []
    for src in sources:
        for idx, chunk in enumerate(_chunk_text(src["text"], chunk_size, overlap)):
            emb = np.array(_embed_text(chunk), dtype="float32")
            embeddings.append(emb)
            metadata.append({
                "file": src["file"],
                "page": src["page"],
                "chunk": idx,
                "text": chunk,
            })
    if not embeddings:
        dim = 0
        index = faiss.IndexFlatL2(0)
    else:
        emb_matrix = np.vstack(embeddings)
        dim = emb_matrix.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(emb_matrix)
    save_index(index, metadata, index_file, meta_file)
    return index, metadata


def save_index(
    index: faiss.IndexFlatL2,
    metadata: List[Dict[str, str]],
    index_file: str = INDEX_FILE,
    meta_file: str = META_FILE,
) -> None:
    """Persist index and metadata to disk."""
    faiss.write_index(index, index_file)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)


def load_index(
    index_file: str = INDEX_FILE,
    meta_file: str = META_FILE,
) -> Tuple[faiss.IndexFlatL2, List[Dict[str, str]]]:
    """Load index and metadata from disk."""
    index = faiss.read_index(index_file)
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def ensure_index(
    sources: List[Dict[str, str]],
    index_file: str = INDEX_FILE,
    meta_file: str = META_FILE,
) -> Tuple[faiss.IndexFlatL2, List[Dict[str, str]]]:
    """Load existing index or build a new one from sources."""
    if os.path.exists(index_file) and os.path.exists(meta_file):
        return load_index(index_file, meta_file)
    return build_index(sources, index_file=index_file, meta_file=meta_file)


def search_index(
    query: str,
    k: int,
    index: faiss.IndexFlatL2,
    metadata: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Retrieve ``k`` most similar chunks to ``query`` from ``index``."""
    if index.ntotal == 0:
        return []
    emb = np.array([_embed_text(query)], dtype="float32")
    _, idxs = index.search(emb, k)
    results: List[Dict[str, str]] = []
    for i in idxs[0]:
        if 0 <= i < len(metadata):
            results.append(metadata[i])
    return results
