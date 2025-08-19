import json
import os
import hashlib
from typing import Dict, List, Tuple, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

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


_MODEL: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """Lazily initialize and return the sentence-transformer model."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def _embed_text(text: str) -> List[float]:
    """Return embedding for ``text`` using a local sentence-transformer model."""
    model = _get_model()
    return model.encode(text).tolist()


def _hash_sources(sources: List[Dict[str, str]]) -> str:
    """Return a stable hash for the provided sources."""
    payload = json.dumps(sources, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


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
        dim = len(_embed_text(""))
        index = faiss.IndexFlatL2(dim)
    else:
        emb_matrix = np.vstack(embeddings)
        dim = emb_matrix.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(emb_matrix)
    sources_hash = _hash_sources(sources)
    save_index(index, metadata, index_file, meta_file, sources_hash=sources_hash, dim=dim)
    return index, metadata


def save_index(
    index: faiss.IndexFlatL2,
    metadata: List[Dict[str, str]],
    index_file: str = INDEX_FILE,
    meta_file: str = META_FILE,
    *,
    sources_hash: Optional[str] = None,
    dim: Optional[int] = None,
) -> None:
    """Persist index and metadata to disk."""
    faiss.write_index(index, index_file)
    meta_payload = {
        "dim": dim if dim is not None else index.d,
        "sources_hash": sources_hash,
        "chunks": metadata,
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, ensure_ascii=False)


def load_index(
    index_file: str = INDEX_FILE,
    meta_file: str = META_FILE,
) -> Tuple[faiss.IndexFlatL2, List[Dict[str, str]], Optional[int], Optional[str]]:
    """Load index, metadata and extra info from disk."""
    index = faiss.read_index(index_file)
    with open(meta_file, "r", encoding="utf-8") as f:
        meta_payload = json.load(f)
    if isinstance(meta_payload, dict):
        metadata = meta_payload.get("chunks", [])
        dim = meta_payload.get("dim")
        sources_hash = meta_payload.get("sources_hash")
    else:
        metadata = meta_payload
        dim = None
        sources_hash = None
    return index, metadata, dim, sources_hash


def ensure_index(
    sources: List[Dict[str, str]],
    index_file: str = INDEX_FILE,
    meta_file: str = META_FILE,
) -> Tuple[faiss.IndexFlatL2, List[Dict[str, str]]]:
    """Load existing index or build a new one from sources."""
    embed_dim = len(_embed_text(""))
    current_hash = _hash_sources(sources)
    if os.path.exists(index_file) and os.path.exists(meta_file):
        index, metadata, stored_dim, stored_hash = load_index(index_file, meta_file)
        if index.d != embed_dim or stored_hash != current_hash:
            index, metadata = build_index(sources, index_file=index_file, meta_file=meta_file)
        else:
            if stored_dim != embed_dim or stored_hash is None:
                save_index(index, metadata, index_file, meta_file, sources_hash=current_hash, dim=embed_dim)
        return index, metadata
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
