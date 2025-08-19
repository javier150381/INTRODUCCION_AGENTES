import os
import sys
import types

import pytest

dummy_module = types.ModuleType("sentence_transformers")
dummy_module.SentenceTransformer = object
sys.modules.setdefault("sentence_transformers", dummy_module)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import rag_faiss


def test_local_model_used(monkeypatch):
    calls = {"init": 0, "encode": []}

    class DummyModel:
        def __init__(self, name: str):
            calls["init"] += 1
            assert name == "all-MiniLM-L6-v2"

        def encode(self, text: str):
            calls["encode"].append(text)
            import numpy as np
            return np.array([0.1, 0.2, 0.3])

    monkeypatch.setattr(rag_faiss, "SentenceTransformer", DummyModel)
    rag_faiss._MODEL = None

    emb1 = rag_faiss._embed_text("hola")
    emb2 = rag_faiss._embed_text("mundo")

    assert emb1 == [0.1, 0.2, 0.3]
    assert emb2 == [0.1, 0.2, 0.3]
    assert calls["init"] == 1
    assert calls["encode"] == ["hola", "mundo"]

