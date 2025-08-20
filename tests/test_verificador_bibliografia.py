import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pirjo_pipeline


def test_verificador_uses_only_known_sources():
    text = "Dato [doc1.pdf:1:1] y más [doc2.pdf:2:1]."
    sources = [
        {"file": "doc1.pdf", "page": 1, "chunk": 1, "text": ""},
        {"file": "doc2.pdf", "page": 2, "chunk": 1, "text": ""},
    ]
    metadata = {
        "doc1.pdf": {"author": "Autor1", "title": "Título1", "year": "2020"},
        "doc2.pdf": {"author": "Autor2", "title": "Título2", "year": "2021"},
    }
    result = pirjo_pipeline.verificador_bibliografia(text, sources, metadata)
    assert "Referencias" in result
    assert "- Autor1 (2020). Título1." in result
    assert "- Autor2 (2021). Título2." in result


def test_verificador_ignores_unknown_citations():
    text = "Dato [doc3.pdf:1:1]."
    sources = [{"file": "doc1.pdf", "page": 1, "chunk": 1, "text": ""}]
    metadata = {"doc1.pdf": {"author": "Autor", "title": "Título", "year": "2020"}}
    result = pirjo_pipeline.verificador_bibliografia(text, sources, metadata)
    assert "Referencias" not in result
