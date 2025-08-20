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
    result = pirjo_pipeline.verificador_bibliografia(text, sources)
    assert "Referencias" in result
    assert "- doc1.pdf" in result
    assert "- doc2.pdf" in result


def test_verificador_ignores_unknown_citations():
    text = "Dato [doc3.pdf:1:1]."
    sources = [{"file": "doc1.pdf", "page": 1, "chunk": 1, "text": ""}]
    result = pirjo_pipeline.verificador_bibliografia(text, sources)
    assert "Referencias" not in result
