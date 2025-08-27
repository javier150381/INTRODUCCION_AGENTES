import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pirjo_pipeline


def test_unir_bloques_concatenates_texts():
    raw = {
        "P": "p1",
        "P2": "p2",
        "I": {"doc1": {"Resumen": "i1"}, "doc2": {"Resumen": "i2"}},
        "O1": "o1",
        "O": "o2",
    }
    merged = pirjo_pipeline.unir_bloques_pirjo(raw)
    assert merged["P"] == "p1 p2"
    assert merged["I"].strip().startswith("i1") and "i2" in merged["I"]
    assert merged["O"] == "o1 o2"


def test_redactor_desde_json_uses_merged_blocks(monkeypatch):
    captured = {}

    def fake_redactor(blocks):
        captured["blocks"] = blocks
        return "texto"

    monkeypatch.setattr(pirjo_pipeline, "redactor_cientifico", fake_redactor)
    raw = {"P": "a", "P2": "b", "O": "c"}
    result = pirjo_pipeline.redactor_desde_json(raw)
    assert result == "texto"
    assert captured["blocks"]["P"] == "a b"
