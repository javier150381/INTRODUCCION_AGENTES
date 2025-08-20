import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pirjo_pipeline


def test_redactor_cientifico_prompt_is_coherent(monkeypatch):
    captured = {}

    def fake_call(prompt, system="", client=None):
        captured["prompt"] = prompt
        captured["system"] = system
        return ""

    monkeypatch.setattr(pirjo_pipeline, "_call_openai", fake_call)
    blocks = {"P": "p", "I": "i", "R": "r", "J": "j", "O": "o"}
    pirjo_pipeline.redactor_cientifico(blocks)
    assert "texto coherente" in captured["prompt"]
    assert "experto redactor" in captured["system"]
