import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pirjo_pipeline


def test_redactor_prompt_mentions_five_paragraphs(monkeypatch):
    captured = {}

    def fake_call(prompt, system="", client=None):
        captured["prompt"] = prompt
        return ""

    monkeypatch.setattr(pirjo_pipeline, "_call_openai", fake_call)
    blocks = {"P": "p", "I": "i", "R": "r", "J": "j", "O": "o"}
    pirjo_pipeline.redactor_academico(blocks)
    assert "cinco párrafos" in captured["prompt"]
    assert "orden inalterable" in captured["prompt"]
    assert "500 y 700 palabras" in captured["prompt"]
