import os
import sys
import json

# Ensure project root on path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pirjo_pipeline


def test_manager_prompt_mentions_title_and_objective(monkeypatch):
    captured = {}

    def fake_call(prompt, system="", client=None):
        captured["prompt"] = prompt
        return json.dumps({"P": "p", "I": "i", "R": "r", "J": "j", "O": "o"})

    monkeypatch.setattr(pirjo_pipeline, "_call_openai", fake_call)
    blocks = {k: k.lower() for k in "PIRJO"}
    pirjo_pipeline.agente_manager("Título", "Objetivo", blocks)
    assert "Título" in captured["prompt"]
    assert "Objetivo" in captured["prompt"]
