import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pirjo_pipeline


def test_revisor_prompt_mentions_apa(monkeypatch):
    captured = {}

    def fake_call(prompt, system="", client=None):
        captured["prompt"] = prompt
        return ""

    monkeypatch.setattr(pirjo_pipeline, "_call_openai", fake_call)
    pirjo_pipeline.revisor_citas_referencias("texto")
    assert "APA 7" in captured["prompt"]
    assert "Referencias" in captured["prompt"]
