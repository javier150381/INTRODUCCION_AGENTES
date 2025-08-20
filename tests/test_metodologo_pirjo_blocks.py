import os
import sys
import json
import re

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pirjo_pipeline


def test_metodologo_pirjo_makes_separate_calls(monkeypatch):
    prompts = []

    def fake_call(prompt, system="", client=None):
        prompts.append(prompt)
        match = re.search(r"bloque ([PIRJO])", prompt)
        letter = match.group(1) if match else "X"
        return json.dumps({letter: letter.lower()})

    monkeypatch.setattr(pirjo_pipeline, "_call_openai", fake_call)
    bullets = "- ejemplo"
    blocks = pirjo_pipeline.metodologo_pirjo(bullets)

    assert len(prompts) == 5
    assert blocks == {k: k.lower() for k in "PIRJO"}


def test_metodologo_prompt_mentions_citations(monkeypatch):
    captured = {}

    def fake_call(prompt, system="", client=None):
        captured.setdefault("prompts", []).append(prompt)
        return json.dumps({"P": "p"})

    monkeypatch.setattr(pirjo_pipeline, "_call_openai", fake_call)
    pirjo_pipeline.metodologo_pirjo("- ejemplo [f:1:1]")
    assert any("citas" in p for p in captured["prompts"])


def test_metodologo_prompt_for_I_requests_mini_summaries(monkeypatch):
    prompts = []

    def fake_call(prompt, system="", client=None):
        prompts.append(prompt)
        match = re.search(r"bloque ([PIRJO])", prompt)
        letter = match.group(1) if match else "X"
        return json.dumps({letter: letter.lower()})

    monkeypatch.setattr(pirjo_pipeline, "_call_openai", fake_call)
    pirjo_pipeline.metodologo_pirjo("- ejemplo [f:1:1]")

    i_prompt = prompts[1].lower()
    assert "mini-res" in i_prompt and "fuente" in i_prompt
