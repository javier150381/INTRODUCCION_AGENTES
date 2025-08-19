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
