import os
import sys
import importlib

# Ensure project root on path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import openai_utils


def test_import_has_no_side_effects(monkeypatch):
    called = False

    def fake_get_client():
        nonlocal called
        called = True

    monkeypatch.setattr(openai_utils, "get_client", fake_get_client)
    sys.modules.pop("pirjo_pipeline", None)
    importlib.import_module("pirjo_pipeline")
    assert not called
