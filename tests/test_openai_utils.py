import os
import sys
import importlib

import pytest

# Ensure project root on path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import openai_utils


def test_raises_when_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EnvironmentError):
        openai_utils.ensure_openai_api_key()


def test_passes_when_key_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    openai_utils.ensure_openai_api_key()
