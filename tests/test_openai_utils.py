import os
import sys
from pathlib import Path

import pytest

# Ensure project root on path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import openai_utils


def test_raises_when_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(EnvironmentError) as excinfo:
        openai_utils.ensure_openai_api_key()
    # The error message should guide the user to create a .env file or set the variable
    assert ".env" in str(excinfo.value)


def test_passes_when_key_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    openai_utils.ensure_openai_api_key()


def test_passes_when_deepseek_key_present(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    openai_utils.ensure_openai_api_key()


def test_loads_key_from_env_file(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    env_file = Path(".env")
    env_file.write_text("OPENAI_API_KEY=from_env_file\n")
    try:
        openai_utils.ensure_openai_api_key()
        assert os.getenv("OPENAI_API_KEY") == "from_env_file"
    finally:
        env_file.unlink(missing_ok=True)
