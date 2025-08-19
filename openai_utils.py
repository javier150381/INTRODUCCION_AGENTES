import os
from dotenv import load_dotenv


def ensure_openai_api_key() -> None:
    """Ensure that an API key for OpenAI or DeepSeek is available.

    The function first attempts to load variables from a ``.env`` file using
    :func:`dotenv.load_dotenv`. If neither ``OPENAI_API_KEY`` nor
    ``DEEPSEEK_API_KEY`` is defined, an :class:`EnvironmentError` is raised with
    guidance on how to set the variable.

    Raises:
        EnvironmentError: If no API key is defined.
    """
    load_dotenv()
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")):
        raise EnvironmentError(
            "Set OPENAI_API_KEY or DEEPSEEK_API_KEY in a .env file or the environment.",
        )


def get_client():
    """Return an OpenAI-compatible client for OpenAI or DeepSeek.

    Prefers OpenAI when ``OPENAI_API_KEY`` is present; otherwise attempts to use
    DeepSeek via its OpenAI-compatible endpoint.
    """
    ensure_openai_api_key()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    from openai import OpenAI  # Imported here to avoid dependency during tests
    if os.getenv("OPENAI_API_KEY"):
        return OpenAI(api_key=api_key)
    # DeepSeek uses an OpenAI-compatible API
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
