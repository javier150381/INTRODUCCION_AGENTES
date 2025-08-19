import os
from dotenv import load_dotenv


def ensure_openai_api_key():
    """Ensure that ``OPENAI_API_KEY`` is available.

    The function first attempts to load variables from a ``.env`` file using
    :func:`dotenv.load_dotenv`. If the key is still missing, an
    :class:`EnvironmentError` is raised with guidance on how to set the
    variable.

    Raises:
        EnvironmentError: If ``OPENAI_API_KEY`` is not defined.
    """
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is missing. Create a .env file or set the variable manually."
        )
