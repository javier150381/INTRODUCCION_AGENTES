import os


def ensure_openai_api_key():
    """Verifica que la variable OPENAI_API_KEY esté configurada.

    Raises:
        EnvironmentError: si la variable no existe o está vacía.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Falta OPENAI_API_KEY en variables de entorno.")
