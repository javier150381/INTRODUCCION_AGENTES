"""Entry point for the INTRODUCCION_AGENTES application."""

from openai_utils import ensure_openai_api_key
from app import build_demo


def main() -> None:
    """Launch the Gradio PIRJO assistant."""
    ensure_openai_api_key()
    demo = build_demo()
    demo.launch()


if __name__ == "__main__":
    main()
