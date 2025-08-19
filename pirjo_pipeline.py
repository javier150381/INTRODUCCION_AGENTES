import json
import os
from typing import Dict, List

from PyPDF2 import PdfReader
import tiktoken

from openai_utils import ensure_openai_api_key, get_client
from rag_faiss import ensure_index, search_index

client = get_client()


def _call_openai(prompt: str, system: str = "") -> str:
    """Helper to call OpenAI chat completion and return content."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    model = "gpt-3.5-turbo"
    if os.getenv("DEEPSEEK_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        model = "deepseek-chat"
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()


def chunk_text(text: str, chunk_size: int = 700) -> List[str]:
    """Split text into roughly ``chunk_size``-token fragments.

    The function uses ``tiktoken`` to count tokens with the same encoding
    as ``gpt-3.5-turbo``. Chunks are created sequentially without overlap and
    decoded back into strings.
    """

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)
    if not tokens:
        return []

    chunks: List[str] = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i : i + chunk_size]
        chunks.append(encoding.decode(chunk_tokens))
    return chunks


def extract_sources(files: List[str]) -> List[Dict[str, str]]:
    """Extract text from PDFs and split into token chunks with metadata."""

    sources: List[Dict[str, str]] = []
    for path in files:
        reader = PdfReader(path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for c_id, chunk in enumerate(chunk_text(text), start=1):
                sources.append(
                    {
                        "file": os.path.basename(path),
                        "page": page_number,
                        "chunk_id": c_id,
                        "text": chunk,
                    }
                )
    return sources



def analista_de_fuentes(title: str, chunks: List[Dict[str, str]]) -> str:
    """Run the source analysis agent over retrieved chunks and return bullets with citations."""
    compiled = "".join(
        f"[{c['file']}:{c['page']}]\n{c['text']}\n\n" for c in chunks if c["text"].strip()
    )

def analista_de_fuentes(title: str, sources: List[Dict[str, str]]) -> str:
    """Run the analysis agent and return bullets with citations.

    Only as many fragments as fit within ``max_tokens`` are included in the prompt
    sent to the language model, ensuring the request stays within model limits.
    """

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    max_tokens = 12_000
    compiled_parts: List[str] = []
    token_count = 0
    for s in sources:
        if not s["text"].strip():
            continue
        fragment = f"[{s['file']}:{s['page']}:{s['chunk_id']}]\n{s['text']}\n\n"
        frag_tokens = len(encoding.encode(fragment))
        if token_count + frag_tokens > max_tokens:
            break
        compiled_parts.append(fragment)
        token_count += frag_tokens
    compiled = "".join(compiled_parts)

    prompt = (
        f"Título de investigación: {title}\n\n"
        "A partir de los textos con su cita entre corchetes, extrae conceptos, datos y hallazgos "
        "relevantes. Responde en viñetas breves y termina cada viñeta con la cita correspondiente."
    ) + "\n\n" + compiled
    return _call_openai(prompt, system="Agente Analista de Fuentes")


def metodologo_pirjo(bullets: str) -> Dict[str, str]:
    """Transform bullets into PIRJO blocks."""
    prompt = (
        "Convierte las viñetas siguientes en bloques PIRJO. Cada bloque debe contener 2-3 "
        "oraciones claras. Responde estrictamente en JSON con las claves P, I, R, J, O.\n\n"
        f"Viñetas:\n{bullets}"
    )
    content = _call_openai(prompt, system="Agente Metodólogo PIRJO")
    try:
        blocks = json.loads(content)
    except json.JSONDecodeError:
        # if parsing fails, return content under a generic key
        blocks = {"error": content}
    return blocks


def redactor_academico(blocks: Dict[str, str]) -> str:
    """Create the final introduction from PIRJO blocks."""
    prompt = (
        "Eres un redactor académico. Con los bloques PIRJO dados en formato JSON, escribe una "
        "introducción cohesionada de 2-3 párrafos en estilo formal. Asegúrate de incluir las "
        "citas cuando se proporcionen en los bloques.\n\n" + json.dumps(blocks, ensure_ascii=False)
    )
    return _call_openai(prompt, system="Agente Redactor Académico")


def retrieve_relevant_chunks(
    title: str,
    sources: List[Dict[str, str]],
    k: int = 5,
) -> List[Dict[str, str]]:
    """Retrieve ``k`` chunks relevant to ``title`` using a FAISS index."""
    index, metadata = ensure_index(sources)
    return search_index(title, k, index, metadata)


def generate_introduction(title: str, file_paths: List[str]) -> Dict[str, str]:
    """Orchestrate the PIRJO pipeline and return results."""
    ensure_openai_api_key()
    sources = extract_sources(file_paths)
    chunks = retrieve_relevant_chunks(title, sources)
    bullets = analista_de_fuentes(title, chunks)
    blocks = metodologo_pirjo(bullets)
    introduction = redactor_academico(blocks)
    return {
        "introduction": introduction,
        "blocks": blocks,
        "files": [os.path.basename(p) for p in file_paths],
    }
