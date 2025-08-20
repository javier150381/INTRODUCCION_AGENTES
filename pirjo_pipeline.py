import json
import os
from functools import lru_cache
from typing import Dict, List

from PyPDF2 import PdfReader
import tiktoken

from openai_utils import ensure_openai_api_key, get_client
from rag_faiss import ensure_index, search_index


@lru_cache(maxsize=1)
def _get_client():
    """Return a cached OpenAI-compatible client."""
    return get_client()


def _call_openai(prompt: str, system: str = "", client=None) -> str:
    """Helper to call OpenAI chat completion and return content.

    A client is created lazily on first use to avoid side effects at import time.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    model = "gpt-3.5-turbo"
    if os.getenv("DEEPSEEK_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        model = "deepseek-chat"
    if client is None:
        client = _get_client()
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


def analista_de_fuentes(
    title: str, objective: str, summary: str, chunks: List[Dict[str, str]]
) -> str:

    """Run the analysis agent and return bullets with citations.

    Only as many fragments as fit within ``max_tokens`` are included in the prompt
    sent to the language model, ensuring the request stays within model limits.
    """

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    max_tokens = 12_000
    compiled_parts: List[str] = []
    token_count = 0
    for c in chunks:
        if not c["text"].strip():
            continue
        # ``search_index`` may return metadata with either ``chunk_id`` or
        # ``chunk`` as the identifier field. Support both to avoid ``KeyError``
        # when the caller provides one or the other.
        chunk_label = c.get("chunk_id", c.get("chunk", ""))
        fragment = f"[{c['file']}:{c['page']}:{chunk_label}]\n{c['text']}\n\n"
        frag_tokens = len(encoding.encode(fragment))
        if token_count + frag_tokens > max_tokens:
            break
        compiled_parts.append(fragment)
        token_count += frag_tokens
    compiled = "".join(compiled_parts)

    prompt = (
        f"Título de investigación: {title}\n"
        f"Objetivo: {objective}\n"
        f"Resumen: {summary}\n\n"
        "A partir de los textos con su cita entre corchetes, extrae conceptos, datos y hallazgos "
        "relevantes. Responde en viñetas breves y termina cada viñeta con la cita correspondiente."
    ) + "\n\n" + compiled
    return _call_openai(prompt, system="Agente Analista de Fuentes")


def metodologo_pirjo(bullets: str) -> Dict[str, str]:
    """Transform bullets into PIRJO blocks with individual JSON calls.

    Each block (P, I, R, J y O) is requested separately from the language
    model, which must respond with a JSON object containing only the
    corresponding key. The resulting values are gathered into a single
    dictionary for downstream use.
    """

    bloques = {
        "P": "Problema",
        "I": "Información relevante",
        "R": "Restricción o brecha",
        "J": "Justificación",
        "O": "Objetivo",
    }
    results: Dict[str, str] = {}
    for clave, nombre in bloques.items():
        prompt = (
            f"Convierte las viñetas siguientes en el bloque {clave} ({nombre}) con 2-3 "
            f"oraciones claras. Responde estrictamente en JSON con la clave \"{clave}\". "
            "Mantén las citas entre corchetes exactamente como aparecen.\n\n"
            f"Viñetas:\n{bullets}"
        )
        content = _call_openai(prompt, system="Agente Metodólogo PIRJO")
        try:
            parsed = json.loads(content)
            results[clave] = parsed.get(clave, content)
        except json.JSONDecodeError:
            results[clave] = content
    return results


def redactor_academico(blocks: Dict[str, str]) -> str:
    """Create the final introduction from PIRJO blocks."""
    prompt = (
        "Eres un redactor académico. Con los bloques PIRJO dados en formato JSON, redacta una "
        "introducción de cinco párrafos, un párrafo para cada bloque (P, I, R, J y O) y en ese "
        "orden inalterable. Mantén un estilo formal e incluye las citas cuando se proporcionen."\
        "\n\n" + json.dumps(blocks, ensure_ascii=False)
    )
    return _call_openai(prompt, system="Agente Redactor Académico")


def revisor_citas_referencias(text: str) -> str:
    """Review text and ensure citations and references in APA 7 format."""
    prompt = (
        "Eres un revisor académico. Verifica que el texto incluya citas en el cuerpo "
        "y agrega una sección final titulada \"Referencias\" con las entradas en formato APA 7.\n\n"
        f"Texto:\n{text}"
    )
    return _call_openai(prompt, system="Agente Revisor Académico")


def verificador_bibliografia(text: str, sources: List[Dict[str, str]]) -> str:
    """Append a reference list based on citations present in ``text``.

    The function searches for citation labels of the form
    ``[file:page:chunk]`` and only includes those that match the provided
    ``sources``. The reference list simply enumerates the unique file names
    used in the citations to ensure no bibliographic entries are invented.
    """

    import re

    pattern = r"\[([^\[\]]+)\]"
    citations = re.findall(pattern, text)
    valid_keys = {
        f"{s['file']}:{s['page']}:{s.get('chunk_id', s.get('chunk', ''))}"
        for s in sources
    }
    used_files: List[str] = []
    for cit in citations:
        if cit in valid_keys:
            file = cit.split(":")[0]
            if file not in used_files:
                used_files.append(file)
    if not used_files:
        return text
    refs = "\n".join(f"- {f}" for f in used_files)
    return f"{text}\n\nReferencias\n{refs}"


def retrieve_relevant_chunks(
    title: str,
    sources: List[Dict[str, str]],
    k: int = 5,
) -> List[Dict[str, str]]:
    """Retrieve ``k`` chunks relevant to ``title`` using a FAISS index."""
    index, metadata = ensure_index(sources)
    return search_index(title, k, index, metadata)


def generate_introduction(
    title: str, objective: str, summary: str, file_paths: List[str]
) -> Dict[str, str]:
    """Orchestrate the PIRJO pipeline and return results."""
    ensure_openai_api_key()
    sources = extract_sources(file_paths)
    query = " ".join([title, summary, objective]).strip()
    chunks = retrieve_relevant_chunks(query, sources)
    bullets = analista_de_fuentes(title, objective, summary, chunks)
    blocks = metodologo_pirjo(bullets)
    introduction = redactor_academico(blocks)
    introduction = verificador_bibliografia(introduction, chunks)
    return {
        "introduction": introduction,
        "blocks": blocks,
        "files": [os.path.basename(p) for p in file_paths],
    }
