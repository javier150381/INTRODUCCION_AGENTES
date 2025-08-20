import json
import os
from functools import lru_cache
from typing import Dict, List, Tuple

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


def _parse_year(date_str: str) -> str:
    """Extract a year from a PDF metadata date string."""
    if not date_str:
        return ""
    digits = "".join(ch for ch in date_str if ch.isdigit())
    return digits[:4]


def extract_sources(files: List[str]) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, str]]]:
    """Extract text and metadata from PDFs.

    Returns a tuple ``(sources, metadata)`` where ``sources`` contains token
    chunks with citation information and ``metadata`` maps file names to basic
    bibliographic fields (author, title, year).
    """

    sources: List[Dict[str, str]] = []
    metadata: Dict[str, Dict[str, str]] = {}
    for path in files:
        reader = PdfReader(path)
        info = reader.metadata or {}
        fname = os.path.basename(path)
        metadata[fname] = {
            "author": info.get("/Author", ""),
            "title": info.get("/Title", os.path.splitext(fname)[0]),
            "year": _parse_year(info.get("/CreationDate", "")),
        }
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for c_id, chunk in enumerate(chunk_text(text), start=1):
                sources.append(
                    {
                        "file": fname,
                        "page": page_number,
                        "chunk_id": c_id,
                        "text": chunk,
                    }
                )
    return sources, metadata


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
    corresponding key. A dedicated agent role is used for every block to
    keep responsibilities isolated. The resulting values are gathered into a
    single dictionary for downstream use.
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
        system = f"Agente {clave} - {nombre}"
        content = _call_openai(prompt, system=system)
        try:
            parsed = json.loads(content)
            results[clave] = parsed.get(clave, content)
        except json.JSONDecodeError:
            results[clave] = content
    return results


def agente_manager(title: str, objective: str, blocks: Dict[str, str]) -> Dict[str, str]:
    """Ensure PIRJO blocks align with title and objective."""
    prompt = (
        f"Título: {title}\nObjetivo: {objective}\n\n"
        "Revisa la coherencia de los siguientes bloques PIRJO con el título y el objetivo. "
        "Si detectas inconsistencias, corrígelas. Responde en JSON con las claves P, I, R, J y O.\n\n"
        f"Bloques:\n{json.dumps(blocks, ensure_ascii=False)}"
    )
    system = "Agente Manager"
    content = _call_openai(prompt, system=system)
    try:
        parsed = json.loads(content)
        return {k: parsed.get(k, blocks.get(k, "")) for k in blocks}
    except json.JSONDecodeError:
        return blocks


def redactor_academico(blocks: Dict[str, str]) -> str:
    """Create the final introduction from PIRJO blocks."""
    prompt = (
        "Eres un redactor académico. Con los bloques PIRJO dados en formato JSON, redacta una "
        "introducción de cinco párrafos, un párrafo para cada bloque (P, I, R, J y O) y en ese "
        "orden inalterable. Mantén un estilo formal, incluye las citas cuando se proporcionen y "
        "asegúrate de que el texto completo tenga entre 500 y 700 palabras.\n\n"
        + json.dumps(blocks, ensure_ascii=False)
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


def verificador_bibliografia(
    text: str, sources: List[Dict[str, str]], metadata: Dict[str, Dict[str, str]]
) -> str:
    """Append a reference list based on citations present in ``text``.

    The function searches for citation labels of the form ``[file:page:chunk]``
    and only includes those that match the provided ``sources``. Bibliographic
    entries are formatted using metadata extracted from the PDFs to avoid
    inventing references.
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

    def _format_apa(fname: str) -> str:
        meta = metadata.get(fname, {})
        author = meta.get("author") or "s.a."
        year = meta.get("year") or "s.f."
        title = meta.get("title") or fname
        return f"{author} ({year}). {title}."

    refs = "\n".join(f"- {_format_apa(f)}" for f in used_files)
    return f"{text}\n\nReferencias\n{refs}"


def retrieve_relevant_chunks(
    title: str,
    objective: str,
    summary: str,
    sources: List[Dict[str, str]],
    k: int = 5,
) -> Tuple[str, List[Dict[str, str]]]:
    """Return a summary of prior studies and the supporting chunks.

    This function searches the FAISS index built from ``sources`` using a
    composite query derived from the research ``title``, ``objective`` and
    ``summary``. The top ``k`` matching fragments are then analysed by the
    ``analista_de_fuentes`` agent to extract relevant findings, which are
    returned as bullet points with citations. Both the bullet string and the
    underlying chunk metadata are provided so that later stages can verify
    references.
    """

    query = " ".join([title, summary, objective]).strip()
    index, metadata = ensure_index(sources)
    chunks = search_index(query, k, index, metadata)
    bullets = analista_de_fuentes(title, objective, summary, chunks)
    return bullets, chunks


def generate_introduction(
    title: str, objective: str, summary: str, file_paths: List[str]
) -> Dict[str, str]:
    """Orchestrate the PIRJO pipeline and return results."""
    ensure_openai_api_key()
    sources, metadata = extract_sources(file_paths)
    bullets, chunks = retrieve_relevant_chunks(title, objective, summary, sources)
    blocks = metodologo_pirjo(bullets)
    blocks = agente_manager(title, objective, blocks)
    introduction = redactor_academico(blocks)
    introduction = revisor_citas_referencias(introduction)
    introduction = verificador_bibliografia(introduction, chunks, metadata)
    return {
        "introduction": introduction,
        "blocks": blocks,
        "files": [os.path.basename(p) for p in file_paths],
    }
