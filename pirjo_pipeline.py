import json
import os
from typing import List, Dict

from openai import OpenAI
from PyPDF2 import PdfReader

from openai_utils import ensure_openai_api_key

client = OpenAI()


def _call_openai(prompt: str, system: str = "") -> str:
    """Helper to call OpenAI chat completion and return content."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    return response.choices[0].message.content.strip()


def extract_sources(files: List[str]) -> List[Dict[str, str]]:
    """Extract text and citation info from PDF files."""
    sources = []
    for path in files:
        reader = PdfReader(path)
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            sources.append({"file": os.path.basename(path), "page": idx, "text": text})
    return sources


def analista_de_fuentes(title: str, sources: List[Dict[str, str]]) -> str:
    """Run the source analysis agent and return bullets with citations."""
    compiled = "".join(
        f"[{s['file']}:{s['page']}]\n{s['text']}\n\n" for s in sources if s["text"].strip()
    )
    prompt = (
        f"Título de investigación: {title}\n\n"
        "A partir de los textos con su cita entre corchetes, extrae conceptos, datos y hallazgos "
        "relevantes. Responde en viñetas breves y termina cada viñeta con la cita correspondiente."\
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


def generate_introduction(title: str, file_paths: List[str]) -> Dict[str, str]:
    """Orchestrate the PIRJO pipeline and return results."""
    ensure_openai_api_key()
    sources = extract_sources(file_paths)
    bullets = analista_de_fuentes(title, sources)
    blocks = metodologo_pirjo(bullets)
    introduction = redactor_academico(blocks)
    return {
        "introduction": introduction,
        "blocks": blocks,
        "files": [os.path.basename(p) for p in file_paths],
    }
