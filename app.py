from typing import List

import gradio as gr

from pirjo_pipeline import generate_introduction


def run_pipeline(title: str, objective: str, summary: str, files: List[gr.File]) -> tuple:
    """Execute the PIRJO pipeline and format outputs for Gradio.

    The introduction is returned as a single string while the PIRJO blocks are
    rendered in a human friendly way (one block per section with the full
    Spanish label) instead of raw JSON.
    """

    file_paths = [f.name for f in files] if files else []
    if not title or not objective or not summary or not file_paths:
        return "Se requiere título, objetivo, resumen y al menos un PDF.", "", ""

    result = generate_introduction(title, objective, summary, file_paths)

    block_labels = {
        "P": "Problema",
        "I": "Información relevante",
        "R": "Restricción o brecha",
        "J": "Justificación",
        "O": "Objetivo",
    }
    blocks_text = "\n\n".join(
        f"{block_labels.get(k, k)}:\n{v}" for k, v in result["blocks"].items()
    )

    processed = ", ".join(result["files"])
    return result["introduction"], blocks_text, processed


def export_to_docx(text: str) -> str:
    """Generate a DOCX file from the provided text and return its path."""
    from docx import Document
    import tempfile
    import os

    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    tmp_dir = tempfile.mkdtemp()
    file_path = os.path.join(tmp_dir, "resultado.docx")
    doc.save(file_path)
    return file_path


def export_to_pdf(text: str) -> str:
    """Generate a simple PDF file from the provided text and return its path."""
    from fpdf import FPDF
    import tempfile
    import os

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    tmp_dir = tempfile.mkdtemp()
    file_path = os.path.join(tmp_dir, "resultado.pdf")
    pdf.output(file_path)
    return file_path


def build_demo() -> gr.Blocks:
    with gr.Blocks(css=".scrollable textarea {overflow-y: auto; max-height: 500px;}") as demo:
        gr.Markdown("### Asistente de Introducciones de Investigación (PIRJO)")
        with gr.Row():
            title = gr.Textbox(label="Título del trabajo")
            objective = gr.Textbox(label="Objetivo del artículo")
            summary = gr.Textbox(label="Resumen del artículo", lines=2)
            pdfs = gr.File(label="PDFs", file_count="multiple", file_types=[".pdf"])
        btn = gr.Button("Generar Introducción")
        intro = gr.Textbox(
            label="Resultado final",
            lines=20,
            max_lines=20,
            autoscroll=True,
            elem_classes="scrollable",
        )
        blocks = gr.Textbox(label="Bloques PIRJO", lines=8)
        files_out = gr.Textbox(label="Archivos procesados")
        download_word = gr.File(label="Descargar Word")
        download_pdf = gr.File(label="Descargar PDF")
        export_word = gr.Button("Exportar a Word")
        export_pdf = gr.Button("Exportar a PDF")
        btn.click(run_pipeline, inputs=[title, objective, summary, pdfs], outputs=[intro, blocks, files_out])
        export_word.click(export_to_docx, inputs=intro, outputs=download_word)
        export_pdf.click(export_to_pdf, inputs=intro, outputs=download_pdf)
    return demo


if __name__ == "__main__":
    build_demo().launch()
