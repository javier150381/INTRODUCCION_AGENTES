from typing import List

import gradio as gr

from pirjo_pipeline import generate_introduction


def run_pipeline(title: str, objective: str, summary: str, files: List[gr.File]) -> tuple:
    """Execute the PIRJO pipeline and format outputs for Gradio."""
    file_paths = [f.name for f in files] if files else []
    if not title or not objective or not summary or not file_paths:
        return "Se requiere título, objetivo, resumen y al menos un PDF.", "", ""
    result = generate_introduction(title, objective, summary, file_paths)
    blocks_text = "\n".join(f"{k}: {v}" for k, v in result["blocks"].items())
    processed = ", ".join(result["files"])
    return result["introduction"], blocks_text, processed


def build_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("### Asistente de Introducciones de Investigación (PIRJO)")
        with gr.Row():
            title = gr.Textbox(label="Título del trabajo")
            objective = gr.Textbox(label="Objetivo del artículo")
            summary = gr.Textbox(label="Resumen del artículo", lines=2)
            pdfs = gr.File(label="PDFs", file_count="multiple", file_types=[".pdf"])
        btn = gr.Button("Generar Introducción")
        intro = gr.Textbox(label="Introducción", lines=8)
        blocks = gr.Textbox(label="Bloques PIRJO", lines=8)
        files_out = gr.Textbox(label="Archivos procesados")
        btn.click(run_pipeline, inputs=[title, objective, summary, pdfs], outputs=[intro, blocks, files_out])
    return demo


if __name__ == "__main__":
    build_demo().launch()
