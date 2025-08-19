import gradio as gr

with gr.Blocks() as demo:
    with gr.Row():
        titulo = gr.Markdown("Sube tus archivos")
        files = gr.File(label="Seleccionar archivo")

demo.launch()
