# INTRODUCCION_AGENTES

Asistente de Introducciones de Investigación (PIRJO). El sistema analiza PDFs proporcionados y genera una introducción académica siguiendo los bloques PIRJO (Propósito, Importancia, Relevancia, Justificación, Originalidad).

## Configuración

Este proyecto requiere la variable de entorno `OPENAI_API_KEY` para acceder a la API de OpenAI:

```bash
export OPENAI_API_KEY="tu_api_key"
```

Instala las dependencias necesarias ejecutando:

```bash
pip install -r requirements.txt
```

## Uso

Ejecuta la aplicación Gradio:

```bash
python main.py
```

La interfaz permite ingresar el título del trabajo, subir archivos PDF y obtener la introducción final, los bloques PIRJO intermedios y la lista de documentos procesados.

## Pruebas

Para ejecutar las pruebas unitarias:

```bash
pytest
```
