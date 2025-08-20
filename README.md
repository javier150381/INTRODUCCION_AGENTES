# INTRODUCCION_AGENTES

Se desarrollará un Asistente de Introducciones de Investigación (PIRJO). El sistema analizará PDFs proporcionados y generará una introducción académica de cinco párrafos siguiendo los bloques PIRJO (Problema, Información relevante, Restricción, Justificación y Objetivo).

## Resumen

PIRJO analizará PDFs de artículos o tesis y generará de forma automática una introducción académica estructurada en cinco párrafos según los bloques PIRJO.

## Detalles de implementación

El módulo `metodologo_pirjo` solicitará al modelo un JSON independiente para cada bloque (P, I, R, J y O) y luego combinará las respuestas antes de redactar la introducción final.

Para evitar referencias inventadas, la etapa de revisión se sustituirá por un verificador
que extraerá las citas directamente de los fragmentos recuperados en la base FAISS y
construirá una sección de *Referencias* únicamente con los nombres de los PDFs
utilizados.

## Objetivo

El objetivo será ofrecer una herramienta que agilice la redacción de introducciones de investigación a partir de la información extraída de los documentos proporcionados.

## Configuración

Este proyecto requiere la variable de entorno `OPENAI_API_KEY` para acceder a la API de OpenAI:

```bash
export OPENAI_API_KEY="tu_api_key"
```

En PowerShell (Windows) puedes establecerla con:

```powershell
setx OPENAI_API_KEY "tu_api_key"
```

Después reinicia la terminal para que el cambio surta efecto. Verifica que la variable esté disponible con:

```powershell
echo $env:OPENAI_API_KEY
```

La aplicación mostrará un error si la clave sigue sin configurarse.

Este proyecto fija la dependencia `openai` en la versión `0.28.1` porque el código utiliza la API legacy `ChatCompletion`. Si se migra a la nueva versión (`openai>=1.0.0`), será necesario actualizar las llamadas a la API.

Instala las dependencias necesarias ejecutando:

```bash
pip install -r requirements.txt
```

## Uso

Se podrá ejecutar la aplicación Gradio:

```bash
python main.py
```

La interfaz permitirá ingresar el título del trabajo, el objetivo, un breve resumen y subir archivos PDF para obtener la introducción final, los bloques PIRJO intermedios y la lista de documentos procesados.

## Pruebas

Para ejecutar las pruebas unitarias:

```bash
pytest
```
