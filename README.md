# Sistema de Recuperación de Información Multimodal con RAG

Este proyecto implementa un sistema de Recuperación de Información Multimodal (RIM) utilizando arquitecturas *Retrieval-Augmented Generation* (RAG). Permite responder consultas conversacionales sobre un corpus compuesto por texto e imágenes.

## Características Principales

*   **Corpus Multimodal Ampliado**: Emplea un subconjunto del **Amazon Reviews 2023 Dataset**, que incluye más de 2,100 productos distribuidos en 20 categorías (Electrónica, Videojuegos, Instrumentos, Juguetes, Celulares, Hogar, Software, etc.) junto con sus metadatos e imágenes.
*   **Embeddings Multimodales**: Utiliza modelos basados en CLIP para representar tanto el texto como las imágenes en un mismo espacio vectorial.
*   **Base de Datos Vectorial**: Almacena e indexa los embeddings utilizando **ChromaDB** para una recuperación eficiente (búsqueda Top-k).
*   **Pipeline RAG Avanzado**: Implementa un flujo completo: recibe la consulta, recupera documentos, aplica **Re-ranking (Cross-Encoder)**, construye el contexto y genera una respuesta utilizando **Gemini 2.5 Flash**.
*   **Query Expansion**: Utiliza técnicas Multi-Query para generar sinónimos y variaciones semánticas de la consulta del usuario, mejorando el *Recall*.
*   **Relevance Feedback**: Sistema interactivo tipo Rocchio que aprende de los votos (👍/👎) del usuario para priorizar o castigar productos en el ranking dinámicamente.
*   **Búsqueda Visual Bidireccional**: Permite subir fotografías para encontrar productos visualmente similares usando los embeddings de CLIP.
*   **Búsqueda por Voz (STT)**: Permite grabar audio desde el navegador y convertirlo a texto usando las capacidades nativas multimodales de la API de Gemini.
*   **Memoria Conversacional**: El Asistente IA recuerda los últimos turnos del chat para resolver ambigüedades en la conversación.
*   **Evaluación del Sistema**: Módulo de evaluación experimental con métricas estándar de Recuperación de Información: Precision@k, Recall@k y NDCG@k.

## Requisitos Previos

*   Python 3.8+
*   Entorno virtual recomendado (`venv` o `conda`)

## Instalación

1.  Clonar el repositorio.
2.  Crear y activar un entorno virtual:
    ```bash
    python -m venv venv
    # En Windows:
    venv\\Scripts\\activate
    # En Linux/Mac:
    source venv/bin/activate
    ```
3.  Instalar las dependencias:
    ```bash
    pip install -r requirements.txt
    ```
4.  Configurar variables de entorno:
    Crea un archivo llamado `.env` en la raíz del proyecto (puedes basarte en `.env.example`) y agrega tu API Key de Google (Gemini):
    ```text
    GOOGLE_API_KEY="tu_clave_aqui"
    ```
5.  Ejecutar el sistema localmente o desplegar (¡Base de datos lista!):
    La base de datos vectorial pre-calculada (`data/chroma_db`) ya se encuentra incluida en este repositorio para que el sistema funcione de inmediato sin necesidad de descargas pesadas.
    
    *Opcional:* Si deseas descargar las imágenes de alta calidad localmente y re-generar el corpus desde cero, ejecuta:
    ```bash
    # Descarga imágenes y metadata (~15 min)
    python src/data_processing.py
    
    # Genera embeddings e indexa en ChromaDB (~3 min)
    $env:PYTHONPATH="."  # En Windows PowerShell
    python src/index_corpus.py
    ```

## Estructura del Proyecto

```text
├── data/                       # Almacenamiento de datos (corpus, embeddings)
│   ├── raw/images/             # Imágenes descargadas del corpus
│   ├── processed/corpus.json   # Datos preprocesados (metadatos + rutas)
│   └── evaluation/             # Queries de evaluación (qrels) y resultados
├── src/                        # Código fuente de la aplicación
│   ├── app.py                  # Interfaz web principal en Streamlit
│   ├── data_processing.py      # Scripts para cargar y procesar el corpus
│   ├── embeddings.py           # Generación de representaciones vectoriales (CLIP)
│   ├── vector_db.py            # Configuración y consultas a ChromaDB
│   ├── retrieval.py            # Lógica de búsqueda vectorial (Top-k)
│   ├── generation.py           # Generación de respuestas (LLM Gemini)
│   ├── index_corpus.py         # Script de indexación del corpus en ChromaDB
│   ├── evaluation_metrics.py   # Métricas: Precision@k, Recall@k, NDCG@k
│   └── evaluate.py             # Script para ejecutar la evaluación completa
├── tests/                      # Tests unitarios para los módulos
├── .agents/                    # Reglas y contexto para asistentes de IA
├── arquitectura_y_flujo.md     # Documentación detallada de la arquitectura
├── .env.example                # Ejemplo de variables de entorno
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Documentación del proyecto
```

## Ejecución

Para iniciar la interfaz web conversacional, ejecuta el siguiente comando desde la raíz del proyecto:

```bash
streamlit run src/app.py
```

## Evaluación del Sistema

Para ejecutar la evaluación experimental del sistema de recuperación con las métricas Precision@k, Recall@k y NDCG@k:

```bash
python src/evaluate.py
```

Los resultados se guardan en `data/evaluation/evaluation_results.json` y se imprimen en consola.

## Desarrollo y Tests

Para ejecutar las pruebas automatizadas del proyecto:

```bash
pytest tests/
```
