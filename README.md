# Sistema de Recuperación de Información Multimodal con RAG

Este proyecto implementa un sistema de Recuperación de Información Multimodal (RIM) utilizando arquitecturas *Retrieval-Augmented Generation* (RAG). Permite responder consultas conversacionales sobre un corpus compuesto por texto e imágenes.

## Características Principales

*   **Corpus Multimodal**: Emplea un subconjunto del **Amazon Reviews 2023 Dataset**, que incluye la metadata (títulos, características, descripciones) e imágenes de productos de diferentes categorías (como Electrónica, Videojuegos, Instrumentos Musicales).
*   **Embeddings Multimodales**: Utiliza modelos basados en CLIP para representar tanto el texto como las imágenes en un mismo espacio vectorial.
*   **Base de Datos Vectorial**: Almacena e indexa los embeddings utilizando FAISS o ChromaDB para una recuperación eficiente (búsqueda Top-k).
*   **Pipeline RAG**: Implementa un flujo completo: recibe la consulta, recupera documentos relevantes, construye el contexto y genera una respuesta utilizando un LLM.
*   **Interfaz Conversacional**: Interfaz web construida con Streamlit que permite realizar consultas, visualizar respuestas y examinar las evidencias (documentos e imágenes recuperadas) utilizadas para la generación.

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

## Estructura del Proyecto

```text
├── data/                  # Almacenamiento de datos (corpus, embeddings)
│   ├── raw/               # Corpus original (texto e imágenes)
│   ├── processed/         # Datos preprocesados
│   └── embeddings/        # Vectores generados por CLIP
├── src/                   # Código fuente de la aplicación
│   ├── app.py             # Interfaz web principal en Streamlit
│   ├── data_processing.py # Scripts para cargar y procesar el corpus
│   ├── embeddings.py      # Generación de representaciones vectoriales
│   ├── vector_db.py       # Configuración y consultas a FAISS/ChromaDB
│   ├── retrieval.py       # Lógica de búsqueda vectorial (Top-k)
│   └── generation.py      # Generación de respuestas (LLM)
├── tests/                 # Tests unitarios para los módulos
├── .agents/               # Reglas y contexto para asistentes de IA (AGENTS.md)
├── .env.example           # Ejemplo de variables de entorno
├── requirements.txt       # Dependencias del proyecto
└── README.md              # Documentación del proyecto
```

## Ejecución

Para iniciar la interfaz web conversacional, ejecuta el siguiente comando desde la raíz del proyecto:

```bash
streamlit run src/app.py
```

## Estado Actual (Progreso)
- ✅ **Fase A (Corpus)**: Dataset de Amazon 2023 con metadatos e imágenes descargados y estructurados.
- ✅ **Fase B (Embeddings)**: Integración del modelo CLIP de Hugging Face para generar vectores multimodales.
- ✅ **Fase C (VectorDB)**: Configuración e inicialización de ChromaDB para guardar índices vectoriales y metadatos localmente.
- ✅ **Fase D (Pipeline RAG)**: Retriever conectando queries con ChromaDB + Generator con LangChain (OpenAI).
- ✅ **Fase E (Interfaz Web)**: Streamlit chat implementado con capacidad de mostrar evidencia visual.
- ⏳ **Siguientes Pasos (Fase Excelencia)**:
  - Implementar Re-Ranking (Cross-Encoder)
  - Añadir Memoria Conversacional (Chat History con LLM)
  - Testeo intensivo y ajuste fino

## Desarrollo y Tests

Para ejecutar las pruebas automatizadas del proyecto:

```bash
pytest tests/
```
