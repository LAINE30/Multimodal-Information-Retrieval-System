# Arquitectura y Flujo del Sistema (RAG Multimodal)

Este documento explica paso a paso cómo funciona el proyecto, desde que se descargan los datos hasta que el usuario obtiene una respuesta en la interfaz de chat.

## Conceptos Clave

Antes de entender el flujo, es vital conocer tres conceptos fundamentales que le dan vida al sistema:

1. **CLIP (Contrastive Language-Image Pretraining):** Es un modelo de Inteligencia Artificial creado por OpenAI. Su "magia" es que puede "leer" texto y "ver" imágenes, y convertir ambos en **Vectores (Embeddings)** en un mismo espacio matemático. Si tienes una foto de un perro y la palabra "Perro", CLIP los convertirá en dos vectores que estarán matemáticamente muy cerca el uno del otro.
2. **ChromaDB (Base de Datos Vectorial):** A diferencia de las bases de datos tradicionales (SQL) que buscan palabras exactas, ChromaDB guarda listas de números (los vectores de CLIP) y nos permite buscar por **similitud semántica** midiendo la distancia entre ellos.
3. **RAG (Retrieval-Augmented Generation):** Es una técnica donde, antes de pedirle a un LLM (como Gemini) que responda algo, primero buscamos "Evidencias" en nuestra base de datos y se las "inyectamos" en el prompt. Así evitamos que la IA alucine e invente cosas, obligándola a responder basándose solo en los datos de nuestra tienda.

---

## Flujo del Proyecto: Paso a Paso

El sistema se divide en dos grandes fases: **1. Indexación (Preparación)** y **2. Inferencia (Ejecución de consultas)**.

### FASE 1: Preparación de Datos (Indexación)

Esta fase ocurre "tras bambalinas" cuando el desarrollador corre los scripts por primera vez.

#### Paso 1: Descarga y Procesamiento (`data_processing.py`)
- El script se conecta al dataset público de Amazon Reviews 2023.
- Extrae un subconjunto de productos (por defecto 250), obteniendo sus títulos, características, categorías y URLs de imágenes.
- Descarga físicamente las imágenes a la carpeta `data/raw/images/`.
- Guarda todos los metadatos en un archivo estructurado: `data/processed/corpus.json`.

#### Paso 2: Generación de Embeddings e Indexación (`index_corpus.py` y `vector_db.py`)
- El script de indexación lee el `corpus.json`.
- Para cada producto, une el título y sus características en un solo texto.
- Envía tanto el texto como la imagen física al módulo `embeddings.py` (donde vive el modelo CLIP). CLIP devuelve un vector (una lista de números) que representa el significado de ese producto.
- Finalmente, se envía este vector junto con los datos (ID, texto descriptivo, ruta de la imagen, categoría) al módulo `vector_db.py`, el cual lo guarda permanentemente en la carpeta `data/chroma_db`.

> **Resultado de la Fase 1:** Tenemos una base de datos local poblada de productos, donde cada producto tiene una representación matemática multimodal (mezcla texto + imagen).

---

### FASE 2: Ejecución de Consultas (Inferencia RAG)

Esta es la fase interactiva, donde el usuario final utiliza la aplicación web (`app.py`).

#### Paso 1: Entrada del Usuario (`app.py`)
- El usuario abre Streamlit y escribe una consulta en lenguaje natural (ej. *"Busca una guitarra acústica para principiantes"*).

#### Paso 2: Recuperación / Retrieval (`retrieval.py`)
- La consulta del usuario se envía al `MultimodalRetriever`.
- El Retriever pasa la consulta de texto por CLIP para convertirla en un vector matemático.
- Luego, hace una consulta a **ChromaDB**: *"Devuélveme los 3 productos cuyos vectores estén matemáticamente más cerca de este vector de consulta"*.
- ChromaDB devuelve los Top-K resultados (evidencias), que incluyen el texto original, la ruta de la imagen y el "Score" de similitud.

#### Paso 3: Generación de Respuesta (`generation.py`)
- Los documentos recuperados se envían a `RAGGenerator`.
- El sistema construye un **Contexto** concatenando el texto de esos 3 productos.
- Se crea un "Prompt" (Instrucción) para **Gemini-2.5-flash** que dice esencialmente: *"Eres un asistente de compras. Responde a la pregunta del usuario usando ÚNICAMENTE esta información de contexto: [Contexto recuperado]"*.
- Gemini lee el contexto, razona, y formula una respuesta natural y amable.

#### Paso 4: Presentación Final (`app.py`)
- La interfaz de Streamlit muestra la respuesta generada por Gemini en el chat.
- Inmediatamente debajo de la respuesta, extrae las rutas de las imágenes de las "evidencias" y las renderiza en la pantalla junto con la información detallada, permitiendo al usuario "ver" de dónde sacó la información la Inteligencia Artificial.
