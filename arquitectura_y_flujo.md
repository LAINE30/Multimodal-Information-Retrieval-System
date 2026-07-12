# Arquitectura y Flujo del Sistema (RAG Multimodal)

Este documento explica en detalle cómo funciona el proyecto internamente: qué hace cada archivo `.py`, qué clases y funciones contiene, a quién llama, y cómo se conectan entre sí desde que se descargan los datos hasta que el usuario obtiene una respuesta en la interfaz de chat.

---

## Conceptos Clave

Antes de entender el flujo, es vital conocer los conceptos fundamentales que le dan vida al sistema:

1. **Embedding (Vector):** Es una lista de números (ej. 512 floats) que representa el "significado" de un texto o imagen en un espacio matemático. Dos cosas con significados similares tendrán vectores cercanos entre sí.
2. **CLIP (Contrastive Language-Image Pretraining):** Es un modelo de IA creado por OpenAI. Su "magia" es que puede "leer" texto y "ver" imágenes, y convertir ambos en vectores dentro del **mismo espacio matemático**. Si tienes una foto de un perro y la palabra "Perro", CLIP los convertirá en dos vectores muy cercanos entre sí.
3. **ChromaDB (Base de Datos Vectorial):** A diferencia de las bases de datos tradicionales (SQL) que buscan palabras exactas, ChromaDB guarda vectores y nos permite buscar por **similitud semántica** midiendo la distancia coseno entre ellos.
4. **RAG (Retrieval-Augmented Generation):** Técnica donde, antes de pedirle a un LLM (como Gemini) que responda, primero buscamos "evidencias" en nuestra base de datos y se las "inyectamos" en el prompt. Así evitamos que la IA alucine, obligándola a responder basándose solo en nuestros datos reales.
5. **LangChain (LCEL):** Framework que nos permite construir "cadenas" de procesamiento. En nuestro caso, conectamos un `PromptTemplate` → `ChatGoogleGenerativeAI` para crear un pipeline de generación de respuestas.
6. **Similitud Coseno:** Medida matemática que compara dos vectores. Un valor cercano a 1.0 indica que son muy similares semánticamente; cercano a 0.0, que no se parecen.

---

## Mapa de Módulos y sus Relaciones

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FASE 1: INDEXACIÓN                          │
│                    (Scripts que se corren una vez)                  │
│                                                                    │
│   data_processing.py ──────────► corpus.json + imágenes/           │
│                                       │                            │
│   index_corpus.py ◄──────────────────┘                             │
│       │                                                            │
│       ├── CLIPEmbedder (embeddings.py) → genera vectores           │
│       └── VectorDB (vector_db.py) → guarda en ChromaDB             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    FASE 2: INFERENCIA (CONSULTAS)                   │
│                   (Cada vez que el usuario pregunta)                │
│                                                                    │
│   app.py (Streamlit UI)                                            │
│       │                                                            │
│       ├── Texto ──► MultimodalRetriever.retrieve()                 │
│       │                 ├── CLIPEmbedder.embed_text()               │
│       │                 └── VectorDB.search()                       │
│       │                                                            │
│       ├── Imagen ► MultimodalRetriever.retrieve_by_image()         │
│       │                 ├── CLIPEmbedder.embed_image()              │
│       │                 └── VectorDB.search()                       │
│       │                                                            │
│       └── Resultados ──► RAGGenerator.generate_response()          │
│                              ├── format_context()                   │
│                              └── chain.invoke() → Gemini API        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Descripción Detallada de Cada Módulo

### `data_processing.py` — Descarga y Preparación del Corpus

**Propósito:** Conectarse al dataset público de Amazon Reviews 2023 (Hugging Face), descargar un subconjunto de productos con sus imágenes y crear el archivo estructurado `corpus.json`.

**Constantes de configuración:**
- `CATEGORIES`: Lista de 6 categorías del dataset (Instrumentos Musicales, Videojuegos, Mascotas, Cámaras, Electrónica, Deportes).
- `LIMIT_PER_CATEGORY`: 50 productos por categoría (250 total).
- `OUTPUT_CORPUS`: Ruta de salida → `data/processed/corpus.json`.
- `IMAGES_DIR`: Carpeta donde se guardan las fotos → `data/raw/images/`.

**Funciones:**

| Función | Qué hace | Qué llama |
|---------|----------|-----------|
| `setup_directories()` | Crea las carpetas `data/raw/images/` y `data/processed/` si no existen. | `os.makedirs()` |
| `download_image(url, path)` | Descarga una imagen desde una URL, la convierte a RGB y la guarda como `.jpg` local. | `requests.get()` → `PIL.Image.open()` → `img.save()` |
| `extract_product_data(product, category)` | Extrae título, descripción, features, URL de imagen y categoría de un producto crudo del dataset. Devuelve `None` si le falta información esencial. | — (procesamiento puro de diccionarios) |
| `main()` | Orquesta todo: carga cada categoría en streaming, extrae datos con `extract_product_data()`, descarga la imagen con `download_image()`, y acumula los resultados. Al final guarda el `corpus.json`. | `load_dataset()` (HuggingFace) → `extract_product_data()` → `download_image()` → `json.dump()` |

**Salida generada:**
- `data/processed/corpus.json` — Array JSON con 250 objetos, cada uno con: `id`, `title`, `text`, `image_url`, `local_image_path`, `category`.
- `data/raw/images/*.jpg` — 250 imágenes descargadas.

---

### `embeddings.py` — Generación de Vectores con CLIP

**Propósito:** Cargar el modelo CLIP y proporcionar métodos para convertir textos e imágenes en vectores numéricos (embeddings) de 512 dimensiones.

**Clase: `CLIPEmbedder`**

| Método | Qué hace | Qué llama internamente |
|--------|----------|----------------------|
| `__init__(model_name, pretrained)` | Carga el modelo CLIP (`ViT-B-32`) y sus transformaciones de preprocesamiento. Detecta automáticamente si hay GPU disponible. | `open_clip.create_model_and_transforms()` → `model.to(device)` → `open_clip.get_tokenizer()` |
| `embed_image(image_paths)` | Recibe una o varias rutas de imagen. Abre cada imagen con PIL, le aplica las transformaciones de CLIP (resize a 224x224, normalización), la pasa por la red neuronal y devuelve el vector normalizado. | `Image.open()` → `self.preprocess()` → `self.model.encode_image()` → normalización L2 |
| `embed_text(texts)` | Recibe uno o varios strings. Los tokeniza y los pasa por la red neuronal de texto de CLIP, devolviendo el vector normalizado. | `self.tokenizer()` → `self.model.encode_text()` → normalización L2 |

**Detalle técnico importante:** Ambos métodos normalizan los vectores resultantes dividiéndolos por su norma (`vector / ||vector||`). Esto es necesario para que la métrica de distancia coseno funcione correctamente en ChromaDB. Un vector normalizado tiene magnitud 1.0, por lo que la similitud coseno se reduce a un simple producto punto.

---

### `vector_db.py` — Base de Datos Vectorial (ChromaDB)

**Propósito:** Manejar la persistencia en disco y las consultas de búsqueda por similitud vectorial.

**Clase: `VectorDB`**

| Método | Qué hace | Qué llama internamente |
|--------|----------|----------------------|
| `__init__(db_dir, collection_name)` | Inicializa un cliente persistente de ChromaDB que guarda los datos en `data/chroma_db/`. Crea (o reabre) una colección llamada `multimodal_corpus` con métrica coseno. | `chromadb.PersistentClient()` → `client.get_or_create_collection()` |
| `index_documents(ids, embeddings, documents, metadatas)` | Inserta (o actualiza) documentos en la colección. Recibe los IDs, los vectores de CLIP, el texto descriptivo y los metadatos (categoría, ruta de imagen, URL). Los inserta en lotes de 100. | `collection.upsert()` |
| `search(query_embedding, top_k)` | Dado un vector de consulta, busca los `top_k` documentos más cercanos usando distancia coseno. Devuelve un diccionario con `ids`, `distances`, `documents` y `metadatas`. | `collection.query()` |

**Almacenamiento en disco:** ChromaDB guarda automáticamente los datos en la carpeta `data/chroma_db/` usando un índice HNSW (Hierarchical Navigable Small World), que es una estructura de datos optimizada para búsquedas aproximadas de vecinos más cercanos en espacios de alta dimensión.

---

### `index_corpus.py` — Script de Indexación

**Propósito:** Leer el `corpus.json` generado por `data_processing.py`, generar los embeddings de cada producto usando CLIP, y almacenarlos en ChromaDB. Es el puente entre la Fase 1 y la Fase 2.

**Flujo de `main()`:**

```
1. Lee data/processed/corpus.json
2. Instancia CLIPEmbedder() y VectorDB()
3. Itera en batches de 32 productos:
   a. Para cada producto del batch:
      - Extrae: id, text, metadatos, local_image_path
   b. Llama a embedder.embed_image(image_paths)
      → CLIP genera un vector de 512 floats por imagen
   c. Llama a db.index_documents(ids, embeddings, documents, metadatas)
      → ChromaDB guarda vectores + texto + metadatos en disco
4. Imprime "¡Indexación completada!"
```

**Decisión de diseño:** Se indexan los **embeddings de las imágenes** (no del texto). Esto es intencional: como CLIP proyecta texto e imágenes al mismo espacio vectorial, cuando el usuario escribe un texto, su vector de texto ya es comparable con los vectores de imagen almacenados. Esto permite búsquedas cross-modales (texto → imagen e imagen → imagen).

---

### `retrieval.py` — Recuperación de Documentos

**Propósito:** Recibir la consulta del usuario (texto o imagen), convertirla a un vector con CLIP, buscar en ChromaDB y devolver los resultados formateados.

**Clase: `MultimodalRetriever`**

| Método | Qué hace | Cadena de llamadas |
|--------|----------|--------------------|
| `__init__()` | Crea instancias de `CLIPEmbedder` y `VectorDB` para tenerlas listas. | `CLIPEmbedder()` → `VectorDB()` |
| `retrieve(query_text, top_k)` | **Búsqueda por texto.** Convierte el texto a vector, busca en ChromaDB y formatea los resultados. | `self.embedder.embed_text(query)` → `self.db.search(vector, top_k)` → `self._format_results()` |
| `retrieve_by_image(image, top_k)` | **Búsqueda por imagen.** Si recibe un `PIL.Image`, lo guarda temporalmente en disco. Luego genera el vector visual con CLIP y busca en ChromaDB. | `image.save(tmp)` → `self.embedder.embed_image(tmp_path)` → `os.unlink(tmp)` → `self.db.search(vector, top_k)` → `self._format_results()` |
| `_format_results(results)` | Método interno. Toma la respuesta cruda de ChromaDB (listas anidadas) y la convierte en una lista limpia de diccionarios con campos: `id`, `score`, `text`, `image_url`, `local_image_path`, `category`. | — (procesamiento de datos) |

**Formato de salida (ejemplo):**
```python
[
    {
        "id": "raw_meta_Musical_Instruments_B000...",
        "score": 0.8234,
        "text": "Guitarra Acústica Yamaha FG800...",
        "image_url": "https://...",
        "local_image_path": "data/raw/images/...",
        "category": "Musical_Instruments"
    },
    # ... (top_k resultados)
]
```

---

### `generation.py` — Generación de Respuestas (RAG con Gemini)

**Propósito:** Tomar los documentos recuperados por el Retriever, construir un contexto textual y enviarlo junto con la pregunta del usuario al LLM (Gemini 2.5 Flash) para generar una respuesta conversacional.

**Clase: `RAGGenerator`**

| Método | Qué hace | Cadena de llamadas |
|--------|----------|--------------------|
| `__init__(model_name, temperature)` | Inicializa el cliente de Gemini vía LangChain. Define el `PromptTemplate` con 3 variables: `context`, `question`, `query_type`. Conecta el prompt al LLM en una cadena LCEL (`self.chain = self.prompt \| self.llm`). | `ChatGoogleGenerativeAI()` → `PromptTemplate()` → operador pipe `\|` |
| `format_context(retrieved_docs)` | Toma la lista de diccionarios de evidencias y los concatena en un solo string con formato legible para el LLM. | — (concatenación de strings) |
| `generate_response(query, retrieved_docs, query_type)` | Orquesta todo: formatea el contexto, invoca la cadena LCEL con las 3 variables y devuelve el texto de la respuesta. Si falla la API, captura la excepción y devuelve un mensaje de error amigable. | `self.format_context()` → `self.chain.invoke({context, question, query_type})` → `response.content` |

**El Prompt Template (lo que Gemini realmente recibe):**
```
Eres un asistente experto de compras. Utiliza la siguiente información de productos
recuperada de nuestra base de datos para responder a la pregunta del usuario.

Reglas:
1. Responde de forma amigable y conversacional.
2. Si recomiendas un producto, menciona su título y características.
3. Si la respuesta no está en el contexto, indícalo amablemente.
4. Si el tipo de consulta es "imagen", describe los productos similares encontrados.

Tipo de Consulta: {query_type}
Contexto Recuperado: {context}
Pregunta del Usuario: {question}
```

---

### `app.py` — Interfaz Web (Streamlit)

**Propósito:** Proveer la interfaz gráfica del usuario. Maneja dos flujos de entrada (texto e imagen), muestra el historial del chat, renderiza las evidencias visuales y muestra las métricas de evaluación.

**Flujo de inicialización (se ejecuta una sola vez al arrancar):**
```
1. sys.path.append(raíz del proyecto) → soluciona imports de 'src.*'
2. load_dotenv() → carga GOOGLE_API_KEY desde .env
3. Verifica que GOOGLE_API_KEY exista → si no, st.error() y st.stop()
4. load_pipeline() [cacheada con @st.cache_resource]:
   a. Instancia MultimodalRetriever() → carga CLIP + ChromaDB
   b. Instancia RAGGenerator("gemini-2.5-flash") → conecta a la API de Google
5. Crea el sidebar con:
   a. st.file_uploader para imágenes
   b. Checkbox "Ver Dashboard de Evaluación Global" → Lee evaluation_results.json y dibuja métricas con st.line_chart()
6. Inicializa st.session_state.messages = [] (historial del chat)
```

**Flujo A — Consulta por texto (cada vez que el usuario escribe en el chat):**
```
1. Usuario escribe en st.chat_input()
2. Se guarda el mensaje en st.session_state.messages
3. retriever.retrieve(prompt, top_k=3)
   → CLIPEmbedder.embed_text(prompt) → VectorDB.search() → _format_results()
4. generator.generate_response(prompt, evidences, query_type="texto")
   → format_context() → chain.invoke() → Gemini API → response.content
5. Se renderiza la respuesta + imágenes de las evidencias con st.image()
6. Se dibuja un st.bar_chart() interactivo con los scores de confianza (Similitud) de los productos.
7. Se guarda la respuesta en el historial
```

**Flujo B — Consulta por imagen (cuando el usuario sube una foto en el sidebar):**
```
1. Usuario arrastra una imagen en st.file_uploader()
2. Se muestra preview con st.image() en el sidebar
3. Usuario presiona botón "🔍 Buscar productos similares"
4. retriever.retrieve_by_image(uploaded_image, top_k=3)
   → Guarda PIL.Image como archivo temporal (.png)
   → CLIPEmbedder.embed_image(tmp_path) → VectorDB.search() → _format_results()
   → Elimina el archivo temporal
5. generator.generate_response("...subió una imagen...", evidences, query_type="imagen")
   → format_context() → chain.invoke() → Gemini API → response.content
6. Se renderiza la respuesta + imágenes similares encontradas
7. Se dibuja un st.bar_chart() interactivo con los scores de confianza.
8. Se guarda la respuesta en el historial
```

---

## Tecnologías y Versiones

---

## Módulo de Evaluación

### `evaluation_metrics.py` — Métricas de Recuperación de Información

**Propósito:** Implementar las métricas estándar de IR para medir la calidad de los resultados del sistema de recuperación.

**Funciones:**

| Función | Qué mide | Fórmula simplificada |
|---------|----------|---------------------|
| `precision_at_k(retrieved, relevant, k)` | Proporción de documentos relevantes entre los Top-K recuperados. | `|relevantes en Top-k| / k` |
| `recall_at_k(retrieved, relevant, k)` | Proporción de documentos relevantes totales que fueron encontrados en el Top-K. | `|relevantes en Top-k| / |total relevantes|` |
| `dcg_at_k(retrieved, relevant, k)` | Ganancia acumulativa descontada: penaliza documentos relevantes que aparecen abajo en el ranking. | `Σ rel_i / log2(i+1)` |
| `ndcg_at_k(retrieved, relevant, k)` | DCG normalizado por el DCG ideal (si todos los relevantes estuvieran arriba). Escala de 0 a 1. | `DCG@k / IDCG@k` |

---

### `evaluate.py` — Script de Evaluación Completa

**Propósito:** Orquestar la evaluación del sistema de extremo a extremo.

**Flujo de `run_evaluation()`:**

```
1. Carga data/evaluation/qrels.json (20 queries con sus documentos relevantes)
2. Instancia MultimodalRetriever() (CLIP + ChromaDB)
3. Para cada query con documentos relevantes anotados:
   a. Ejecuta retriever.retrieve(query_text, top_k=10)
   b. Extrae los IDs de los documentos recuperados
   c. Calcula Precision@k, Recall@k y NDCG@k para k = [1, 3, 5, 10]
   d. Acumula las métricas para calcular promedios
4. Imprime un reporte tabular en consola
5. Guarda los resultados completos en data/evaluation/evaluation_results.json
```

**Archivo de Qrels (`data/evaluation/qrels.json`):**
Contiene 20 consultas manualmente anotadas que cubren las categorías del corpus: Musical Instruments, Pet Supplies y Video Games. Cada query especifica los IDs de los documentos que se consideran relevantes (ground truth).

---

## Tecnologías y Versiones

| Tecnología | Uso en el proyecto | Versión |
|------------|-------------------|---------|
| Python | Lenguaje principal | 3.8+ |
| Streamlit | Interfaz web conversacional | latest |
| open_clip | Modelo CLIP (ViT-B-32) para embeddings multimodales | latest |
| ChromaDB | Base de datos vectorial con persistencia en disco | latest |
| LangChain | Orquestación del pipeline RAG (LCEL: Prompt → LLM) | latest |
| langchain-google-genai | Conector de LangChain con la API de Gemini | latest |
| Gemini 2.5 Flash | LLM para generación de respuestas conversacionales | 2.5 |
| Pillow (PIL) | Manipulación de imágenes (abrir, convertir, guardar) | latest |
| HuggingFace Datasets | Carga del dataset Amazon Reviews 2023 en streaming | latest |
| NumPy | Cálculos numéricos y métricas de evaluación | latest |

