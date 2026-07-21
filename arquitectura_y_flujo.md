# Arquitectura y Flujo del Sistema (RAG Multimodal)

Este documento explica en detalle cómo funciona el proyecto internamente: qué hace cada archivo `.py`, qué clases y funciones contiene, a quién llama, y cómo se conectan entre sí desde que se descargan los datos hasta que el usuario obtiene una respuesta en la interfaz de chat.

---

## Conceptos Clave

Antes de entender el flujo, es vital conocer los conceptos fundamentales que le dan vida al sistema:

1. **Embedding (Vector):** Es una lista de números (ej. 512 floats) que representa el "significado" de un texto o imagen en un espacio matemático.
2. **CLIP (Contrastive Language-Image Pretraining):** Modelo de IA de OpenAI que convierte texto e imágenes al mismo espacio vectorial.
3. **ChromaDB:** Base de datos que guarda vectores y busca por similitud semántica.
4. **RAG (Retrieval-Augmented Generation):** Técnica donde se recupera información y se "inyecta" en el prompt de un LLM para basar sus respuestas en datos reales.
5. **LCEL (LangChain Expression Language):** Framework para construir cadenas de procesamiento.

---

## Diseño del Sistema de Recuperación de Información (RI)

El proyecto implementa un pipeline moderno de Recuperación de Información adaptado a espacios latentes (Vectores). A continuación se explican las decisiones de diseño y los algoritmos matemáticos que intervienen internamente en cada parte del flujo:

### 1. Representación de Documentos y Consultas (Embeddings)
*   **Algoritmo / Enfoque:** Recuperación Semántica Densificada en Espacios Vectoriales.
*   **Por qué usamos CLIP:** En la Recuperación de Información tradicional (como los modelos TF-IDF o BM25 usados en Elasticsearch), las búsquedas son puramente léxicas (coincidencia de palabras exactas). Esto falla si buscas "teléfono móvil" y el documento dice "smartphone". Nosotros usamos embeddings densos generados por CLIP (ViT-B-32). CLIP fue elegido específicamente porque proyecta el significado del texto y las características de la imagen en un **mismo espacio matemático de 512 dimensiones**, permitiendo búsquedas cross-modales (buscar una imagen describiéndola con texto, o buscar productos usando otra foto de referencia) de forma nativa sin necesitar OCR ni etiquetado manual.

### 2. Indexación y Búsqueda Vectorial
*   **Algoritmo / Estructura de Datos:** Búsqueda de Vecinos Más Cercanos Aproximados (ANN) utilizando grafos **HNSW** (Hierarchical Navigable Small World).
*   **Por qué ChromaDB en lugar de FAISS:** 
    *   Aunque el requerimiento inicial permitía FAISS o ChromaDB, se optó por **ChromaDB**. FAISS (creado por Facebook) es una librería ultrarrápida para operaciones en memoria, pero **carece de persistencia robusta de metadatos**. Si usáramos FAISS, tendríamos que manejar los vectores en RAM y mantener una base de datos SQLite separada para guardar los títulos, textos e imágenes, uniéndolos manualmente por ID.
    *   **ChromaDB** resuelve esto ya que actúa como una base de datos completa. Internamente guarda los vectores organizados en un grafo HNSW (para búsquedas en tiempo logarítmico $\mathcal{O}(\log n)$ en vez de barridos lineales lentos) y paralelamente utiliza SQLite/Parquet para almacenar los metadatos (categoría, texto, URLs). Además, permite añadir persistencia a disco (`PersistentClient`) en una sola línea de código, evitando que el índice deba reconstruirse cada vez que se reinicia la aplicación en Streamlit.

### 3. Función de Similitud (Ranking Top-K)
*   **Algoritmo / Concepto:** Similitud Coseno y Producto Punto.
*   **Cómo funciona en el flujo:** Cuando el Retriever recibe una consulta (ej. la voz transcrita), la convierte a vector y ChromaDB debe calcular su **Score de Relevancia (RSV)** contra los 700 productos indexados. Para hacer esto extremadamente rápido, en el módulo `embeddings.py` aplicamos una normalización $L2$ (Euclidiana) a los vectores de CLIP inmediatamente después de generarlos. Matematicamente, si dos vectores están normalizados a magnitud $1.0$, el cálculo de la Similitud Coseno se simplifica a un simple **Producto Punto (Dot Product)**. Esto acelera el ranking a nivel de CPU, devolviendo los $K$ vecinos más cercanos al instante.

### 4. Re-ranking Semántico (Cross-Encoder) — *Pipeline de Dos Etapas*

*   **Algoritmo / Modelo:** Cross-Encoder `ms-marco-MiniLM-L-6-v2` (Hugging Face, ~22MB).
*   **Por qué no basta con CLIP:** CLIP es un **Bi-Encoder**: genera vectores de texto e imagen de forma *independiente* y luego los compara. Esto lo hace extremadamente rápido, pero pierde matices de relación sintáctica entre la consulta y el documento. Un **Cross-Encoder** analiza el par `(consulta, documento)` *en conjunto* usando **atención cruzada completa** (mecanismo de Transformers), lo que le permite capturar relaciones semánticas mucho más finas. El precio es la velocidad: no escala a millones de documentos.
*   **Solución — Pipeline de Dos Etapas (Recall + Precision):**

| Etapa | Modelo | Candidatos entrada | Candidatos salida | Objetivo |
|---|---|---|---|---|
| **Stage 1** | CLIP (Bi-Encoder) | 700 docs (corpus completo) | Top-20 candidatos | Máximo Recall rápido |
| **Stage 2** | Cross-Encoder (ms-marco) | Top-20 candidatos | Top-3 finales | Máxima Precision semántica |

*   **Implementación:** El método `retrieve_and_rerank()` en `retrieval.py` orquesta ambas etapas. El Cross-Encoder produce un score logit real (puede ser negativo). Los candidatos se re-ordenan de mayor a menor score. Solo se aplica a búsquedas de **texto y voz** (no a imagen, ya que el Cross-Encoder trabaja exclusivamente con pares texto-texto).

### 5. Query Expansion (Reformulación Automática de Consultas)

*   **Algoritmo / Concepto:** Expansión de consultas basada en LLM (Pseudo-Relevance Expansion).
*   **Por qué es necesaria:** Un usuario puede buscar "guitarra para principiantes" pero el corpus contiene "acoustic guitar beginner" o "instrumento de cuerdas para novatos". CLIP captura similitud semántica, pero tiene límites. Expandir la consulta a múltiples variantes lingüísticas amplía la cobertura del espacio vectorial y mejora el **Recall**.
*   **Implementación:** El módulo `src/query_expansion.py` usa Gemini para generar N reformulaciones. El método `retrieve_with_expansion()` en `retrieval.py` ejecuta un pipeline de 3 etapas:

| Etapa | Modelo | Entrada | Salida | Objetivo |
|---|---|---|---|---|
| **Stage 0** | Gemini (LLM) | 1 query original | 4 variantes (orig + 3 expansiones) | Diversidad lingüística |
| **Stage 1** | CLIP × 4 | 4 variantes | Pool unificado ~60 candidatos únicos | Máximo Recall |
| **Stage 2** | Cross-Encoder | Pool completo | Top-3 finales | Máxima Precision |

### 6. Relevance Feedback (Retroalimentación del Usuario)

*   **Algoritmo / Concepto:** Algoritmo de Rocchio simplificado (Score-based).
*   **En la teoría clásica de RI:** El algoritmo de Rocchio modifica el **vector de la consulta** acercándolo a los centros de los documentos relevantes y alejándolo de los no-relevantes. Nuestra implementación es una variante score-based: en vez de modificar vectores, ajustamos los **scores finales** del Cross-Encoder con un factor multiplicativo basado en el historial de feedback.
*   **Fórmula:** `score_ajustado = score_original × (1.0 + α × (likes - dislikes) / (total + 1))` donde `α = 0.3`.
*   **Implementación:** El módulo `src/relevance_feedback.py` persiste los votos 👍/👎 en `data/evaluation/relevance_feedback.json`. Cada vez que se realiza una búsqueda, `apply_feedback_to_results()` aplica el boost/penalización a los candidatos antes de presentarlos.

### 7. Memoria Conversacional (Context-Aware Generation)

*   **Concepto:** Inyección de historial conversacional en el prompt del LLM.
*   **Por qué es necesaria:** Sin memoria, cada pregunta del usuario se trata de forma aislada. Si el usuario pregunta "¿tienes guitarras?" y luego dice "¿y alguna más barata?", el sistema sin memoria no entendería a qué se refiere. Con memoria conversacional, el LLM recibe los últimos turnos del chat como contexto adicional.
*   **Implementación:** El método estático `RAGGenerator.format_chat_history()` en `src/generation.py` toma los últimos 6 mensajes de `st.session_state.messages`, los formatea como pares `Usuario:/Asistente:` (truncando respuestas largas a 300 caracteres) y los inyecta en la variable `{chat_history}` del prompt template. El LLM usa esta ventana deslizante para resolver **referencias anafóricas** ("ese producto", "el primero", "cuéntame más") y mantener coherencia conversacional.

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
- `CATEGORIES`: Lista de 10 categorías del dataset (Instrumentos Musicales, Videojuegos, Mascotas, Cámaras, Electrónica, Deportes, Juguetes, Hogar/Cocina, Celulares, Productos de Oficina).
- `LIMIT_PER_CATEGORY`: 50 productos por categoría o más (aproximadamente 700 productos en total que pasaron los filtros de calidad).
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
- `data/processed/corpus.json` — Array JSON con ~700 objetos, cada uno con: `id`, `title`, `text`, `image_url`, `local_image_path`, `category`.
- `data/raw/images/*.jpg` — ~700 imágenes descargadas localmente.

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

---

## Funcionalidades de Excelencia

Este proyecto implementa cuatro funcionalidades avanzadas ("de excelencia") que otorgan un total de 60 puntos adicionales, mejorando significativamente la calidad de la recuperación y la experiencia del usuario mediante técnicas de IA en estado del arte.

### 1. Re-ranking (+15)
Se implementó un modelo de re-ranking (Cross-Encoder `ms-marco-MiniLM-L-6-v2`) que refina el ranking inicial obtenido mediante la búsqueda vectorial pura antes de generar la respuesta del sistema RAG.
- **Diferencia Bi-Encoder vs Cross-Encoder:** El modelo base (CLIP) es un *Bi-Encoder*. Calcula embeddings por separado para el texto y la imagen, lo que es rapidísimo (permite indexar miles de productos) pero sacrifica precisión porque no procesa la interacción semántica cruzada. El modelo de Re-ranking es un *Cross-Encoder*, el cual procesa la consulta del usuario y el texto del documento **simultáneamente** a través de capas de auto-atención (self-attention). Esto es computacionalmente costoso, por lo que solo se aplica a un subconjunto pequeño.
- **Flujo en el Pipeline (`MultimodalRetriever`):** 
  1. Durante una búsqueda de texto, CLIP primero recupera un conjunto grande de candidatos (ej. `candidate_k=15`). Esto maximiza el *Recall* (evita que el producto correcto se escape).
  2. El Cross-Encoder toma la consulta y cada uno de los 15 candidatos. 
  3. Ejecuta su método `predict()` para asignar un nuevo puntaje de relevancia semántica profunda (Re-rank Score).
  4. La lista se ordena descendentemente basada en este nuevo puntaje, y se realiza un *slicing* para retornar solo el `top_k=6` a la etapa de generación. Esto dispara dramáticamente la métrica *Precision@k*.

### 2. Query Expansion (+15)
Se implementó un mecanismo automático de expansión (Multi-query Retrieval) para resolver el problema del "vocabulario cerrado" o desajuste de términos entre lo que dice el usuario y lo que dicen las descripciones de los productos (ej. jerga técnica vs lenguaje coloquial).
- **Implementación (`query_expansion.py`):** Antes de interactuar con la base de datos, la consulta cruda pasa por un `QueryExpander`. Este módulo invoca al LLM (Gemini 1.5 Flash) usando técnicas de *Prompt Engineering*. El LLM asume el rol de un experto en comercio electrónico y recibe instrucciones estrictas para retornar `n_expansions=3` frases equivalentes (sinónimos o variantes de búsqueda) formateadas obligatoriamente como un arreglo JSON.
- **Flujo Vectorial Paralelo:** Tanto la consulta original como sus 3 expansiones son convertidas en embeddings a través de CLIP. El sistema consulta a ChromaDB utilizando todos estos vectores. Finalmente, los conjuntos de resultados se fusionan (pooling) y se deduplican utilizando los IDs únicos de los productos. Esto genera una piscina de candidatos (candidate pool) mucho más rica antes del Re-ranking.

### 3. Relevance Feedback (+15)
El sistema permite que el usuario califique las respuestas en la interfaz gráfica (botones 👍/👎), y utiliza este feedback para alterar dinámicamente los pesos matemáticos en búsquedas posteriores.
- **Algoritmo (Variante de Rocchio):** A diferencia de Rocchio puro (que mueve el vector de la consulta en el espacio latente sumando/restando vectores de documentos), esta implementación actúa directamente sobre el puntaje final del documento por motivos de eficiencia en bases de datos pre-indexadas.
- **Mecanismo de Persistencia:** Los votos (likes y dislikes por ID de producto) se almacenan en disco (`data/evaluation/relevance_feedback.json`) a través de la clase `RelevanceFeedbackStore`, asegurando que la memoria del sistema persista entre reinicios.
- **Matemáticas del Boost:** Al recuperar los documentos en una nueva consulta, el sistema revisa el JSON de votos. Para cada producto evaluado, calcula: `Boost = 1.0 + \alpha * ((likes - dislikes) / (total_votos + 1))`. (Con `\alpha = 0.3`).
- El puntaje del Cross-Encoder o CLIP se multiplica por este `Boost`. Los productos con historial positivo reciben un aumento (empujándolos al Top-1), mientras que el ruido irrelevante recibe un castigo algorítmico, descendiendo en el ranking final.

### 4. Memoria Conversacional (+15)
Se implementó una memoria de corto plazo que permite al sistema RAG inyectar el contexto de interacciones pasadas para resolver referencias cruzadas (co-reference resolution), permitiendo un flujo de diálogo humano.
- **Gestión de Estado (State Management):** La interfaz web (Streamlit) actúa como gestor del estado de la sesión, almacenando las parejas de pregunta-respuesta (User-Assistant) en `st.session_state.messages`.
- **Inyección RAG (`generation.py`):** Cuando se activa el flujo RAG, la función estática `format_chat_history()` extrae y empaqueta los últimos $N$ turnos de la conversación en un gran string de transcripción (ej. `Usuario: X \n IA: Y`).
- **Resolución de Ambigüedades:** Esta transcripción se inyecta dinámicamente en el bloque `{chat_history}` de la plantilla de LangChain (`PromptTemplate`). Cuando el usuario hace una pregunta ambigua como *"¿Y tienes ese en color negro?"*, el LLM puede leer el historial inyectado, entender que *"ese"* se refiere al modelo exacto discutido en el turno anterior, y formular la respuesta o extracción correcta.
