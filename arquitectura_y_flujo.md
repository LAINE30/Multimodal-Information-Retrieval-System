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
*   **Cómo funciona en el flujo:** Cuando el Retriever recibe una consulta (ej. la voz transcrita), la convierte a vector y ChromaDB debe calcular su **Score de Relevancia (RSV)** contra los miles de productos indexados. Para hacer esto extremadamente rápido, en el módulo `embeddings.py` aplicamos una normalización $L2$ (Euclidiana) a los vectores de CLIP inmediatamente después de generarlos. Matematicamente, si dos vectores están normalizados a magnitud $1.0$, el cálculo de la Similitud Coseno se simplifica a un simple **Producto Punto (Dot Product)**. Esto acelera el ranking a nivel de CPU, devolviendo los $K$ vecinos más cercanos al instante.

### 4. Re-ranking Semántico (Cross-Encoder) — *Pipeline de Dos Etapas*

*   **Algoritmo / Modelo:** Cross-Encoder `ms-marco-MiniLM-L-6-v2` (Hugging Face, ~22MB).
*   **Por qué no basta con CLIP:** CLIP es un **Bi-Encoder**: genera vectores de texto e imagen de forma *independiente* y luego los compara. Esto lo hace extremadamente rápido, pero pierde matices de relación sintáctica entre la consulta y el documento. Un **Cross-Encoder** analiza el par `(consulta, documento)` *en conjunto* usando **atención cruzada completa** (mecanismo de Transformers), lo que le permite capturar relaciones semánticas mucho más finas. El precio es la velocidad: no escala a millones de documentos.
*   **Solución — Pipeline de Dos Etapas (Recall + Precision):**

| Etapa | Modelo | Candidatos entrada | Candidatos salida | Objetivo |
|---|---|---|---|---|
| **Stage 1** | CLIP (Bi-Encoder) | Corpus completo | Top-20 candidatos | Máximo Recall rápido |
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
*   **Fórmula:** `score_ajustado = score_original / Boost (si score < 0) o score_original × Boost (si score > 0)` donde `Boost = 1.0 + α × (likes - dislikes) / (total + 1)` y `α = 0.3`.
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
- `CATEGORIES`: Lista de categorías del dataset (20 en total: Instrumentos Musicales, Videojuegos, Mascotas, Cámaras, Electrónica, etc.).
- `LIMIT_PER_CATEGORY`: 75 productos por categoría (aproximadamente 1,500 productos en total).
- `OUTPUT_CORPUS`: Ruta de salida → `data/processed/corpus.json`.
- `IMAGES_DIR`: Carpeta donde se guardan las fotos → `data/raw/images/`.

### `embeddings.py` — Generación de Vectores con CLIP

**Propósito:** Cargar el modelo CLIP y proporcionar métodos para convertir textos e imágenes en vectores numéricos (embeddings) de 512 dimensiones. Usa `ViT-B-32` y normaliza los vectores con norma L2 para búsquedas de producto punto extremadamente rápidas.

### `vector_db.py` — Base de Datos Vectorial (ChromaDB)

**Propósito:** Manejar la persistencia en disco y las consultas de búsqueda por similitud vectorial. Utiliza `chromadb.PersistentClient()` para guardar los datos en `data/chroma_db/` usando un índice HNSW optimizado.

### `index_corpus.py` — Script de Indexación

**Propósito:** Leer el `corpus.json` generado, generar los embeddings multimodales (basados en imágenes) de cada producto usando CLIP, y almacenarlos en ChromaDB junto con sus metadatos textuales.

### `retrieval.py` — Recuperación de Documentos

**Propósito:** Orquestar la búsqueda. Transforma la consulta (texto o imagen subida por el usuario) en vector usando `CLIPEmbedder`, interroga a la base de datos `VectorDB`, aplica `query_expansion` si se requiere y luego pasa los candidatos iniciales por el Cross-Encoder (Stage 2) para el **re-ranking semántico**.

### `generation.py` — Generación de Respuestas (RAG con Gemini)

**Propósito:** Conectar con la API de Google Gemini (v2.5 Flash) vía LangChain. Inyecta el contexto recuperado, el historial de chat y la pregunta en un `PromptTemplate` para generar una respuesta en lenguaje natural.

### `app.py` — Interfaz Web (Streamlit)

**Propósito:** Proveer la interfaz gráfica del usuario. Maneja entrada por voz, texto e imagen; mantiene la sesión; dibuja gráficas de similitud vectorial y gestiona el Relevance Feedback interactivo.

---

## Módulo de Evaluación

### `evaluation_metrics.py` — Métricas de Recuperación de Información

**Propósito:** Implementar las métricas estándar de IR para medir la calidad de los resultados del sistema de recuperación.

**Funciones:**
| Función | Qué mide | Fórmula simplificada |
|---------|----------|---------------------|
| `precision_at_k` | Proporción de documentos relevantes entre los Top-K recuperados. | `\|relevantes en Top-k\| / k` |
| `recall_at_k` | Proporción de documentos relevantes totales encontrados en el Top-K. | `\|relevantes en Top-k\| / \|total relevantes\|` |
| `dcg_at_k` | Ganancia acumulativa descontada (penaliza documentos abajo en el ranking). | `Σ rel_i / log2(i+1)` |
| `ndcg_at_k` | DCG normalizado por el DCG ideal. Escala de 0 a 1. | `DCG@k / IDCG@k` |

### `evaluate.py` — Script de Evaluación Completa

**Propósito:** Orquestar la evaluación del sistema de extremo a extremo.

**Flujo:**
1. Carga `data/evaluation/qrels.json` (queries manuales con sus documentos relevantes, ground truth).
2. Instancia `MultimodalRetriever()`.
3. Para cada query, ejecuta una búsqueda y extrae los IDs.
4. Calcula Precision@k, Recall@k y NDCG@k para k = [1, 3, 5, 10].
5. Imprime el reporte y guarda en `data/evaluation/evaluation_results.json`.

---

## Tecnologías y Versiones

| Tecnología | Uso en el proyecto | Versión |
|------------|-------------------|---------|
| Python | Lenguaje principal | 3.8+ |
| Streamlit | Interfaz web conversacional | latest |
| open_clip | Modelo CLIP (ViT-B-32) para embeddings multimodales | latest |
| ChromaDB | Base de datos vectorial con persistencia en disco | latest |
| LangChain | Orquestación del pipeline RAG | latest |
| langchain-google-genai | Conector de LangChain con API Gemini | latest |
| Gemini 2.5 Flash | LLM para generación de respuestas | 2.5 |
| Pillow (PIL) | Manipulación de imágenes | latest |
| HuggingFace Datasets | Carga del dataset Amazon Reviews 2023 | latest |
| NumPy | Cálculos numéricos y métricas de evaluación | latest |

---

## Funcionalidades de Excelencia (Implementadas)

Este proyecto implementa cuatro funcionalidades avanzadas que mejoran drásticamente la experiencia de búsqueda:

1. **Re-ranking (Cross-Encoder):** Refina la búsqueda rápida de CLIP (Bi-Encoder) pasando el top-20 a un modelo de atención cruzada más denso (`ms-marco`), maximizando la Precision semántica final antes de pasársela al LLM.
2. **Query Expansion (Multi-query Retrieval):** Usa Gemini para generar N-variantes de la consulta de usuario y consulta todas en la base de datos, resolviendo la barrera del "vocabulario cerrado" y maximizando el Recall.
3. **Relevance Feedback:** Emplea un algoritmo tipo Rocchio (basado en score) guardando los "Likes" y "Dislikes" del usuario en un JSON persistente para castigar o premiar resultados en futuras consultas idénticas.
4. **Memoria Conversacional:** Inyecta automáticamente los últimos $N$ turnos del chat en el prompt final del LLM para resolver anáforas como "quiero ese en negro" basándose en el historial.
