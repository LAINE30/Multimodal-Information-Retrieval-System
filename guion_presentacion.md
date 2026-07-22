# 🎤 Guión de Presentación
## Sistema de Recuperación de Información Multimodal con RAG

> **Duración estimada:** 12-15 minutos  
> **Formato:** Presentación técnica con demostración en vivo  
> **Audiencia:** Profesores / evaluadores de proyecto

---

## 🟢 INTRODUCCIÓN (1-2 min)

**[Diapositiva: Portada del proyecto]**

> *Buenas [días/tardes]. Mi nombre es [tu nombre] y hoy les voy a presentar un sistema de Recuperación de Información Multimodal basado en técnicas de Inteligencia Artificial de última generación.*

> *¿Alguna vez han buscado un producto en Amazon escribiendo algo como "mochila para acampar" y el resultado fue perfecto aunque la descripción del producto diga "backpack for camping"? Eso es exactamente lo que este sistema hace, pero construido desde cero, con código propio y arquitectura de IA explicable.*

> *El objetivo del proyecto fue construir un asistente de compras inteligente capaz de entender preguntas en lenguaje natural, imágenes de referencia, y hasta comandos de voz, devolviendo los productos más relevantes de un catálogo de más de 2,100 artículos de Amazon distribuidos en 20 categorías.*

---

## 📐 ARQUITECTURA GENERAL (2-3 min)

**[Diapositiva o diagrama: Mapa de módulos]**

> *El sistema se compone de dos grandes fases:*

> *La **Fase de Indexación** ocurre una sola vez: se descarga el corpus de Amazon, se extraen las imágenes de cada producto, y se generan representaciones matemáticas, que llamamos **embeddings**, usando un modelo de IA llamado **CLIP**. Esos vectores se almacenan en una base de datos especializada llamada **ChromaDB**.*

> *La **Fase de Inferencia** ocurre cada vez que el usuario hace una pregunta. La consulta entra por la interfaz web en Streamlit, pasa por un pipeline de cuatro etapas de refinamiento progresivo, y el resultado final se le da a un LLM (Gemini 2.5 Flash de Google) para que construya una respuesta en lenguaje natural.*

> *Todo el proyecto está desplegado en **Google Cloud Run**, accesible públicamente desde cualquier navegador, sin instalar nada.*

---

## 🧠 CONCEPTOS TÉCNICOS CLAVE (3-4 min)

**[Diapositiva: Diagrama del pipeline de búsqueda]**

> *Permítanme explicar brevemente los 4 pilares técnicos del sistema:*

### 1. CLIP — Búsqueda Semántica Cross-Modal
> *La base del sistema es el modelo CLIP de OpenAI. A diferencia de búsquedas tradicionales como TF-IDF o BM25, que comparan palabras exactas, CLIP proyecta texto e imágenes en el **mismo espacio matemático de 512 dimensiones**. Esto significa que si busco "teléfono con cámara potente", el sistema encuentra un producto cuya descripción dice "smartphone with high-resolution lens", porque en ese espacio vectorial ambos significan lo mismo.*

### 2. Re-ranking con Cross-Encoder — Pipeline de Dos Etapas
> *CLIP recupera rápidamente los 20 candidatos más parecidos del catálogo (maximizando Recall). Luego entra en juego un segundo modelo: el Cross-Encoder "ms-marco-MiniLM-L6". Este modelo analiza la consulta y cada candidato simultáneamente con atención cruzada completa, lo que le permite detectar matices semánticos mucho más finos. El resultado son los Top 3 con la mayor precisión posible.*

### 3. Query Expansion — Superando el Vocabulario Cerrado
> *Un usuario puede buscar "guitarra para principiantes" pero el corpus tiene "beginner acoustic guitar". Para resolver esta brecha, antes de buscar, el sistema le pide a Gemini que genere 3 variantes semánticas de la consulta. Luego busca con las 4 versiones en paralelo y combina los resultados, maximizando el Recall del sistema.*

### 4. Relevance Feedback — Aprendizaje del Usuario
> *Cada producto en los resultados tiene un botón de 👍 o 👎. Cuando el usuario vota, el sistema guarda ese historial. En búsquedas posteriores similares, aplica una variante del **Algoritmo de Rocchio**: calcula un factor de boost para ese producto y ajusta su score hacia arriba (con Like) o hacia abajo (con Dislike), personalizando el ranking de forma progresiva.*

---

## 🖥️ DEMOSTRACIÓN EN VIVO (4-5 min)

**[Abrir la URL pública de Cloud Run en el navegador]**

> *Pasemos ahora a la demostración en vivo. Esta es la aplicación corriendo en Google Cloud.*

### Demo 1: Búsqueda por texto natural
> *Voy a escribir en el chat: "busco unos audífonos inalámbricos para hacer ejercicio".*  
> *[Esperar resultado]*  
> *Como pueden ver, el sistema encontró productos de la categoría de Electrónica y Celulares. Noten que en la parte inferior aparece el score de similitud de cada producto. El asistente también genera una respuesta en lenguaje natural describiendo las opciones.*

> *Ahora hago clic en "Ver consultas expandidas" — aquí pueden ver las 3 variantes que Gemini generó: "wireless earbuds for sports", "Bluetooth headphones workout"... por eso el Recall fue tan alto.*

### Demo 2: Búsqueda por imagen
> *Ahora voy al menú lateral, subo una foto de una bicicleta [usar foto de celular o descargada].*  
> *[Clic en "Buscar productos similares"]*  
> *El modelo CLIP reconoció la imagen y encontró estos productos visualmente similares, aunque no escribí ni una sola palabra.*

### Demo 3: Feedback y aprendizaje
> *Voy a darle "Like" a este primer resultado.*  
> *Ahora repito la misma búsqueda: "busco unos audífonos..."*  
> *[Esperar resultado]*  
> *Pueden notar que el producto al que le di Like aparece ahora más arriba. El sistema aprendió mi preferencia.*

### Demo 4: Memoria conversacional
> *[En el mismo chat]* "¿Tiene alguno en color blanco?"  
> *El sistema entiende que estoy hablando del mismo producto del turno anterior, gracias a la memoria de conversación inyectada en el prompt.*

---

## 📊 EVALUACIÓN DEL SISTEMA (1-2 min)

**[Diapositiva: Tabla de métricas]**

> *El sistema fue evaluado usando métricas estándar de Recuperación de Información: Precision@k, Recall@k y NDCG@k, sobre un conjunto de 20 consultas manualmente anotadas.*

> *NDCG es especialmente importante porque no solo mide si el sistema encontró el producto correcto, sino **en qué posición del ranking lo puso**. Un NDCG cercano a 1.0 significa que los productos relevantes aparecen siempre primero.*

> *El módulo de evaluación está disponible en el panel lateral de la aplicación, y los resultados quedan guardados para seguimiento histórico.*

---

## ☁️ DESPLIEGUE EN LA NUBE (1 min)

**[Diapositiva: Arquitectura Cloud]**

> *El proyecto está completamente desplegado en **Google Cloud Run** con despliegue continuo (CI/CD). Cada vez que hago un cambio en el código y lo subo a GitHub, Google Cloud Build lo detecta automáticamente, construye una nueva imagen Docker del sistema y la publica en producción. Todo sin intervención manual.*

> *La base de datos vectorial se empaqueta dentro de la imagen Docker para evitar tiempos de inicio largos en la nube.*

---

## 🏁 CONCLUSIÓN (1 min)

**[Diapositiva: Resumen]**

> *Para resumir: este proyecto implementó un sistema completo de Recuperación de Información Multimodal que va mucho más allá de una búsqueda simple. Combina cuatro técnicas de Inteligencia Artificial — CLIP, Cross-Encoder, Query Expansion y Relevance Feedback — sobre un corpus de 2,100 productos en 20 categorías, con una interfaz conversacional que acepta texto, voz e imágenes, y todo desplegado en producción en la nube.*

> *El código fuente está disponible en GitHub con documentación técnica completa en `arquitectura_y_flujo.md`.*

> *Muchas gracias. ¿Alguna pregunta?*

---

## ❓ POSIBLES PREGUNTAS Y RESPUESTAS

**P: ¿Por qué usaron ChromaDB y no FAISS?**
> ChromaDB nos da persistencia de metadatos integrada: guarda los vectores, los textos y las URLs de las imágenes en un solo lugar. Con FAISS tendríamos que mantener una base de datos SQLite separada y unir ambas por ID manualmente.

**P: ¿Cuál es la diferencia entre el Bi-Encoder y el Cross-Encoder?**
> El Bi-Encoder (CLIP) procesa el texto y la imagen por separado y luego compara los vectores. Es rapidísimo pero pierde relaciones contextuales. El Cross-Encoder analiza el par completo (consulta + documento) de forma conjunta con atención cruzada, capturando matices mucho más finos. Por eso lo usamos solo en la segunda etapa, sobre un subconjunto pequeño de candidatos.

**P: ¿Cómo escala el sistema a un catálogo de un millón de productos?**
> ChromaDB usa un índice HNSW (grafo jerárquico) cuya complejidad de búsqueda es O(log n), no lineal. El Cross-Encoder solo se aplica al Top-20 de candidatos pre-filtrados por CLIP, por lo que su costo computacional es constante independientemente del tamaño del corpus.

**P: ¿Qué pasa si la API de Gemini no está disponible?**
> Implementamos manejo de excepciones en el módulo de generación. Si falla la API, el sistema devuelve un mensaje de error amigable al usuario y sigue mostrando los resultados de búsqueda visual, que no dependen de Gemini.

**P: ¿Por qué las imágenes se cargan desde Amazon y no están en el servidor?**
> Almacenar 2,100 imágenes en el servidor incrementaría el tamaño de la imagen Docker en varios GB, ralentizando el despliegue y aumentando el costo. La solución implementada es guardar el URL original de Amazon en la base de datos vectorial. El navegador del usuario descarga la imagen directamente desde Amazon, lo que también reduce la carga en nuestro servidor.
