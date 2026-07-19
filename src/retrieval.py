"""
Módulo para la recuperación de documentos más relevantes (Retrieval).
Implementa un pipeline de dos etapas:
  - Stage 1: Bi-Encoder (CLIP) → recuperación rápida de candidatos por similitud vectorial.
  - Stage 2: Cross-Encoder (ms-marco-MiniLM-L-6-v2) → re-ranking semántico profundo.
"""
from src.embeddings import CLIPEmbedder
from src.vector_db import VectorDB
from PIL import Image
import numpy as np
import tempfile
import os

class MultimodalRetriever:
    """
    Clase que encapsula la lógica para convertir un query (texto o imagen)
    y buscarlo en la base de datos vectorial usando CLIP.
    Opcionalmente aplica un Cross-Encoder para re-ordenar los candidatos
    con mayor precisión semántica.
    """
    def __init__(self):
        # Inicializa bajo demanda para no saturar la memoria si solo importamos la clase
        self.embedder = CLIPEmbedder()
        self.db = VectorDB()
        # El Cross-Encoder y el QueryExpander se cargan de forma lazy
        self._cross_encoder = None
        self._query_expander = None

    def _get_cross_encoder(self):
        """Carga el Cross-Encoder de forma lazy la primera vez que se necesita."""
        if self._cross_encoder is None:
            from sentence_transformers.cross_encoder import CrossEncoder
            print("Cargando Cross-Encoder (ms-marco-MiniLM-L-6-v2)...")
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("Cross-Encoder cargado.")
        return self._cross_encoder

    def _get_query_expander(self):
        """Carga el QueryExpander de forma lazy la primera vez que se necesita."""
        if self._query_expander is None:
            from src.query_expansion import QueryExpander
            self._query_expander = QueryExpander()
        return self._query_expander

    def _format_results(self, results, reranked: bool = False):
        """
        Formatea los resultados crudos de ChromaDB en una lista de diccionarios legibles.
        Método interno compartido por retrieve y retrieve_by_image.
        """
        formatted_results = []
        
        # ChromaDB retorna listas anidadas, iteramos sobre el primer batch (query único)
        if not results["ids"] or not results["ids"][0]:
            return formatted_results
            
        ids = results["ids"][0]
        distances = results.get("distances", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        for i in range(len(ids)):
            formatted_results.append({
                "id": ids[i],
                "score": distances[i] if i < len(distances) else 0.0,
                "text": documents[i] if documents else "",
                "image_url": metadatas[i].get("image_url", ""),
                "local_image_path": metadatas[i].get("local_image_path", ""),
                "category": metadatas[i].get("category", ""),
                "reranked": reranked
            })
            
        return formatted_results

    def retrieve(self, query_text: str, top_k: int = 5):
        """
        [STAGE 1 ONLY] Dada una consulta de texto, genera el embedding con CLIP
        y busca el top_k en ChromaDB por similitud coseno.
        No aplica re-ranking.

        Args:
            query_text: La consulta del usuario.
            top_k: Cantidad de resultados a recuperar.
            
        Returns:
            Una lista de diccionarios con la información de cada documento recuperado.
        """
        query_embedding = self.embedder.embed_text(query_text)
        query_emb_list = query_embedding[0].tolist()
        results = self.db.search(query_embedding=query_emb_list, top_k=top_k)
        return self._format_results(results, reranked=False)

    def retrieve_and_rerank(self, query_text: str, top_k: int = 5, candidate_k: int = 20):
        """
        [PIPELINE DE DOS ETAPAS] Recuperación + Re-ranking semántico.

        Stage 1 — CLIP (Bi-Encoder):
            Recupera un pool ampliado de `candidate_k` candidatos desde ChromaDB
            usando similitud coseno en el espacio de embeddings de 512 dims.
            Es rápido y escala a millones de documentos.

        Stage 2 — Cross-Encoder (ms-marco-MiniLM-L-6-v2):
            Analiza cada par (query, texto_documento) con atención cruzada completa,
            produciendo un score de relevancia semántica más preciso que el Bi-Encoder.
            Re-ordena los candidatos y retorna solo los top_k finales.

        Args:
            query_text: La consulta del usuario.
            top_k: Cantidad de resultados FINALES a retornar (post-reranking).
            candidate_k: Pool de candidatos que recupera CLIP en Stage 1.
                         Debe ser >= top_k. Por defecto 20.

        Returns:
            Lista de diccionarios con los top_k documentos re-ordenados.
            Cada elemento incluye 'rerank_score' y 'reranked': True.
        """
        # --- STAGE 1: Bi-Encoder (CLIP) ---
        query_embedding = self.embedder.embed_text(query_text)
        query_emb_list = query_embedding[0].tolist()
        # Recuperar un pool más amplio que top_k para que el re-ranker tenga más candidatos
        effective_candidate_k = max(candidate_k, top_k)
        results = self.db.search(query_embedding=query_emb_list, top_k=effective_candidate_k)
        candidates = self._format_results(results, reranked=True)

        if not candidates:
            return []

        # --- STAGE 2: Cross-Encoder Re-ranking ---
        cross_encoder = self._get_cross_encoder()

        # Construir pares (query, documento) para el Cross-Encoder
        # Usamos texto truncado para no exceder los límites del tokenizador
        pairs = [(query_text, doc["text"][:512]) for doc in candidates]

        # Calcular scores de relevancia (el Cross-Encoder produce un score logit)
        rerank_scores = cross_encoder.predict(pairs)

        # Añadir el score del re-ranker a cada candidato
        for i, doc in enumerate(candidates):
            doc["rerank_score"] = float(rerank_scores[i])

        # Re-ordenar por score de relevancia (mayor = más relevante)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Retornar solo los top_k mejores según el Cross-Encoder
        return candidates[:top_k]

    def retrieve_with_expansion(
        self,
        query_text: str,
        top_k: int = 5,
        candidate_k: int = 20,
        n_expansions: int = 3
    ):
        """
        [PIPELINE DE TRES ETAPAS] Query Expansion + Retrieval + Re-ranking.

        Stage 0 — Query Expansion (Gemini LLM):
            Genera n_expansions reformulaciones de la consulta original.
            Ej: "guitarra acústica" → ["guitarra acústica", "guitarra clásica aprendizaje",
            "instrumento cuerdas principiante", "acoustic guitar beginner"]

        Stage 1 — CLIP (Bi-Encoder) por cada variante:
            Busca los top-candidate_k productos por cada variante de la consulta.
            Une los resultados y elimina duplicados por ID.
            Este pool ampliado maximiza el Recall: captura productos que la formulación
            original habría perdido.

        Stage 2 — Cross-Encoder Re-ranking:
            El Cross-Encoder puntúa cada par (query_original, documento) de forma profunda
            y selecciona los top_k con mayor relevancia semántica.

        Args:
            query_text: La consulta original del usuario.
            top_k: Número de resultados FINALES a retornar.
            candidate_k: Candidatos que recupera CLIP por cada variante.
            n_expansions: Número de reformulaciones adicionales a generar.

        Returns:
            Lista con los top_k documentos finales con campos:
            'rerank_score', 'reranked': True, 'expanded_from': query usada.
        """
        # --- STAGE 0: Query Expansion ---
        expander = self._get_query_expander()
        all_queries = expander.expand(query_text, n=n_expansions)

        # --- STAGE 1: CLIP multi-query retrieval ---
        seen_ids = set()
        all_candidates = []

        for q in all_queries:
            query_embedding = self.embedder.embed_text(q)
            query_emb_list = query_embedding[0].tolist()
            results = self.db.search(query_embedding=query_emb_list, top_k=candidate_k)
            candidates = self._format_results(results, reranked=True)
            for doc in candidates:
                if doc["id"] not in seen_ids:
                    seen_ids.add(doc["id"])
                    doc["expanded_from"] = q  # trazabilidad: qué variante lo encontró
                    all_candidates.append(doc)

        if not all_candidates:
            return []

        # --- STAGE 2: Cross-Encoder Re-ranking sobre el pool expandido ---
        cross_encoder = self._get_cross_encoder()
        # Siempre puntuar contra la consulta ORIGINAL para mayor consistencia
        pairs = [(query_text, doc["text"][:512]) for doc in all_candidates]
        rerank_scores = cross_encoder.predict(pairs)

        for i, doc in enumerate(all_candidates):
            doc["rerank_score"] = float(rerank_scores[i])

        all_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return all_candidates[:top_k]

    def retrieve_by_image(self, image, top_k: int = 5):
        """
        Dada una imagen (PIL.Image o ruta de archivo), genera el embedding visual
        usando CLIP y busca los productos más similares en ChromaDB.
        No aplica re-ranking (el Cross-Encoder trabaja solo con pares texto-texto).

        Args:
            image: Un objeto PIL.Image o una ruta (str) a un archivo de imagen.
            top_k: Cantidad de resultados a recuperar.
            
        Returns:
            Una lista de diccionarios con la información de cada documento recuperado.
        """
        if isinstance(image, Image.Image):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp, format="PNG")
                tmp_path = tmp.name
            try:
                image_embedding = self.embedder.embed_image(tmp_path)
            finally:
                os.unlink(tmp_path)
        else:
            image_embedding = self.embedder.embed_image(image)
        
        query_emb_list = image_embedding[0].tolist()
        results = self.db.search(query_embedding=query_emb_list, top_k=top_k)
        
        return self._format_results(results, reranked=False)
