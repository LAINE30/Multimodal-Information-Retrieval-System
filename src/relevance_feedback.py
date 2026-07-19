"""
Módulo de Relevance Feedback.

Implementa un sistema de retroalimentación del usuario que:
1. Almacena calificaciones (👍/👎) por documento recuperado.
2. Calcula un factor de boost/penalización por documento basado en su historial.
3. Aplica ese factor al score de relevancia para mejorar búsquedas futuras.

Concepto de RI:
    Rocchio Algorithm (simplificado): en IR clásica, Rocchio modifica el vector
    de la consulta acercándolo a documentos relevantes y alejándolo de los
    no-relevantes. Nuestra implementación es una versión score-based: en vez
    de modificar vectores, ajustamos los scores finales del Cross-Encoder
    con un factor multiplicativo basado en el historial de feedback del usuario.
"""
import json
import os
from typing import Dict, List, Any, Optional
from collections import defaultdict

FEEDBACK_FILE = "data/evaluation/relevance_feedback.json"


class RelevanceFeedbackStore:
    """
    Almacena y consulta el feedback del usuario sobre documentos recuperados.
    Persiste en un archivo JSON local.
    """

    def __init__(self, filepath: str = FEEDBACK_FILE):
        self.filepath = filepath
        self._data = self._load()

    def _load(self) -> Dict:
        """Carga el archivo de feedback. Si no existe, inicializa vacío."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {"feedbacks": {}}
        return {"feedbacks": {}}

    def _save(self):
        """Persiste el feedback a disco."""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def add_feedback(self, doc_id: str, is_relevant: bool):
        """
        Registra un feedback del usuario sobre un documento.

        Args:
            doc_id: ID del documento en ChromaDB.
            is_relevant: True = 👍 (relevante), False = 👎 (no relevante).
        """
        if doc_id not in self._data["feedbacks"]:
            self._data["feedbacks"][doc_id] = {"likes": 0, "dislikes": 0}

        if is_relevant:
            self._data["feedbacks"][doc_id]["likes"] += 1
        else:
            self._data["feedbacks"][doc_id]["dislikes"] += 1

        self._save()

    def get_boost_factor(self, doc_id: str) -> float:
        """
        Calcula un factor de ajuste para el score de un documento basado
        en su historial de feedback.

        Fórmula (Rocchio score-based simplificado):
            factor = 1.0 + α * (likes - dislikes) / (likes + dislikes + 1)

        Donde α = 0.3 (peso máximo del boost/penalty).

        Retorna:
            float entre ~0.7 (muy penalizado) y ~1.3 (muy boosteado).
            1.0 si no hay feedback registrado.
        """
        if doc_id not in self._data["feedbacks"]:
            return 1.0

        fb = self._data["feedbacks"][doc_id]
        likes = fb["likes"]
        dislikes = fb["dislikes"]
        total = likes + dislikes

        if total == 0:
            return 1.0

        alpha = 0.3  # Peso máximo del ajuste
        net_score = (likes - dislikes) / (total + 1)
        return 1.0 + alpha * net_score

    def apply_feedback_to_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aplica el factor de boost/penalización del feedback a los scores
        de los resultados de búsqueda.

        Modifica el campo 'rerank_score' (si existe) o 'score' con el factor.
        Añade el campo 'feedback_boost' para transparencia.

        Args:
            results: Lista de documentos recuperados (con 'id' y 'score'/'rerank_score').

        Returns:
            La misma lista, con scores ajustados y re-ordenada.
        """
        for doc in results:
            boost = self.get_boost_factor(doc["id"])
            doc["feedback_boost"] = boost

            if "rerank_score" in doc:
                doc["rerank_score"] = doc["rerank_score"] * boost
            else:
                doc["score"] = doc["score"] * boost

        # Re-ordenar por score ajustado
        sort_key = "rerank_score" if results and "rerank_score" in results[0] else "score"
        results.sort(key=lambda x: x.get(sort_key, 0), reverse=True)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas globales del feedback recopilado.
        Útil para mostrar en el módulo de análisis.
        """
        total_docs = len(self._data["feedbacks"])
        total_likes = sum(fb["likes"] for fb in self._data["feedbacks"].values())
        total_dislikes = sum(fb["dislikes"] for fb in self._data["feedbacks"].values())

        return {
            "total_documents_rated": total_docs,
            "total_likes": total_likes,
            "total_dislikes": total_dislikes,
            "total_interactions": total_likes + total_dislikes,
            "satisfaction_rate": total_likes / max(total_likes + total_dislikes, 1)
        }
