"""
Módulo para el cálculo de métricas de evaluación de Recuperación de Información.
Implementa Precision@k, Recall@k y NDCG@k.
"""
import numpy as np
from typing import List, Set


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calcula la Precisión en el Top-K.
    
    Mide qué proporción de los documentos recuperados son realmente relevantes.
    Fórmula: Precision@k = |{documentos relevantes en Top-k}| / k
    
    Args:
        retrieved_ids: Lista ordenada de IDs de documentos recuperados por el sistema.
        relevant_ids: Conjunto de IDs de documentos que sabemos son relevantes (ground truth).
        k: Número de resultados a considerar.
        
    Returns:
        Float entre 0.0 y 1.0.
    """
    if k == 0:
        return 0.0
    # Tomar solo los primeros k resultados
    top_k = retrieved_ids[:k]
    # Contar cuántos de los recuperados son relevantes
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calcula el Recall en el Top-K.
    
    Mide qué proporción de los documentos relevantes totales fueron encontrados en el Top-K.
    Fórmula: Recall@k = |{documentos relevantes en Top-k}| / |{total de documentos relevantes}|
    
    Args:
        retrieved_ids: Lista ordenada de IDs de documentos recuperados por el sistema.
        relevant_ids: Conjunto de IDs de documentos que sabemos son relevantes (ground truth).
        k: Número de resultados a considerar.
        
    Returns:
        Float entre 0.0 y 1.0. Si no hay documentos relevantes, retorna 0.0.
    """
    if len(relevant_ids) == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / len(relevant_ids)


def dcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calcula el DCG (Discounted Cumulative Gain) en el Top-K.
    
    El DCG penaliza los documentos relevantes que aparecen más abajo en el ranking,
    aplicando un descuento logarítmico a cada posición.
    Fórmula: DCG@k = Σ (rel_i / log2(i + 1)) para i de 1 a k
    
    Args:
        retrieved_ids: Lista ordenada de IDs de documentos recuperados.
        relevant_ids: Conjunto de IDs de documentos relevantes.
        k: Número de resultados a considerar.
        
    Returns:
        Float con el valor de DCG.
    """
    top_k = retrieved_ids[:k]
    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        # Relevancia binaria: 1 si es relevante, 0 si no
        rel = 1.0 if doc_id in relevant_ids else 0.0
        # Descuento logarítmico: posición 1 → log2(2)=1, pos 2 → log2(3)≈1.58, etc.
        dcg += rel / np.log2(i + 2)  # i+2 porque i empieza en 0 y la fórmula usa i+1 (1-indexed)
    return dcg


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calcula el NDCG (Normalized Discounted Cumulative Gain) en el Top-K.
    
    Normaliza el DCG dividiendo por el DCG ideal (IDCG), que es el mejor DCG posible
    si los documentos relevantes estuvieran todos en las primeras posiciones.
    Fórmula: NDCG@k = DCG@k / IDCG@k
    
    Args:
        retrieved_ids: Lista ordenada de IDs de documentos recuperados.
        relevant_ids: Conjunto de IDs de documentos relevantes.
        k: Número de resultados a considerar.
        
    Returns:
        Float entre 0.0 y 1.0. Si no hay documentos relevantes, retorna 0.0.
    """
    if len(relevant_ids) == 0:
        return 0.0
    
    # Calcular DCG real
    dcg = dcg_at_k(retrieved_ids, relevant_ids, k)
    
    # Calcular IDCG (DCG ideal): los relevantes en las primeras posiciones
    # Crear un ranking "perfecto" con todos los relevantes primero
    ideal_retrieved = list(relevant_ids)[:k]
    idcg = dcg_at_k(ideal_retrieved, relevant_ids, k)
    
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg
