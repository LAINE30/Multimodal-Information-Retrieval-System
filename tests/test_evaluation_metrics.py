"""
Tests unitarios para las métricas de evaluación de Recuperación de Información.
Verifica el cálculo correcto de Precision@k, Recall@k y NDCG@k.
"""
from src.evaluation_metrics import precision_at_k, recall_at_k, ndcg_at_k


def test_precision_at_k_perfect():
    """Si todos los recuperados son relevantes, Precision@k = 1.0."""
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = {"doc1", "doc2", "doc3"}
    assert precision_at_k(retrieved, relevant, k=3) == 1.0


def test_precision_at_k_none_relevant():
    """Si ninguno de los recuperados es relevante, Precision@k = 0.0."""
    retrieved = ["doc4", "doc5", "doc6"]
    relevant = {"doc1", "doc2", "doc3"}
    assert precision_at_k(retrieved, relevant, k=3) == 0.0


def test_precision_at_k_partial():
    """Si 1 de 3 recuperados es relevante, Precision@3 = 1/3."""
    retrieved = ["doc1", "doc4", "doc5"]
    relevant = {"doc1", "doc2", "doc3"}
    assert abs(precision_at_k(retrieved, relevant, k=3) - 1/3) < 1e-6


def test_recall_at_k_perfect():
    """Si todos los relevantes están en Top-k, Recall@k = 1.0."""
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    relevant = {"doc1", "doc2"}
    assert recall_at_k(retrieved, relevant, k=3) == 1.0


def test_recall_at_k_partial():
    """Si solo 1 de 2 relevantes está en Top-3, Recall@3 = 0.5."""
    retrieved = ["doc1", "doc4", "doc5"]
    relevant = {"doc1", "doc2"}
    assert recall_at_k(retrieved, relevant, k=3) == 0.5


def test_recall_at_k_empty_relevant():
    """Si no hay documentos relevantes, Recall = 0.0 (evitar div by zero)."""
    retrieved = ["doc1", "doc2"]
    relevant = set()
    assert recall_at_k(retrieved, relevant, k=2) == 0.0


def test_ndcg_at_k_perfect_ranking():
    """Si el documento relevante está en la posición 1, NDCG@3 = 1.0."""
    retrieved = ["doc1", "doc4", "doc5"]
    relevant = {"doc1"}
    assert abs(ndcg_at_k(retrieved, relevant, k=3) - 1.0) < 1e-6


def test_ndcg_at_k_worst_ranking():
    """Si el documento relevante está en la última posición del Top-k, NDCG < 1.0."""
    retrieved = ["doc4", "doc5", "doc1"]
    relevant = {"doc1"}
    ndcg = ndcg_at_k(retrieved, relevant, k=3)
    assert ndcg < 1.0
    assert ndcg > 0.0


def test_ndcg_at_k_no_relevant():
    """Si no hay documentos relevantes, NDCG = 0.0."""
    retrieved = ["doc1", "doc2"]
    relevant = set()
    assert ndcg_at_k(retrieved, relevant, k=2) == 0.0


def test_ndcg_at_k_none_found():
    """Si ningún relevante fue recuperado, NDCG = 0.0."""
    retrieved = ["doc4", "doc5", "doc6"]
    relevant = {"doc1", "doc2"}
    assert ndcg_at_k(retrieved, relevant, k=3) == 0.0


def test_precision_at_k_zero():
    """Si k=0, la Precision es 0.0."""
    retrieved = ["doc1"]
    relevant = {"doc1"}
    assert precision_at_k(retrieved, relevant, k=0) == 0.0
