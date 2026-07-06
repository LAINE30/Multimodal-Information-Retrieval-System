import pytest
from src.retrieval import MultimodalRetriever
from unittest.mock import patch, MagicMock

@patch('src.retrieval.CLIPEmbedder')
@patch('src.retrieval.VectorDB')
def test_multimodal_retriever(mock_vector_db, mock_embedder):
    """Prueba que el Retriever coordine bien Embedder y VectorDB."""
    # Configurar los mocks
    import numpy as np
    mock_emb_instance = MagicMock()
    mock_emb_instance.embed_text.return_value = np.array([[0.1, 0.2, 0.3]])
    mock_embedder.return_value = mock_emb_instance
    
    mock_db_instance = MagicMock()
    mock_db_instance.search.return_value = {
        "ids": [["doc1"]],
        "distances": [[0.5]],
        "documents": [["Texto de prueba"]],
        "metadatas": [[{"category": "test", "image_url": "url", "local_image_path": "path"}]]
    }
    mock_vector_db.return_value = mock_db_instance
    
    # Inicializar y probar
    retriever = MultimodalRetriever()
    results = retriever.retrieve("una prueba", top_k=1)
    
    # Verificaciones
    assert len(results) == 1
    assert results[0]["id"] == "doc1"
    assert results[0]["text"] == "Texto de prueba"
    assert results[0]["score"] == 0.5
    assert results[0]["category"] == "test"
