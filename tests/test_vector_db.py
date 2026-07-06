import os
import tempfile
import chromadb
from src.vector_db import VectorDB

def test_vector_db_initialization():
    """Prueba que ChromaDB se inicialice correctamente."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        db = VectorDB(db_dir=temp_dir, collection_name="test_collection")
        assert db.client is not None
        assert db.collection.name == "test_collection"

def test_vector_db_index_and_search():
    """Prueba la indexación y recuperación de vectores."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        db = VectorDB(db_dir=temp_dir, collection_name="test_collection")
        
        # Datos falsos para probar (vectores de 3 dimensiones simulando CLIP)
        ids = ["doc1", "doc2", "doc3"]
        embeddings = [
            [1.0, 0.0, 0.0],  # Documento 1
            [0.0, 1.0, 0.0],  # Documento 2
            [0.0, 0.0, 1.0]   # Documento 3
        ]
        documents = ["Texto sobre manzanas", "Texto sobre computadoras", "Texto sobre autos"]
        metadatas = [{"category": "Frutas"}, {"category": "Tecnología"}, {"category": "Vehículos"}]
        
        # Indexar
        db.index_documents(ids, embeddings, documents, metadatas)
        
        # El documento insertado debe estar en la colección
        assert db.collection.count() == 3
        
        # Búsqueda similar al Doc 2
        query_embedding = [0.1, 0.9, 0.0]
        results = db.search(query_embedding, top_k=1)
        
        # Verificar resultados
        assert len(results["ids"][0]) == 1
        assert results["ids"][0][0] == "doc2"
        assert results["documents"][0][0] == "Texto sobre computadoras"
        assert results["metadatas"][0][0]["category"] == "Tecnología"
