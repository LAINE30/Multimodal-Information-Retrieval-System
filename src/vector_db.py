"""
Módulo para la gestión de la base de datos vectorial (ChromaDB).
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import os

class VectorDB:
    """
    Clase para manejar la persistencia y búsqueda vectorial usando ChromaDB.
    """
    def __init__(self, db_dir: str = "data/chroma_db", collection_name: str = "multimodal_corpus"):
        """
        Inicializa el cliente de ChromaDB con persistencia en disco.
        """
        os.makedirs(db_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_dir)
        
        # Obtenemos o creamos la colección. Usamos la métrica coseno por defecto.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def index_documents(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict[str, Any]]):
        """
        Agrega documentos y sus embeddings a la base de datos.
        
        Args:
            ids: Lista de identificadores únicos.
            embeddings: Lista de listas de floats (los vectores de CLIP).
            documents: Lista de textos del corpus.
            metadatas: Lista de diccionarios con metadatos (ej. URL de imagen, categoría).
        """
        # Añadimos los elementos en lotes para mayor seguridad
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size]
            )
            
    def search(self, query_embedding: List[float], top_k: int = 5) -> Dict[str, Any]:
        """
        Busca los top_k documentos más similares al embedding de consulta.
        
        Args:
            query_embedding: El vector generado por CLIP para la consulta del usuario.
            top_k: Cantidad de documentos a recuperar.
            
        Returns:
            Diccionario con los resultados (ids, distances, documents, metadatas).
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results
