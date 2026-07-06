"""
Módulo para la recuperación de documentos más relevantes (Retrieval).
"""
from src.embeddings import CLIPEmbedder
from src.vector_db import VectorDB

class MultimodalRetriever:
    """
    Clase que encapsula la lógica para convertir un query (texto)
    y buscarlo en la base de datos vectorial usando CLIP.
    """
    def __init__(self):
        # Inicializa bajo demanda para no saturar la memoria si solo importamos la clase
        self.embedder = CLIPEmbedder()
        self.db = VectorDB()

    def retrieve(self, query_text: str, top_k: int = 5):
        """
        Dada una consulta de texto, genera el embedding y busca el top_k en ChromaDB.
        
        Args:
            query_text: La consulta del usuario.
            top_k: Cantidad de resultados a recuperar.
            
        Returns:
            Una lista de diccionarios con la información de cada documento recuperado.
        """
        # 1. Generar embedding para la consulta (texto)
        query_embedding = self.embedder.embed_text(query_text)
        
        # 2. Buscar en la base de datos (convertir np.ndarray a list)
        query_emb_list = query_embedding[0].tolist()
        results = self.db.search(query_embedding=query_emb_list, top_k=top_k)
        
        # 3. Formatear la salida para que sea fácil de consumir por el frontend y el LLM
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
                "category": metadatas[i].get("category", "")
            })
            
        return formatted_results
