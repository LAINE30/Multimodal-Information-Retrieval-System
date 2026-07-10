"""
Módulo para la recuperación de documentos más relevantes (Retrieval).
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
    """
    def __init__(self):
        # Inicializa bajo demanda para no saturar la memoria si solo importamos la clase
        self.embedder = CLIPEmbedder()
        self.db = VectorDB()

    def _format_results(self, results):
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
                "category": metadatas[i].get("category", "")
            })
            
        return formatted_results

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
        
        # 3. Formatear la salida
        return self._format_results(results)

    def retrieve_by_image(self, image, top_k: int = 5):
        """
        Dada una imagen (PIL.Image o ruta de archivo), genera el embedding visual
        usando CLIP y busca los productos más similares en ChromaDB.
        
        Args:
            image: Un objeto PIL.Image o una ruta (str) a un archivo de imagen.
            top_k: Cantidad de resultados a recuperar.
            
        Returns:
            Una lista de diccionarios con la información de cada documento recuperado.
        """
        # Si recibimos un objeto PIL.Image, lo guardamos temporalmente para que
        # embed_image pueda leerlo desde disco (la API de CLIPEmbedder espera rutas)
        if isinstance(image, Image.Image):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp, format="PNG")
                tmp_path = tmp.name
            try:
                image_embedding = self.embedder.embed_image(tmp_path)
            finally:
                os.unlink(tmp_path)
        else:
            # Si es un string (ruta), lo pasamos directamente
            image_embedding = self.embedder.embed_image(image)
        
        # Buscar en ChromaDB con el vector visual
        query_emb_list = image_embedding[0].tolist()
        results = self.db.search(query_embedding=query_emb_list, top_k=top_k)
        
        return self._format_results(results)

