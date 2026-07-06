"""
Script para generar embeddings del corpus procesado e indexarlos en ChromaDB.
"""
import json
import os
from tqdm import tqdm
from src.embeddings import CLIPEmbedder
from src.vector_db import VectorDB

CORPUS_PATH = "data/processed/corpus.json"

def main():
    if not os.path.exists(CORPUS_PATH):
        print(f"No se encontró el corpus en {CORPUS_PATH}. Asegúrate de correr data_processing.py primero.")
        return

    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
        
    if not corpus:
        print("El corpus está vacío.")
        return
        
    print(f"Cargando modelo CLIP y VectorDB...")
    embedder = CLIPEmbedder()
    db = VectorDB()
    
    print(f"Indexando {len(corpus)} elementos...")
    
    batch_size = 32
    for i in tqdm(range(0, len(corpus), batch_size), desc="Generando e indexando batches"):
        batch = corpus[i:i + batch_size]
        
        ids = []
        documents = []
        metadatas = []
        image_paths = []
        
        for item in batch:
            ids.append(item['id'])
            documents.append(item['text'])
            metadatas.append({
                "category": item.get('category', ''),
                "image_url": item.get('image_url', ''),
                "local_image_path": item.get('local_image_path', '')
            })
            image_paths.append(item.get('local_image_path', ''))
            
        # Generamos embeddings de imágenes (multimodal)
        # Podríamos hacer embeddings de texto o combinados, pero CLIP proyecta ambos al mismo espacio.
        # Aquí usamos la imagen como representación principal en el DB, o el texto.
        # Lo ideal para RAG multimodal es tener los embeddings de las imágenes indexados,
        # y cuando el usuario hace un query de TEXTO, buscamos los vectores de imagen más cercanos.
        
        embeddings = embedder.embed_image(image_paths)
        
        # Guardamos en ChromaDB (convertimos np.ndarray a list de float)
        embeddings_list = embeddings.tolist()
        
        db.index_documents(ids, embeddings_list, documents, metadatas)
        
    print("¡Indexación completada exitosamente!")

if __name__ == "__main__":
    main()
