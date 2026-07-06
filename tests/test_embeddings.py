import numpy as np
import torch
from src.embeddings import CLIPEmbedder
from PIL import Image
import os

def test_clip_embedder_initialization():
    """Prueba que el modelo se inicialice correctamente."""
    embedder = CLIPEmbedder(model_name="ViT-B-32", pretrained="openai")
    assert embedder.model is not None
    assert embedder.tokenizer is not None

def test_clip_text_embedding():
    """Prueba que los embeddings de texto se generen y normalicen correctamente."""
    embedder = CLIPEmbedder(model_name="ViT-B-32", pretrained="openai")
    
    query = "Una guitarra acústica roja"
    embedding = embedder.embed_text(query)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, 512)  # ViT-B-32 genera un vector de tamaño 512
    # Verificar normalización (la norma L2 debe ser ~1.0)
    norm = np.linalg.norm(embedding[0])
    assert np.isclose(norm, 1.0, atol=1e-5)

def test_clip_image_embedding(tmp_path):
    """Prueba que los embeddings de imágenes se generen correctamente."""
    # Crear una imagen falsa para el test
    img_path = tmp_path / "test_image.jpg"
    img = Image.new('RGB', (224, 224), color = 'red')
    img.save(img_path)
    
    embedder = CLIPEmbedder(model_name="ViT-B-32", pretrained="openai")
    embedding = embedder.embed_image(str(img_path))
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, 512)
    norm = np.linalg.norm(embedding[0])
    assert np.isclose(norm, 1.0, atol=1e-5)
