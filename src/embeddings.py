"""
Módulo para la generación de embeddings multimodales usando CLIP.
"""
import torch
import open_clip
from PIL import Image
from typing import List, Union
import numpy as np

class CLIPEmbedder:
    """
    Clase para manejar la carga del modelo CLIP y la generación de embeddings
    tanto para imágenes como para texto.
    """
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        """
        Inicializa el modelo CLIP. 
        Se usa ViT-B-32 por defecto ya que es ligero y apto para GPUs como la GTX 1050Ti.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Cargando modelo CLIP ({model_name}) en {self.device}...")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
    @torch.no_grad()
    def embed_image(self, image_paths: Union[str, List[str]]) -> np.ndarray:
        """
        Genera embeddings para una o múltiples imágenes.
        
        Args:
            image_paths: Ruta única o lista de rutas a imágenes.
            
        Returns:
            np.ndarray con los embeddings normalizados.
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]
            
        images = []
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    img_rgb = img.convert("RGB")
                images.append(self.preprocess(img_rgb))
            except Exception as e:
                print(f"Error cargando la imagen {path}: {e}")
                # Si falla, añadimos un tensor vacío para no romper el batch
                images.append(torch.zeros(3, 224, 224))
                
        image_input = torch.tensor(np.stack(images)).to(self.device)
        
        # Generar embeddings
        image_features = self.model.encode_image(image_input)
        
        # Normalizar para facilitar similitud coseno
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()

    @torch.no_grad()
    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Genera embeddings para uno o múltiples textos.
        
        Args:
            texts: String único o lista de strings.
            
        Returns:
            np.ndarray con los embeddings normalizados.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        text_input = self.tokenizer(texts).to(self.device)
        
        # Generar embeddings
        text_features = self.model.encode_text(text_input)
        
        # Normalizar para facilitar similitud coseno
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()
