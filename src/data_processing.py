"""
Módulo para el procesamiento de datos multimodales (texto e imágenes).
Descarga un subset del dataset Amazon Reviews 2023 usando Hugging Face Datasets.
"""
import os
import json
import requests
from io import BytesIO
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

# Configuración
CATEGORIES = [
    "raw_meta_Musical_Instruments",
    "raw_meta_Video_Games",
    "raw_meta_Pet_Supplies",
    "raw_meta_Camera_and_Photo",
    "raw_meta_Electronics",
    "raw_meta_Sports_and_Outdoors"
]
LIMIT_PER_CATEGORY = 50  # Puedes subir esto a 500, 1000 o más dependiendo de tu tiempo y espacio
OUTPUT_CORPUS = "data/processed/corpus.json"
IMAGES_DIR = "data/raw/images"

def setup_directories():
    """Asegura que los directorios necesarios existan."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_CORPUS), exist_ok=True)

def download_image(url: str, save_path: str) -> bool:
    """Descarga y guarda una imagen desde una URL. Retorna True si es exitoso."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        # Convertir a RGB para evitar problemas con RGBA/P
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(save_path, "JPEG")
        return True
    except Exception as e:
        print(f"Error descargando {url}: {e}")
        return False

def extract_product_data(product, category: str):
    """Extrae la información relevante de un producto."""
    # Verificar que tenga título
    title = product.get('title', '')
    if not title:
        return None
        
    # Verificar que tenga imágenes
    images = product.get('images', [])
    if not images or len(images) == 0:
        return None
        
    img_url = None
    if isinstance(images, dict):
        # En Amazon 2023, images es un dict con listas bajo 'hi_res' o 'large'
        hi_res = images.get('hi_res', [])
        if hi_res and isinstance(hi_res, list) and hi_res[0]:
            img_url = hi_res[0]
        else:
            large = images.get('large', [])
            if large and isinstance(large, list) and large[0]:
                img_url = large[0]
    elif isinstance(images, list) and len(images) > 0:
        # A veces es una lista de diccionarios
        if isinstance(images[0], dict):
            img_url = images[0].get('hi_res', None) or images[0].get('large', None)
        # O una lista de strings
        elif isinstance(images[0], str):
            img_url = images[0]
            
    if not img_url:
        return None

    # Extraer descripción (puede ser una lista)
    description = product.get('description', [])
    if isinstance(description, list):
        description = " ".join(description)
        
    # Extraer características (puede ser una lista)
    features = product.get('features', [])
    if isinstance(features, list):
        features = " ".join(features)
        
    # Combinar texto rico
    full_text = f"{title}. {description} {features}".strip()
    
    if len(full_text) < 20: # Ignorar productos sin información suficiente
        return None
        
    # Crear un ID único
    product_id = f"{category}_{product.get('parent_asin', 'unknown')}"
    
    return {
        "id": product_id,
        "title": title,
        "text": full_text,
        "image_url": img_url,
        "category": category.replace("raw_meta_", "")
    }

def main():
    setup_directories()
    corpus = []
    
    # Cargar corpus existente si existe para no empezar de cero
    if os.path.exists(OUTPUT_CORPUS):
        with open(OUTPUT_CORPUS, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
            
    existing_ids = {item['id'] for item in corpus}
    
    for category in CATEGORIES:
        print(f"\\nProcesando categoría: {category}")
        try:
            # Usar streaming para no descargar el metadata completo
            dataset = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023", 
                category, 
                split="full", 
                streaming=True,
                trust_remote_code=True
            )
            
            count = 0
            for item in tqdm(dataset, desc=f"Descargando {category}"):
                if count >= LIMIT_PER_CATEGORY:
                    break
                    
                processed_item = extract_product_data(item, category)
                if processed_item is None:
                    continue
                    
                if processed_item['id'] in existing_ids:
                    continue
                    
                # Definir ruta de guardado local para la imagen
                local_image_path = os.path.join(IMAGES_DIR, f"{processed_item['id']}.jpg")
                
                # Descargar imagen
                if download_image(processed_item['image_url'], local_image_path):
                    processed_item['local_image_path'] = local_image_path
                    corpus.append(processed_item)
                    existing_ids.add(processed_item['id'])
                    count += 1
                    
        except Exception as e:
            print(f"Error procesando la categoría {category}: {e}")
            
    # Guardar corpus actualizado
    with open(OUTPUT_CORPUS, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=4)
        
    print(f"\\nProcesamiento completado. Corpus guardado en {OUTPUT_CORPUS} con {len(corpus)} elementos en total.")

if __name__ == "__main__":
    main()
