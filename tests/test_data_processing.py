import os
from src.data_processing import extract_product_data

def test_extract_product_data_valid():
    """Prueba que un producto válido se procesa correctamente."""
    dummy_product = {
        "title": "Guitarra Acústica",
        "description": ["Una guitarra para principiantes."],
        "features": ["Cuerdas de nylon", "Tamaño 4/4"],
        "images": [{"hi_res": "http://example.com/image.jpg"}],
        "parent_asin": "B012345"
    }
    
    result = extract_product_data(dummy_product, "raw_meta_Musical_Instruments")
    
    assert result is not None
    assert result["id"] == "raw_meta_Musical_Instruments_B012345"
    assert result["title"] == "Guitarra Acústica"
    assert "Una guitarra para principiantes." in result["text"]
    assert "Cuerdas de nylon" in result["text"]
    assert result["image_url"] == "http://example.com/image.jpg"
    assert result["category"] == "Musical_Instruments"

def test_extract_product_data_missing_image():
    """Prueba que un producto sin imagen se ignora."""
    dummy_product = {
        "title": "Guitarra sin imagen",
        "description": ["Descripción de prueba"],
        "images": [],
        "parent_asin": "B012346"
    }
    
    result = extract_product_data(dummy_product, "raw_meta_Musical_Instruments")
    assert result is None

def test_extract_product_data_missing_title():
    """Prueba que un producto sin título se ignora."""
    dummy_product = {
        "description": ["Descripción de prueba"],
        "images": [{"hi_res": "http://example.com/img.jpg"}],
        "parent_asin": "B012347"
    }
    
    result = extract_product_data(dummy_product, "raw_meta_Musical_Instruments")
    assert result is None
