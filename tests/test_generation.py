import pytest
from src.generation import RAGGenerator
from unittest.mock import patch, MagicMock

@patch('src.generation.ChatGoogleGenerativeAI')
def test_rag_generator_format_context(mock_chat):
    generator = RAGGenerator()
    docs = [
        {"title": "Guitarra", "text": "Acústica de madera"},
        {"title": "Piano", "text": "De cola"}
    ]
    
    context = generator.format_context(docs)
    assert "Guitarra" in context
    assert "Acústica de madera" in context
    assert "Piano" in context

@patch('src.generation.PromptTemplate')
@patch('src.generation.ChatGoogleGenerativeAI')
def test_rag_generator_response(mock_chat, mock_prompt):
    # Simular la respuesta del modelo en la cadena LCEL
    mock_response = MagicMock()
    mock_response.content = "Esta es la respuesta generada."
    
    mock_chain_instance = MagicMock()
    mock_chain_instance.invoke.return_value = mock_response
    
    generator = RAGGenerator()
    generator.chain = mock_chain_instance
    
    response = generator.generate_response("pregunta", [])
    assert response == "Esta es la respuesta generada."
