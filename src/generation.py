"""
Módulo para la generación de respuestas usando un LLM (Retrieval-Augmented Generation).
"""
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Any

class RAGGenerator:
    """
    Clase que maneja la formulación del contexto y la generación de la respuesta
    conversacional usando un modelo de lenguaje.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.7):
        """
        Inicializa el modelo de lenguaje de LangChain (Gemini).
        Nota: Requiere que la variable de entorno GOOGLE_API_KEY esté configurada.
        """
        # Si no hay API key, inicializar fallará al intentar invocar
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        
        # Definir el template del RAG (soporta consultas por texto e imagen)
        prompt_template = """
        Eres un asistente experto de compras. Utiliza la siguiente información de productos (contexto)
        recuperada de nuestra base de datos para responder a la pregunta del usuario.
        
        Reglas:
        1. Responde de forma amigable y conversacional.
        2. Si recomiendas un producto del contexto, menciona su título y alguna característica relevante.
        3. Si la respuesta no está en el contexto, indica amablemente que no encontraste información exacta, pero sugiere algo relacionado si es posible.
        4. Si el tipo de consulta es "imagen", el usuario subió una foto para buscar productos similares. Describe los productos encontrados que se asemejan visualmente a lo que el usuario buscó.
        
        Tipo de Consulta: {query_type}
        
        Contexto Recuperado:
        {context}
        
        Pregunta del Usuario: {question}
        
        Respuesta:
        """
        
        self.prompt = PromptTemplate(
            input_variables=["context", "question", "query_type"],
            template=prompt_template
        )
        
        self.chain = self.prompt | self.llm

    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Formatea los documentos recuperados en un solo texto para pasarlo al LLM.
        """
        if not retrieved_docs:
            return "No se encontraron productos relevantes."
            
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            title = doc.get('title', 'Producto sin nombre')
            text = doc.get('text', '')
            context_parts.append(f"Producto {i}:\n- Título: {title}\n- Descripción: {text}")
            
        return "\n\n".join(context_parts)

    def generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]], query_type: str = "texto") -> str:
        """
        Genera la respuesta final usando el contexto recuperado.
        
        Args:
            query: La consulta original del usuario (texto descriptivo o "Búsqueda por imagen").
            retrieved_docs: Los resultados devueltos por MultimodalRetriever.
            query_type: "texto" si el usuario escribió, "imagen" si subió una foto.
            
        Returns:
            La respuesta generada por el LLM en formato de texto.
        """
        context_str = self.format_context(retrieved_docs)
        
        try:
            response = self.chain.invoke({
                "context": context_str, 
                "question": query,
                "query_type": query_type
            })
            return response.content
        except Exception as e:
            return f"Error generando la respuesta (¿Revisaste tu API Key en el archivo .env?): {e}"
