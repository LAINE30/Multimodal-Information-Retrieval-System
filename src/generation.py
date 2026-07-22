"""
Módulo para la generación de respuestas usando un LLM (Retrieval-Augmented Generation).
Incluye memoria conversacional para mantener contexto entre turnos.
"""
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Any

class RAGGenerator:
    """
    Clase que maneja la formulación del contexto y la generación de la respuesta
    conversacional usando un modelo de lenguaje.
    Soporta memoria conversacional: inyecta el historial de turnos previos
    en el prompt para que Gemini mantenga coherencia a lo largo del chat.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.7):
        """
        Inicializa el modelo de lenguaje de LangChain (Gemini).
        Nota: Requiere que la variable de entorno GOOGLE_API_KEY esté configurada.
        """
        # Si no hay API key, inicializar fallará al intentar invocar
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        
        # Configurar el SDK nativo de Google para llamadas directas multimodales (ej. audio)
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        
        # Definir el template del RAG con soporte para memoria conversacional.
        # La variable {chat_history} inyecta los últimos turnos del chat.
        prompt_template = """Eres un asistente experto de compras con memoria conversacional.
Utiliza la siguiente información de productos (contexto) recuperada de nuestra base de datos
para responder a la pregunta del usuario.

Reglas:
1. Responde de forma amigable y conversacional.
2. Si recomiendas un producto del contexto, menciona su título y alguna característica relevante.
3. Si la respuesta no está en el contexto, indica amablemente que no encontraste información exacta, pero sugiere algo relacionado si es posible.
4. Si el tipo de consulta es "imagen", el usuario subió una foto para buscar productos similares. Describe los productos encontrados que se asemejan visualmente a lo que el usuario buscó.
5. MEMORIA: Tienes acceso al historial de conversación previo. Úsalo para entender referencias como "ese producto", "el primero", "algo más barato", "cuéntame más", etc. Mantén coherencia con lo que ya dijiste.

Tipo de Consulta: {query_type}

--- Historial de Conversación (últimos turnos) ---
{chat_history}
--- Fin del Historial ---

Contexto Recuperado (productos de la base de datos):
{context}

Pregunta Actual del Usuario: {question}

Respuesta:"""
        
        self.prompt = PromptTemplate(
            input_variables=["context", "question", "query_type", "chat_history"],
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

    @staticmethod
    def format_chat_history(messages: List[Dict[str, Any]], max_turns: int = 6) -> str:
        """
        Convierte el historial de mensajes de Streamlit en un texto resumido
        para inyectarlo en el prompt del LLM.
        
        Solo incluye los últimos `max_turns` mensajes (pares usuario/asistente)
        para no exceder la ventana de contexto ni degradar la calidad.
        Omite las evidencias/metadatos; solo usa el texto de cada mensaje.

        Args:
            messages: Lista de dicts con 'role' y 'content' (st.session_state.messages).
            max_turns: Número máximo de mensajes a incluir.

        Returns:
            String formateado con el historial conversacional.
        """
        if not messages:
            return "(No hay historial previo. Esta es la primera interacción.)"

        # Tomar solo los últimos N mensajes
        recent = messages[-max_turns:]
        
        history_lines = []
        for msg in recent:
            role_label = "Usuario" if msg["role"] == "user" else "Asistente"
            # Truncar mensajes muy largos del asistente para no llenar la ventana
            content = msg["content"]
            if msg["role"] == "assistant" and len(content) > 300:
                content = content[:300] + "..."
            history_lines.append(f"{role_label}: {content}")
        
        return "\n".join(history_lines)

    def generate_response(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        query_type: str = "texto",
        chat_history: str = ""
    ) -> str:
        """
        Genera la respuesta final usando el contexto recuperado y la memoria conversacional.
        
        Args:
            query: La consulta original del usuario.
            retrieved_docs: Los resultados devueltos por MultimodalRetriever.
            query_type: "texto" si el usuario escribió, "imagen" si subió una foto.
            chat_history: Historial de conversación formateado como texto.
            
        Returns:
            La respuesta generada por el LLM en formato de texto.
        """
        context_str = self.format_context(retrieved_docs)
        
        try:
            response = self.chain.invoke({
                "context": context_str, 
                "question": query,
                "query_type": query_type,
                "chat_history": chat_history if chat_history else "(Primera interacción)"
            })
            return response.content
        except Exception as e:
            return f"Error generando la respuesta (¿Revisaste tu API Key en el archivo .env?): {e}"

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        """
        Transcribe un archivo de audio a texto usando Gemini.
        
        Args:
            audio_bytes: Los bytes crudos del audio capturado (ej. formato WAV).
            
        Returns:
            El texto transcrito por la IA.
        """
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content([
                {"mime_type": "audio/wav", "data": audio_bytes},
                "Transcribe exactamente las palabras de este audio en su idioma original. No agregues comillas, saludos ni contexto. Solo el texto dicho."
            ])
            return response.text.strip()
        except Exception as e:
            return f"[Error en transcripción]: {e}"
