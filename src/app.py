"""
Interfaz web principal utilizando Streamlit.
"""
import streamlit as st
import sys
import os
# Añadir el directorio raíz al path para que Python encuentre el paquete 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Cargar variables de entorno (como GOOGLE_API_KEY)
load_dotenv()

# Configuración de página
st.set_page_config(page_title="RAG Multimodal de Productos", layout="wide")

# Solo inicializamos el pipeline si la API Key está configurada (para no arrojar error feo al inicio)
if "GOOGLE_API_KEY" not in os.environ or not os.environ["GOOGLE_API_KEY"]:
    st.error("⚠️ No se encontró la variable GOOGLE_API_KEY. Por favor, añádela a tu archivo .env o al entorno antes de continuar.")
    st.stop()

# Cachear la inicialización de los modelos para que no se recarguen en cada iteración de Streamlit
@st.cache_resource
def load_pipeline():
    # Importaciones aquí para no ralentizar el arranque inicial de Streamlit si falla la API key
    from src.retrieval import MultimodalRetriever
    from src.generation import RAGGenerator
    
    retriever = MultimodalRetriever()
    generator = RAGGenerator(model_name="gemini-2.5-flash", temperature=0.7)
    return retriever, generator

with st.spinner("Cargando modelo CLIP y base de datos (Esto toma unos segundos la primera vez)..."):
    retriever, generator = load_pipeline()

st.title("🛒 Asistente de Compras Inteligente (RAG Multimodal)")
st.write("Consulta sobre instrumentos, videojuegos, electrónica, etc. El sistema recuperará las imágenes y descripciones relevantes para responderte.")

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial del chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Mostrar evidencias visuales si las hay
        if "evidences" in msg and msg["evidences"]:
            st.write("**Evidencias utilizadas:**")
            cols = st.columns(len(msg["evidences"]))
            for idx, ev in enumerate(msg["evidences"]):
                with cols[idx]:
                    if os.path.exists(ev["local_image_path"]):
                        st.image(ev["local_image_path"], use_container_width=True)
                    st.caption(f"Score: {ev['score']:.4f}")
                    with st.expander("Ver texto original"):
                        st.write(ev["text"])

# Capturar input del usuario
if prompt := st.chat_input("Ej: ¿Tienes alguna guitarra acústica ideal para principiantes?"):
    # Agregar y mostrar mensaje de usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Respuesta del asistente
    with st.chat_message("assistant"):
        with st.spinner("Buscando en la base de datos..."):
            # 1. Recuperación Vectorial (Retrieval)
            evidences = retriever.retrieve(prompt, top_k=3)
            
        with st.spinner("Generando respuesta..."):
            # 2. Generación (LLM)
            answer = generator.generate_response(prompt, evidences)
            
            # 3. Renderizar Respuesta
            st.markdown(answer)
            
            # 4. Renderizar Evidencias
            if evidences:
                st.write("---")
                st.write("**Documentos recuperados (Evidencias):**")
                cols = st.columns(len(evidences))
                for idx, ev in enumerate(evidences):
                    with cols[idx]:
                        if os.path.exists(ev["local_image_path"]):
                            st.image(ev["local_image_path"], use_container_width=True)
                        st.caption(f"Similitud: {ev['score']:.4f}")
                        with st.expander("Ver detalles"):
                            st.write(f"**Categoría:** {ev['category']}")
                            st.write(ev["text"])
            else:
                st.info("No se encontró información relevante en la base de datos para esta consulta.")
                
        # Guardar en historial
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "evidences": evidences
        })
