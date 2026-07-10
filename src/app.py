"""
Interfaz web principal utilizando Streamlit.
"""
import streamlit as st
import sys
import os
# Añadir el directorio raíz al path para que Python encuentre el paquete 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from PIL import Image

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
st.write("Consulta sobre instrumentos, videojuegos, electrónica, etc. Puedes escribir texto o subir una imagen para buscar productos similares.")

# --- Sidebar: Subida de imágenes ---
with st.sidebar:
    st.header("🖼️ Búsqueda por Imagen")
    st.write("Sube una foto de un producto para encontrar artículos similares en nuestra base de datos.")
    uploaded_file = st.file_uploader(
        "Arrastra o selecciona una imagen",
        type=["jpg", "jpeg", "png", "webp"],
        help="Formatos soportados: JPG, PNG, WEBP"
    )
    
    # Si hay imagen subida, mostrar una vista previa en el sidebar
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_image, caption="Imagen subida", use_container_width=True)
        
        # Botón para ejecutar la búsqueda por imagen
        search_by_image = st.button("🔍 Buscar productos similares", use_container_width=True)
    else:
        uploaded_image = None
        search_by_image = False

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial del chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Si el mensaje del usuario incluía una imagen, mostrarla
        if "user_image" in msg and msg["user_image"] is not None:
            st.image(msg["user_image"], caption="Imagen del usuario", width=200)
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

# --- Flujo de Búsqueda por Imagen (desde el sidebar) ---
if search_by_image and uploaded_image is not None:
    # Agregar mensaje del usuario con la imagen
    st.session_state.messages.append({
        "role": "user", 
        "content": "🖼️ *Búsqueda por imagen subida*",
        "user_image": uploaded_image
    })
    with st.chat_message("user"):
        st.image(uploaded_image, caption="Imagen del usuario", width=200)
        st.markdown("🖼️ *Búsqueda por imagen subida*")

    # Respuesta del asistente
    with st.chat_message("assistant"):
        with st.spinner("Analizando imagen con CLIP y buscando en la base de datos..."):
            # 1. Recuperación Visual (Image Retrieval)
            evidences = retriever.retrieve_by_image(uploaded_image, top_k=3)
            
        with st.spinner("Generando respuesta con Gemini..."):
            # 2. Generación (LLM) — le indicamos que fue una consulta por imagen
            answer = generator.generate_response(
                "El usuario subió una imagen de un producto. Describe los productos similares encontrados.",
                evidences,
                query_type="imagen"
            )
            
            # 3. Renderizar Respuesta
            st.markdown(answer)
            
            # 4. Renderizar Evidencias
            if evidences:
                st.write("---")
                st.write("**Productos similares encontrados:**")
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
                st.info("No se encontraron productos similares a la imagen subida.")
                
        # Guardar en historial
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "evidences": evidences
        })

# --- Flujo de Búsqueda por Texto (chat input) ---
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
            answer = generator.generate_response(prompt, evidences, query_type="texto")
            
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
