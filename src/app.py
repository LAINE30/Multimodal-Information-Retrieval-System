"""
Interfaz web principal utilizando Streamlit.
"""
import streamlit as st
import sys
import os
import pandas as pd
import json
import time

# Añadir el directorio raíz al path para que Python encuentre el paquete 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from PIL import Image

# Cargar variables de entorno (como GOOGLE_API_KEY)
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Si no existe en .env, buscar en Streamlit Secrets
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("No se encontró GOOGLE_API_KEY.")
    st.stop()

# Configuración de página
st.set_page_config(page_title="RAG Multimodal", page_icon="🛍️", layout="wide")

# CSS para darle estilo de App Gallery adaptable a Modo Claro/Oscuro
st.markdown("""
<style>
    /* Títulos y tipografía */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
    
    /* Estilizar las tarjetas de productos (evidencias) usando variables del tema actual */
    div[data-testid="stColumn"] {
        background-color: var(--secondary-background-color);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        border: 1px solid var(--faded-text60);
        margin-bottom: 10px;
    }
    
    /* Estilizar imágenes dentro de las columnas para redondearlas */
    div[data-testid="stImage"] img {
        border-radius: 8px;
    }
    
    /* Botones primarios estéticos */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        opacity: 0.8;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    from src.retrieval import MultimodalRetriever
    from src.generation import RAGGenerator
    retriever = MultimodalRetriever()
    generator = RAGGenerator(model_name="gemini-2.5-flash", temperature=0.7)
    return retriever, generator

with st.spinner("Cargando modelo CLIP y base de datos..."):
    retriever, generator = load_pipeline()


# ==========================================
# MODALES (Ventanas Emergentes Full Screen)
# ==========================================

@st.dialog("📊 Análisis de Recuperación Vectorial", width="large")
def show_similarity_dialog(evidences):
    st.write("Este gráfico interactivo muestra el **Nivel de Confianza (Distancia Coseno)** que tuvo el modelo de IA al recuperar cada producto para tu consulta.")
    if not evidences:
        st.warning("No hay evidencias para mostrar.")
        return
    data = []
    for i, ev in enumerate(evidences):
        label = ev['text'][:45] + "..." if len(ev['text']) > 45 else ev['text']
        data.append({"Producto": f"Top {i+1}: {label}", "Confianza (Score)": ev['score']})
    
    df = pd.DataFrame(data).set_index("Producto")
    st.bar_chart(df, height=400, use_container_width=True)


@st.dialog("📈 Dashboard de Evaluación Global", width="large")
def show_global_dashboard():
    eval_path = "data/evaluation/evaluation_results.json"
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        summary = eval_data.get("summary", {})
        
        st.write("### Rendimiento Promedio del Sistema")
        st.write("Resultados calculados sobre el set de evaluación de 20 consultas usando métricas estándar de Information Retrieval.")
        
        metrics_data = []
        for k in eval_data.get("k_values_evaluated", [1,3,5,10]):
            metrics_data.append({
                "Top-K": f"k={k}",
                "Precision": summary.get(f"mean_precision@{k}", 0),
                "Recall": summary.get(f"mean_recall@{k}", 0),
                "NDCG": summary.get(f"mean_ndcg@{k}", 0)
            })
        
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data).set_index("Top-K")
            # Gráfico de líneas
            st.line_chart(df_metrics, height=400, use_container_width=True)
            
            # Tabla de datos
            st.write("#### Tabla de Datos Crudos")
            st.dataframe(df_metrics, use_container_width=True)
    else:
        st.info("No hay métricas disponibles. Ejecuta 'python src/evaluate.py' primero.")


# ==========================================
# INTERFAZ PRINCIPAL Y LAYOUT
# ==========================================

st.title("🤖 Asistente Multimodal de Compras")
st.markdown("Consulta productos por texto o sube una imagen para buscar coincidencias visuales impulsadas por IA.")

# --- Sidebar ---
with st.sidebar:
    st.header("📸 Búsqueda Visual")
    uploaded_file = st.file_uploader("Sube la foto de un producto", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_image, caption="Tu imagen", use_container_width=True)
        search_by_image = st.button("🔎 Buscar productos similares", use_container_width=True)
    else:
        uploaded_image = None
        search_by_image = False
        
    st.markdown("---")
    st.header("🎙️ Búsqueda por Voz")
    st.write("Graba un audio para consultar productos.")
    audio_value = st.audio_input("Grabar consulta")
        
    st.markdown("---")
    st.header("📈 Evaluación del Modelo")
    st.write("Explora el rendimiento matemático oficial del sistema (Precision, Recall, NDCG).")
    
    # Botón para abrir modal global
    if st.button("Ver Dashboard Global", use_container_width=True):
        show_global_dashboard()


# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================

def render_evidences(evidences):
    """Renderiza las evidencias en formato de tarjetas de producto."""
    if not evidences:
        st.info("No se encontró información relevante.")
        return
    
    st.markdown("##### 🛒 Productos Recomendados")
    cols = st.columns(len(evidences))
    for idx, ev in enumerate(evidences):
        with cols[idx]:
            if os.path.exists(ev["local_image_path"]):
                st.image(ev["local_image_path"], use_container_width=True)
            st.markdown(f"**{ev['category']}**")
            st.caption(f"Confianza: {ev['score']:.4f}")
            with st.expander("Ver detalles"):
                st.write(ev["text"])

# ==========================================
# RENDERIZADO DEL HISTORIAL DE CHAT
# ==========================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "user_image" in msg and msg["user_image"] is not None:
            st.image(msg["user_image"], width=150)
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and "evidences" in msg and msg["evidences"]:
            render_evidences(msg["evidences"])
            
            # Botón único por mensaje para abrir su propio modal
            if st.button("📊 Análisis de Similitud", key=f"btn_hist_{msg.get('id_unique', hash(msg['content']))}"):
                show_similarity_dialog(msg["evidences"])


# ==========================================
# FLUJO: BÚSQUEDA POR IMAGEN
# ==========================================
if search_by_image and uploaded_image is not None:
    msg_id = str(time.time())
    st.session_state.messages.append({"role": "user", "content": "🖼️ *Búsqueda por imagen subida*", "user_image": uploaded_image})
    
    # Renderizar query inmediatamente
    with st.chat_message("user"):
        st.image(uploaded_image, width=150)
        st.markdown("🖼️ *Búsqueda por imagen subida*")

    with st.chat_message("assistant"):
        with st.spinner("Analizando imagen..."):
            evidences = retriever.retrieve_by_image(uploaded_image, top_k=3)
        with st.spinner("Generando respuesta inteligente..."):
            answer = generator.generate_response(
                "El usuario subió una imagen de un producto. Describe los productos similares encontrados.",
                evidences, query_type="imagen"
            )
            st.markdown(answer)
            render_evidences(evidences)
            
            # Al final de la generación, se guarda en el historial. 
            # El botón de análisis se mostrará en el siguiente re-render (para evitar bugs de re-ejecución inmediata en Streamlit).
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "evidences": evidences, 
                "id_unique": msg_id
            })
            st.rerun()

# ==========================================
# FLUJO: BÚSQUEDA POR TEXTO
# ==========================================
if prompt := st.chat_input("Ej: ¿Tienes alguna guitarra acústica ideal para principiantes?"):
    msg_id = str(time.time())
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en la base de datos..."):
            evidences = retriever.retrieve(prompt, top_k=3)
        with st.spinner("Generando respuesta inteligente..."):
            answer = generator.generate_response(prompt, evidences, query_type="texto")
            st.markdown(answer)
            render_evidences(evidences)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "evidences": evidences, 
                "id_unique": msg_id
            })
            st.rerun()

# ==========================================
# FLUJO: BÚSQUEDA POR VOZ
# ==========================================
if audio_value is not None:
    audio_bytes = audio_value.getvalue()
    # Evitar re-procesar el mismo audio si Streamlit hace un rerun
    if st.session_state.get("last_audio") != audio_bytes:
        st.session_state["last_audio"] = audio_bytes
        msg_id = str(time.time())
        
        # 1. Transcribir el audio
        with st.spinner("Escuchando tu voz..."):
            transcription = generator.transcribe_audio(audio_bytes)
            
        if transcription.startswith("[Error"):
            st.error(transcription)
        else:
            # 2. Inyectar como prompt de texto
            prompt = f'🎙️ "{transcription}"'
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Buscando en la base de datos..."):
                    evidences = retriever.retrieve(transcription, top_k=3)
                with st.spinner("Generando respuesta inteligente..."):
                    answer = generator.generate_response(transcription, evidences, query_type="texto")
                    st.markdown(answer)
                    render_evidences(evidences)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "evidences": evidences, 
                        "id_unique": msg_id
                    })
                    st.rerun()
