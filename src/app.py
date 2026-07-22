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
    
    /* Eliminar espacio blanco superior */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Estilizar las tarjetas de productos (evidencias) */
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
    
    /* Eliminar sidebar por completo */
    [data-testid="collapsedControl"] {
        display: none;
    }
    section[data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    from src.retrieval import MultimodalRetriever
    from src.generation import RAGGenerator
    retriever = MultimodalRetriever()
    generator = RAGGenerator(temperature=0.7)
    return retriever, generator

with st.spinner("Cargando modelo CLIP y base de datos..."):
    retriever, generator = load_pipeline()

# Cargar el store de Relevance Feedback (persistente en JSON)
from src.relevance_feedback import RelevanceFeedbackStore
if "feedback_store" not in st.session_state:
    st.session_state.feedback_store = RelevanceFeedbackStore()
feedback_store = st.session_state.feedback_store

# Importar el formateador de historial para memoria conversacional
from src.generation import RAGGenerator as _RAGGen


# ==========================================
# FUNCIONES DE RENDERIZADO DE GRÁFICAS
# ==========================================

def render_similarity_chart(evidences):
    st.write("Este gráfico muestra el **Nivel de Confianza (Score Final)** que tuvo el modelo de IA al recuperar cada producto.")
    if not evidences:
        st.warning("No hay evidencias para mostrar.")
        return
    data = []
    for i, ev in enumerate(evidences):
        label = ev['text'][:45] + "..." if len(ev['text']) > 45 else ev['text']
        data.append({"Producto": f"Top {i+1}: {label}", "Confianza (Score)": ev['score']})
    
    df = pd.DataFrame(data).set_index("Producto")
    st.bar_chart(df, height=400, use_container_width=True)


def render_global_dashboard():
    eval_path = "data/evaluation/evaluation_results.json"
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        summary = eval_data.get("summary", {})
        
        st.write("Resultados calculados sobre el set de evaluación (qrels) usando métricas estándar de Information Retrieval.")
        
        metrics_data = []
        for k in eval_data.get("k_values_evaluated", [1,3,5,10]):
            metrics_data.append({
                "K": k,
                "Precision": summary.get(f"mean_precision@{k}", 0),
                "Recall": summary.get(f"mean_recall@{k}", 0),
                "NDCG": summary.get(f"mean_ndcg@{k}", 0)
            })
        
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data).set_index("K")
            # Gráfico de líneas
            st.line_chart(df_metrics, height=400, use_container_width=True)
            
            # Tabla de datos
            with st.expander("Ver Tabla de Datos Crudos"):
                st.dataframe(df_metrics, use_container_width=True)
    else:
        st.info("No hay métricas disponibles. Ejecuta 'python src/evaluate.py' primero.")


# ==========================================
# INTERFAZ PRINCIPAL Y LAYOUT
# ==========================================

st.title("🤖 Asistente Multimodal de Compras")

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

def render_evidences(evidences, msg_key=""):
    """Renderiza las evidencias en formato de tarjetas de producto con botones de feedback."""
    if not evidences:
        st.info("No se encontró información relevante.")
        return
    
    # Determinar si los resultados fueron re-rankeados o expandidos
    was_reranked = any(ev.get("reranked", False) for ev in evidences)
    if was_reranked:
        expanded_froms = list(dict.fromkeys([
            ev.get("expanded_from", "") for ev in evidences if ev.get("expanded_from")
        ]))
        if expanded_froms:
            with st.expander("🔎 Ver consultas expandidas utilizadas", expanded=False):
                st.write("El sistema generó estas reformulaciones automáticas para mejorar la búsqueda:")
                for i, q in enumerate(expanded_froms):
                    icon = "🔵" if i == 0 else "🟢"
                    st.write(f"{icon} `{q}`")
        else:
            st.markdown("✨ **Re-ranking activo:** Resultados refinados por Cross-Encoder.")

    def render_single_evidence(ev, idx):
        # Reemplazamos \ por / (para arreglar el error en Linux) y luego normpath ajusta a la ruta nativa del OS actual
        normalized_path = ev["local_image_path"].replace("\\", "/")
        img_path = os.path.normpath(normalized_path)
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        st.markdown(f"**{ev['category']}**")
        # Mostrar score del re-ranker si existe, si no el score de CLIP
        score_text = ""
        if ev.get("reranked") and "rerank_score" in ev:
            score_text = f"✨ Re-rank: {ev['rerank_score']:.2f} | CLIP: {ev['score']:.4f}"
        else:
            score_text = f"Confianza: {ev['score']:.4f}"
        
        # Mostrar boost de feedback si existe y no es neutro
        boost = ev.get("feedback_boost", 1.0)
        if boost != 1.0:
            score_text += f" | Boost: {boost:.2f}"
        st.caption(score_text)
        
        with st.expander("Ver detalles"):
            st.write(ev["text"])
        
        # Botones de feedback (👍 / 👎)
        fb_key = f"fb_{msg_key}_{ev['id']}_{idx}"
        col_like, col_dislike = st.columns(2)
        with col_like:
            if st.button("👍", key=f"like_{fb_key}", use_container_width=True):
                feedback_store.add_feedback(ev["id"], is_relevant=True)
                st.toast(f"✅ ¡Feedback registrado para mejorar búsquedas futuras!")
        with col_dislike:
            if st.button("👎", key=f"dislike_{fb_key}", use_container_width=True):
                feedback_store.add_feedback(ev["id"], is_relevant=False)
                st.toast(f"📝 ¡Feedback registrado! Este producto será penalizado en futuras búsquedas.")

    st.markdown("##### 🛒 Top 3 Productos Recomendados")
    top_n = min(3, len(evidences))
    cols = st.columns(top_n)
    for idx in range(top_n):
        with cols[idx]:
            render_single_evidence(evidences[idx], idx)
            
    if len(evidences) > 3:
        with st.expander("📦 Ver más resultados recomendados"):
            rest_evidences = evidences[3:]
            for row_start in range(0, len(rest_evidences), 3):
                row_evs = rest_evidences[row_start:row_start+3]
                cols_rest = st.columns(len(row_evs))
                for c_idx, ev in enumerate(row_evs):
                    idx = 3 + row_start + c_idx
                    with cols_rest[c_idx]:
                        render_single_evidence(ev, idx)

# Crear los módulos principales (Tabs)
tab_chat, tab_analysis = st.tabs(["💬 Chat Multimodal", "📈 Módulo de Análisis de Recuperación"])

with tab_chat:
    # ==========================================
    # RENDERIZADO DEL HISTORIAL DE CHAT
    # ==========================================
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if "user_image" in msg and msg["user_image"] is not None:
                st.image(msg["user_image"], width=150)
            st.markdown(msg["content"])
            
            if msg["role"] == "assistant" and "evidences" in msg and msg["evidences"]:
                render_evidences(msg["evidences"], msg_key=f"hist_{msg_idx}")
    
    # ==========================================
    # SELECTOR DE MODO DE BÚSQUEDA
    # ==========================================
    st.markdown("---")
    input_mode = st.radio(
        "Modo de búsqueda:", 
        ["⌨️ Texto", "🎙️ Voz", "📸 Imagen"], 
        horizontal=True,
        label_visibility="collapsed"
    )

    # ==========================================
    # FLUJOS CONDICIONALES DE BÚSQUEDA
    # ==========================================
    
    if input_mode == "📸 Imagen":
        uploaded_file = st.file_uploader("Sube la foto del producto", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file).convert("RGB")
            st.image(uploaded_image, width=150)
            if st.button("🔎 Buscar productos similares", use_container_width=True):
                msg_id = str(time.time())
                st.session_state.messages.append({"role": "user", "content": "🖼️ *Búsqueda por imagen subida*", "user_image": uploaded_image})
                
                with st.chat_message("user"):
                    st.image(uploaded_image, width=150)
                    st.markdown("🖼️ *Búsqueda por imagen subida*")

                with st.chat_message("assistant"):
                    with st.spinner("Analizando imagen..."):
                        evidences = retriever.retrieve_by_image(uploaded_image, top_k=6)
                        evidences = feedback_store.apply_feedback_to_results(evidences)
                    with st.spinner("Generando respuesta inteligente..."):
                        history = _RAGGen.format_chat_history(st.session_state.messages)
                        answer = generator.generate_response(
                            "El usuario subió una imagen de un producto. Describe los productos similares encontrados.",
                            evidences, query_type="imagen", chat_history=history
                        )
                        st.markdown(answer)
                        render_evidences(evidences, msg_key=f"img_{msg_id}")
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer, 
                            "evidences": evidences, 
                            "id_unique": msg_id
                        })
                        st.rerun()

    elif input_mode == "🎙️ Voz":
        audio_value = st.audio_input("🎙️ Grabar nota de voz")
        if audio_value is not None:
            audio_bytes = audio_value.getvalue()
            # Evitar re-ejecución infinita
            if st.session_state.get("last_audio") != audio_bytes:
                st.session_state["last_audio"] = audio_bytes
                msg_id = str(time.time())
                
                with st.spinner("Escuchando tu voz..."):
                    transcription = generator.transcribe_audio(audio_bytes)
                    
                if transcription.startswith("[Error"):
                    st.error(transcription)
                else:
                    prompt_text = f'🎙️ "{transcription}"'
                    st.session_state.messages.append({"role": "user", "content": prompt_text})
                    
                    with st.chat_message("user"):
                        st.markdown(prompt_text)
                        
                    with st.chat_message("assistant"):
                        with st.spinner("💡 Expandiendo consulta + Re-ranking..."):
                            evidences = retriever.retrieve_with_expansion(
                                transcription, top_k=6, candidate_k=15, n_expansions=3
                            )
                            evidences = feedback_store.apply_feedback_to_results(evidences)
                        with st.spinner("Generando respuesta inteligente..."):
                            history = _RAGGen.format_chat_history(st.session_state.messages)
                            answer = generator.generate_response(transcription, evidences, query_type="texto", chat_history=history)
                            st.markdown(answer)
                            render_evidences(evidences, msg_key=f"voz_{msg_id}")
                            
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": answer, 
                                "evidences": evidences, 
                                "id_unique": msg_id
                            })
                            st.rerun()

    else: # input_mode == "⌨️ Texto"
        if prompt := st.chat_input("¿ Tienes alguna guitarra acústica ideal para principiantes?"):
            msg_id = str(time.time())
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("💡 Expandiendo consulta con IA..."):
                    # Pipeline de 3 etapas: Expansion + CLIP + Cross-Encoder
                    evidences = retriever.retrieve_with_expansion(
                        prompt, top_k=6, candidate_k=15, n_expansions=3
                    )
                    # Aplicar Relevance Feedback (ajustar scores según historial)
                    evidences = feedback_store.apply_feedback_to_results(evidences)
                with st.spinner("Generando respuesta inteligente..."):
                    history = _RAGGen.format_chat_history(st.session_state.messages)
                    answer = generator.generate_response(prompt, evidences, query_type="texto", chat_history=history)
                    st.markdown(answer)
                    render_evidences(evidences, msg_key=f"new_{msg_id}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "evidences": evidences, 
                        "id_unique": msg_id
                    })
                    st.rerun()

with tab_analysis:
    st.header("Análisis de Recuperación (Módulo unificado)")
    st.write("Visualiza el desempeño global y examina cómo operan los algoritmos extra en cada consulta.")
    
    # 1. Rendimiento Global y Feedback Stats
    col_dash, col_feed = st.columns(2)
    with col_dash:
        st.subheader("📈 Rendimiento Global (vs qrels)")
        render_global_dashboard()
    
    with col_feed:
        st.subheader("🗳️ Estadísticas de Relevance Feedback")
        stats = feedback_store.get_stats()
        if stats["total_interactions"] > 0:
            st.metric("Documentos calificados", stats["total_documents_rated"])
            col_f1, col_f2 = st.columns(2)
            col_f1.metric("👍 Total Likes", stats["total_likes"])
            col_f2.metric("👎 Total Dislikes", stats["total_dislikes"])
            st.metric("Tasa de satisfacción", f"{stats['satisfaction_rate']:.0%}")
            st.info("Estos datos ajustan automáticamente los scores usando el algoritmo Rocchio (α=0.3).")
        else:
            st.info("Aún no hay feedback de usuarios. Usa los botones 👍/👎 en los productos recomendados para mejorar el ranking futuro.")

    st.markdown("---")
    
    # 2. Inspector por Consulta (Mostrando los 4 Extras)
    st.subheader("🔍 Inspector de Búsquedas (Funcionalidades de Excelencia)")
    
    queries_with_results = []
    for i in range(len(st.session_state.messages)-1):
        if st.session_state.messages[i]["role"] == "user" and st.session_state.messages[i+1]["role"] == "assistant" and "evidences" in st.session_state.messages[i+1]:
            queries_with_results.append({
                "idx": i,
                "user_content": st.session_state.messages[i].get("content", "Búsqueda por imagen"),
                "evidences": st.session_state.messages[i+1]["evidences"]
            })
            
    if queries_with_results:
        query_options = {q["idx"]: f"Consulta {idx+1}: {q['user_content'][:50]}..." for idx, q in enumerate(queries_with_results)}
        selected_idx = st.selectbox("Selecciona una consulta del historial para auditar:", options=list(query_options.keys()), format_func=lambda x: query_options[x])
        
        selected_query = next(q for q in queries_with_results if q["idx"] == selected_idx)
        evs = selected_query["evidences"]
        
        st.markdown(f"**Consulta original:** `{selected_query['user_content']}`")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### 1. Query Expansion (Gemini)")
            expanded_froms = list(dict.fromkeys([ev.get("expanded_from", "") for ev in evs if ev.get("expanded_from")]))
            if expanded_froms:
                st.success("✅ **Activo:** Gemini generó reformulaciones automáticas.")
                for q in expanded_froms:
                    st.write(f"- `{q}`")
            else:
                st.info("No se aplicó expansión de consulta en esta búsqueda (p.ej. fue búsqueda por imagen).")
                
            st.markdown("#### 2. Re-ranking (Cross-Encoder)")
            was_reranked = any(ev.get("reranked", False) for ev in evs)
            if was_reranked:
                st.success("✅ **Activo:** `ms-marco-MiniLM-L-6-v2` re-ordenó el top-20 inicial al top-3 final.")
                data = [{"Producto": ev['text'][:30], "CLIP (Recall)": round(ev['score'], 4), "Cross-Encoder": round(ev.get('rerank_score', 0), 4)} for ev in evs]
                st.dataframe(data, use_container_width=True)
            else:
                st.info("No se aplicó Re-ranking (solo ocurre en flujos de texto/voz).")

        with c2:
            st.markdown("#### 3. Relevance Feedback (Rocchio)")
            has_boost = any(ev.get("feedback_boost", 1.0) != 1.0 for ev in evs)
            if has_boost:
                st.success("✅ **Activo:** Algunos productos sufrieron ajustes por votos previos.")
                for ev in evs:
                    boost = ev.get("feedback_boost", 1.0)
                    if boost != 1.0:
                        st.write(f"- **{ev['text'][:20]}...**: Multiplicador = `{boost:.2f}x`")
            else:
                st.info("Ninguno de los productos recuperados tenía historial de feedback (boost neutral = 1.0x).")
                
            st.markdown("#### 4. Memoria Conversacional")
            st.success("✅ **Activo:** Se inyectaron los últimos turnos del chat en la variable `{chat_history}` del prompt de Gemini para resolver referencias cruzadas (Context-Aware).")
            
        st.markdown("#### 📊 Gráfica de Confianza Final")
        render_similarity_chart(evs)
                
    else:
        st.info("Realiza una búsqueda en el chat para inspeccionar cómo actúan los 4 algoritmos de excelencia (Query Expansion, Re-ranking, Feedback y Memoria).")
