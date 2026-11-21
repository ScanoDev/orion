import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
os.chdir(RAIZ)  # cadena.py busca data/oro/ desde la raíz del repo
sys.path.insert(0, str(RAIZ / "rag"))

import cadena  # noqa: E402

RUTA_FEEDBACK = RAIZ / "rag" / "resultados" / "feedback.jsonl"

st.set_page_config(page_title="ORION · EAFIT")


@st.cache_resource(show_spinner="Cargando el índice de búsqueda...")
def cargar_buscador():
    """Arma el buscador una sola vez, no en cada mensaje."""
    return cadena._buscador()


def formatear_fuentes(fuentes):
    lineas = []
    for f in fuentes:
        titulo = f["titulo"].replace("[", "(").replace("]", ")")
        lineas.append(f"- [{f['n']}] [{titulo}]({f['url']})")
    return "**Fuentes**\n" + "\n".join(lineas)


def mostrar(mensaje):
    # "$" se escapa para que Streamlit no lo tome como fórmula (pasa con costos en pesos)
    st.markdown(mensaje["content"].replace("$", "\\$"))
    if mensaje.get("fuentes"):
        st.markdown(formatear_fuentes(mensaje["fuentes"]))


def guardar_feedback(i):
    """Se ejecuta al hacer clic en los pulgares; agrega una línea al .jsonl."""
    m = st.session_state.mensajes[i]
    voto = st.session_state.get(f"fb_{m['id']}")  # 1 = me gusta, 0 = no me gusta, None = quitó el voto
    registro = {
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sesion": st.session_state.sesion_id,
        "pregunta": m["pregunta"],
        "respuesta": m["content"],
        "chunk_ids": m["chunk_ids"],
        "voto": {1: "me_gusta", 0: "no_me_gusta"}.get(voto),
        "version_cadena": cadena.VERSION_CADENA,
        "modo": os.getenv("ORION_MODO", "bedrock"),
    }
    RUTA_FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_FEEDBACK, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


# --- Barra lateral ---
with st.sidebar:
    st.header("ORION")
    st.caption("Asistente virtual de la Universidad EAFIT")
    if os.getenv("ORION_MODO") == "falso":
        st.warning("Modo de prueba: el modelo es falso y no usa AWS.")
    if st.button("Nueva conversación"):
        st.session_state.mensajes = []
        st.rerun()

cargar_buscador()

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []
if "sesion_id" not in st.session_state:
    st.session_state.sesion_id = uuid.uuid4().hex[:8]

# --- Historial ---
for i, m in enumerate(st.session_state.mensajes):
    with st.chat_message(m["role"]):
        mostrar(m)
        if m["role"] == "assistant":
            st.feedback("thumbs", key=f"fb_{m['id']}", on_change=guardar_feedback, args=(i,))

# --- Nueva pregunta ---
if pregunta := st.chat_input("Pregúntame sobre EAFIT"):
    historial = [{"role": m["role"], "content": m["content"]} for m in st.session_state.mensajes]
    st.session_state.mensajes.append({"id": uuid.uuid4().hex, "role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta.replace("$", "\\$"))

    with st.chat_message("assistant"):
        try:
            with st.spinner("Buscando en el sitio de EAFIT..."):
                r = cadena.responder(pregunta, historial=historial)
        except Exception as e:
            st.error(f"No pude responder: {e}")
            st.stop()

    st.session_state.mensajes.append({
        "id": uuid.uuid4().hex,
        "role": "assistant",
        "content": r["respuesta"],
        "fuentes": r["fuentes"],
        "chunk_ids": r["chunk_ids"],
        "pregunta": pregunta,
    })
    st.rerun()  # redibuja el historial, ahora con la respuesta nueva y sus botones