"""
Parte las páginas de oro en chunks para el RAG de ORION.

Contrato de cada chunk (lo usan la indexación, la cadena y la evaluación):
    page_content: "Título > Sección > Subsección" + línea en blanco + texto
    metadata: chunk_id, pagina_id, url, titulo, categoria, idioma, fecha_extraccion

Uso:
    chunks = fragmentar_documentos(df_oro)   # df_oro: DataFrame de pandas con las columnas de oro
"""
import re

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

TAMANO_CHUNK = 1000   # caracteres; punto de partida, se ajusta con la evaluación
SOLAPE_CHUNK = 150

_por_titulos = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=True,   # el título va en el encabezado del chunk, no repetido en el texto
)
_por_tamano = RecursiveCharacterTextSplitter(chunk_size=TAMANO_CHUNK, chunk_overlap=SOLAPE_CHUNK)


def _limpiar_titulo(texto):
    """'**Requisitos**' -> 'Requisitos'."""
    return re.sub(r"[*_`]", "", texto or "").strip()


def fragmentar_pagina(fila):
    """Convierte una página de oro (una fila) en una lista de Document."""
    titulo = _limpiar_titulo(fila["titulo"]) or fila["url"]
    chunks = []
    for seccion in _por_titulos.split_text(fila["texto"]):
        ruta = [titulo]
        for nivel in ("h1", "h2", "h3"):
            nombre = _limpiar_titulo(seccion.metadata.get(nivel))
            if nombre and nombre.lower() != ruta[-1].lower():
                ruta.append(nombre)
        encabezado = " > ".join(ruta)
        for pedazo in _por_tamano.split_text(seccion.page_content):
            chunks.append((encabezado, pedazo))

    return [
        Document(
            page_content=f"{encabezado}\n\n{pedazo}",
            metadata={
                "chunk_id": f"{fila['pagina_id']}_{i:03d}",
                "pagina_id": fila["pagina_id"],
                "url": fila["url"],
                "titulo": titulo,
                "categoria": fila["categoria"],
                "idioma": fila["idioma"],
                "fecha_extraccion": fila["fecha_extraccion"] or "",
            },
        )
        for i, (encabezado, pedazo) in enumerate(chunks)
    ]


def fragmentar_documentos(df_oro):
    chunks = []
    for fila in df_oro.to_dict("records"):
        chunks.extend(fragmentar_pagina(fila))
    return chunks