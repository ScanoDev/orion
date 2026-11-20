"""
Capa oro de ORION: documentos limpios con metadatos para el RAG.

Entrada: tabla plata_documentos
    pagina_id, url, titulo, idioma, status_pagina, fecha_extraccion, texto, n_caracteres

Salida: una fila por página con
    pagina_id, url, titulo, idioma, seccion, categoria, fecha_extraccion, texto, n_caracteres

Uso:
    oro = enriquecer_documentos(plata)

El chunking NO se hace aquí: va en la indexación (rag/), para poder probar
tamaños de chunk en segundos sin volver a correr Glue.
"""
from pyspark.sql import functions as F

# Primera parte de la URL -> categoría. Calibrado con medallion/athena/03_explorar_para_oro.sql
CATEGORIAS = {
    "noticias": "noticias",
    "sistema-ciencia-tecnologia-innovacion": "investigacion",
    "centros-estudio-incidencia": "investigacion",
    "universidad-parque": "investigacion",
    "calendario-academico": "calendario",
    "calendario-eventos": "calendario",
    "plan-de-estudio": "programas",
    "posgrados": "programas",
    "otros-programas": "programas",
    "idiomas": "programas",
    "becas-y-financiacion": "becas_y_tramites",
    "servicios": "becas_y_tramites",
    "tramites-servicios": "becas_y_tramites",
    "bienestar-universitario": "bienestar",
    "centro-cultural-biblioteca-luis-echavarria-villegas": "cultura",
    "orquesta-sinfonica-eafit": "cultura",
    "institucional": "institucional",
    "calidad-eafit": "institucional",
    "informes-de-gestion-y-sostenibilidad": "institucional",
    "informe-de-sostenibilidad-2024": "institucional",
    "internacionalizacion": "institucional",
    "filantropia": "institucional",
    "sostenibilidad-ambiental": "institucional",
    "eafit-y-el-pacto-global": "institucional",
}

RE_INVISIBLES = "[\u200b\u200c\u200d\ufeff\u00a0]"   # espacios invisibles que trae el sitio en los títulos
RE_SUFIJO = r"\s*[-|]\s*(Universidad EAFIT|EAFIT University|EAFIT)\s*$"


def limpiar_titulo(col):
    """'​E​AFIT en 2024​ - Universidad EAFIT' -> 'EAFIT en 2024'."""
    sin_invisibles = F.regexp_replace(col, RE_INVISIBLES, "")
    sin_sufijo = F.regexp_replace(sin_invisibles, RE_SUFIJO, "")
    return F.trim(F.regexp_replace(sin_sufijo, r"\s{2,}", " "))


def enriquecer_documentos(plata):
    partes = F.split(F.regexp_replace("url", r"[?#].*$", ""), "/")
    primera = F.coalesce(F.try_element_at(partes, F.lit(4)), F.lit(""))
    segunda = F.coalesce(F.try_element_at(partes, F.lit(5)), F.lit(""))
    # En las páginas /en/... la sección real es la segunda parte
    seccion = F.when(primera == "en", segunda).otherwise(primera)

    mapa = F.create_map(*[F.lit(x) for par in CATEGORIAS.items() for x in par])
    categoria = (
        F.when(F.col("seccion").startswith("escuela-"), F.lit("programas"))
        .otherwise(F.coalesce(F.try_element_at(mapa, F.col("seccion")), F.lit("otros")))
    )

    titulo = limpiar_titulo(F.col("titulo"))
    return (
        plata.withColumn("seccion", seccion)
        .withColumn("categoria", categoria)
        .withColumn("titulo", F.when(F.length(titulo) > 0, titulo).otherwise(F.col("titulo")))
        .select(
            "pagina_id", "url", "titulo", "idioma", "seccion", "categoria",
            "fecha_extraccion", "texto", "n_caracteres",
        )
    )