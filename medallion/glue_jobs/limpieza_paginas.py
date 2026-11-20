"""
Reglas de limpieza de ORION para la capa plata.

Vienen de las reglas R1, R2 y R3 de los talleres, pero aplicadas a la
página completa y por línea (no por token), para conservar los saltos de
línea y los títulos del markdown, que sirven después para partir los chunks.

Entrada: DataFrame de páginas con las columnas
    pagina_id, url, titulo, idioma, status_pagina, fecha_extraccion, markdown

Uso:
    documentos, paginas_rechazadas, conteos = limpiar_paginas(df_paginas)

Reglas por página:
    R1: la página respondió con un código distinto de 200 o sin markdown.
    R4: después de limpiar quedan menos de MIN_CARACTERES caracteres.
    R5: el texto limpio es idéntico al de otra página (se conserva la de url menor).
Reglas por línea (se quitan del texto, la página sigue):
    R2: la línea se repite en muchas páginas (menús, pies, widgets de noticias).
        Se conserva en una sola página (la de pagina_id menor) y se quita de
        las demás, para no perder datos útiles como los teléfonos de las sedes.
        Las líneas cortas (etiquetas y valores como "### Modalidad" o
        "Presencial") tienen un umbral más alto, para no borrar los datos
        de las fichas de eventos y programas.
    R3: la línea no tiene letras ni números después de limpiar el markdown,
        o es solo un número o un código de color (#7b9180).

Versión 3: umbrales calibrados con las consultas de
medallion/athena/01_validar_plata.sql (sobre la versión 1) y
medallion/athena/02_validar_plata_v2.sql (sobre la versión 2).
"""
from pyspark.sql import functions as F
from pyspark.sql import Window

MOTIVO_R1 = "R1: página con error HTTP o sin contenido"
MOTIVO_R2 = "R2: línea repetida (menú o pie de página)"
MOTIVO_R3 = "R3: línea sin contenido o solo un código"
MOTIVO_R4 = "R4: página con muy poco texto"
MOTIVO_R5 = "R5: página duplicada"

UMBRAL_PAGINAS = 0.033  # unas 100 de 3.030 páginas: todo lo que se repite tanto es plantilla
UMBRAL_CORTAS = 0.068   # unas 206 de 3.030: por encima de "Presencial" (200) y por debajo de los menús (212 o más)
MAX_PALABRAS_CORTA = 5  # hasta 5 palabras cuenta como línea corta (etiqueta o valor)
MIN_CARACTERES = 200    # menos que esto no alcanza para responder nada

# Expresiones del markdown de Firecrawl
RE_IMAGEN = r"!\[[^\]]*\]\([^)]*\)"            # ![alt](url): se quita
RE_ENLACE = r"\[([^\]]*)\]\([^)]*\)"           # [texto](url): queda solo el texto
RE_URL = r"https?://\S+"                       # url suelta: se quita
RE_CONTENIDO = r"[\p{L}\p{N}]"                 # al menos una letra o número
RE_SOLO_CODIGO = r"^(\d+|#[0-9a-fA-F]{3,8})$"   # "1311" o "#7b9180": basura de widgets


ESQUEMA_RAW = (
    "url_pedida STRING, fecha_extraccion STRING, http_status INT, "
    "respuesta STRUCT<success: BOOLEAN, data: STRUCT<markdown: STRING, "
    "metadata: STRUCT<title: STRING, language: STRING, statusCode: INT>>>"
)


def leer_bronce(spark, ruta):
    """
    Lee los JSON de Firecrawl (uno por página) como texto completo y saca solo
    los campos que usa ORION. Se leen como texto y con esquema fijo porque la
    metadata de Firecrawl trae campos distintos en cada página.
    """
    crudo = spark.read.option("wholetext", True).text(ruta)
    j = F.from_json("value", ESQUEMA_RAW).alias("j")
    return crudo.select(j).select(
        F.md5(F.col("j.url_pedida")).alias("pagina_id"),
        F.col("j.url_pedida").alias("url"),
        F.col("j.respuesta.data.metadata.title").alias("titulo"),
        F.col("j.respuesta.data.metadata.language").alias("idioma"),
        F.col("j.respuesta.data.metadata.statusCode").alias("status_pagina"),
        F.col("j.fecha_extraccion").alias("fecha_extraccion"),
        F.col("j.respuesta.data.markdown").alias("markdown"),
    )


def limpiar_linea(col):
    """Quita imágenes y urls, deja el texto de los enlaces y recorta espacios."""
    sin_imagen = F.regexp_replace(col, RE_IMAGEN, "")
    sin_enlace = F.regexp_replace(sin_imagen, RE_ENLACE, "$1")
    sin_url = F.regexp_replace(sin_enlace, RE_URL, "")
    return F.trim(F.regexp_replace(sin_url, r"\s{2,}", " "))


def limpiar_paginas(df):
    # R1: páginas con error o sin markdown
    es_r1 = F.coalesce(
        (F.col("status_pagina").cast("string") != F.lit("200"))
        | (F.length(F.trim(F.col("markdown"))) == 0),
        F.lit(True),
    )
    marcadas = df.withColumn("es_r1", es_r1)
    rech_r1 = marcadas.filter("es_r1").withColumn("motivo_rechazo", F.lit(MOTIVO_R1))
    validas = marcadas.filter("not es_r1").drop("es_r1")
    n_validas = validas.count()

    # Una fila por línea no vacía
    lineas = (
        validas.select("pagina_id", F.posexplode(F.split("markdown", "\n")).alias("n_linea", "linea"))
        .withColumn("linea_limpia", limpiar_linea(F.col("linea")))
        .filter(F.length(F.trim("linea")) > 0)
    )

    # R2: líneas que se repiten en muchas páginas
    frecuencia = lineas.groupBy("linea_limpia").agg(
        F.countDistinct("pagina_id").alias("paginas_con_linea"),
        F.min("pagina_id").alias("pagina_que_la_conserva"),
    )
    lineas = lineas.join(frecuencia, "linea_limpia", "left")
    n_palabras = F.size(F.split(F.trim(F.regexp_replace("linea_limpia", r"[#*_\\]", "")), r"\s+"))
    es_corta = n_palabras <= MAX_PALABRAS_CORTA
    umbral = F.when(es_corta, F.lit(UMBRAL_CORTAS * n_validas)).otherwise(F.lit(UMBRAL_PAGINAS * n_validas))
    es_r2 = (F.col("paginas_con_linea") >= umbral) & (F.col("pagina_id") != F.col("pagina_que_la_conserva"))
    es_r3 = ~F.col("linea_limpia").rlike(RE_CONTENIDO) | F.col("linea_limpia").rlike(RE_SOLO_CODIGO)
    lineas = lineas.withColumn(
        "motivo_linea",
        F.when(es_r2, F.lit(MOTIVO_R2)).when(es_r3, F.lit(MOTIVO_R3)),
    )

    conteo_lineas = {
        fila["motivo"]: fila["n"]
        for fila in lineas.groupBy(F.coalesce("motivo_linea", F.lit("conservadas")).alias("motivo"))
        .agg(F.count("*").alias("n")).collect()
    }

    # Rearmar el texto de cada página con las líneas que quedan, en su orden
    texto = (
        lineas.filter(F.col("motivo_linea").isNull())
        .groupBy("pagina_id")
        .agg(F.array_sort(F.collect_list(F.struct("n_linea", "linea_limpia"))).alias("partes"))
        .select("pagina_id", F.concat_ws("\n", F.col("partes.linea_limpia")).alias("texto"))
    )
    paginas = (
        validas.drop("markdown")
        .join(texto, "pagina_id", "left")
        .withColumn("texto", F.coalesce("texto", F.lit("")))
        .withColumn("n_caracteres", F.length("texto"))
    )

    # R4: muy poco texto. R5: texto idéntico a otra página.
    orden = Window.partitionBy("huella").orderBy("url")
    paginas = (
        paginas.withColumn("huella", F.sha2("texto", 256))
        .withColumn("puesto", F.row_number().over(orden))
        .withColumn(
            "motivo_rechazo",
            F.when(F.col("n_caracteres") < MIN_CARACTERES, F.lit(MOTIVO_R4))
            .when(F.col("puesto") > 1, F.lit(MOTIVO_R5)),
        )
        .drop("huella", "puesto")
    )

    documentos = paginas.filter(F.col("motivo_rechazo").isNull()).drop("motivo_rechazo")
    columnas_rechazo = ["pagina_id", "url", "titulo", "status_pagina", "motivo_rechazo"]
    paginas_rechazadas = rech_r1.select(columnas_rechazo).unionByName(
        paginas.filter(F.col("motivo_rechazo").isNotNull()).select(columnas_rechazo)
    )

    conteo_paginas = {
        fila["motivo_rechazo"]: fila["n"]
        for fila in paginas_rechazadas.groupBy("motivo_rechazo").agg(F.count("*").alias("n")).collect()
    }
    n_entrada = df.count()
    n_documentos = documentos.count()
    total_lineas = sum(conteo_lineas.values())

    conteos = (
        [("paginas_entrada", n_entrada), ("paginas_validas", n_documentos)]
        + sorted(conteo_paginas.items())
        + [("paginas_diferencia_debe_ser_0", n_entrada - n_documentos - sum(conteo_paginas.values()))]
        + [("lineas_de_paginas_sin_r1", total_lineas)]
        + sorted(conteo_lineas.items())
    )
    return documentos, paginas_rechazadas, conteos


#aws s3 cp medallion\glue_jobs\limpieza_paginas.py s3://orion-eafit-2026/codigo/limpieza_paginas.py