"""
Agrega al índice un catálogo de programas de EAFIT, para las preguntas de lista
("dame 5 maestrías relacionadas con computación", "¿qué doctorados hay?").

Problema: cada programa tiene su propia página y ORION solo le pasa 5 chunks al modelo,
así que una pregunta que pide una lista nunca ve todos los programas.
Solución: chunks extra (categoria "catalogo"), uno por nivel y escuela, con la lista de
programas y su URL. Salen de las páginas del corpus, no se inventa nada.

Qué cuenta como página de un programa:
    /posgrados/<escuela>/<programa>   (las de /en/ no, porque son las mismas en inglés)
El nivel sale del nombre en la URL: maestria, especializacion o doctorado.
Los pregrados no van: en el corpus casi no tienen páginas con esa forma, y un catálogo
incompleto haría que ORION dijera que hay 1 pregrado.
El texto no da totales ni dice "estos son todos", porque el corpus puede no tener todos.

Ficha de cada programa: la página de cada posgrado tiene unas tarjetas arriba (Duración,
Lugar, Modalidad, Idioma, Valor del primer semestre). En el corpus esas tarjetas quedan como
pares de líneas ("Duración" / "3 semestres") en uno de los primeros chunks, pero compiten con
chunks que repiten "Semestre 1, Semestre 2" y el modelo terminaba contando semestres.
Por eso el catálogo también lleva la ficha de cada programa. Solo se leen los primeros
chunks de la página, porque más abajo aparecen las tarjetas de OTROS programas relacionados.

Uso (desde la raíz del repo):
    python rag/catalogo.py v3            # solo muestra el catálogo
    python rag/catalogo.py v3 --agregar  # lo agrega a vectorstore/orion_v3 (reemplaza uno anterior)
    python rag/publicar.py v3            # después, para que el equipo lo baje
"""
import argparse
import re
import sys
import unicodedata
from pathlib import Path

import chromadb
import pandas as pd
from botocore.config import Config
from langchain_aws import BedrockEmbeddings

sys.stdout.reconfigure(encoding="utf-8")

BASE = "https://www.eafit.edu.co"
CATEGORIA = "catalogo"
RE_PROGRAMA = re.compile(r"^/(posgrados)/(escuela-[^/]+)/([^/?#]+)/?$")
ESCUELAS = {
    "escuela-administracion": "Escuela de Administración",
    "escuela-artes-humanidades": "Escuela de Artes y Humanidades",
    "escuela-ciencias-aplicadas-ingenieria": "Escuela de Ciencias Aplicadas e Ingeniería",
    "escuela-derecho": "Escuela de Derecho",
    "escuela-finanzas-economia-gobierno": "Escuela de Finanzas, Economía y Gobierno",
}
CAMPOS_FICHA = {   # etiqueta normalizada en la página -> nombre en el catálogo
    "duracion": "Duración",
    "lugar": "Lugar",
    "modalidad": "Modalidad",
    "idioma": "Idioma",
    "valor del primer semestre": "Valor del primer semestre",
    # páginas que quedaron en inglés
    "duration": "Duración",
    "location": "Lugar",
    "place": "Lugar",
    "modality": "Modalidad",
    "language": "Idioma",
}
# Si una tarjeta está vacía, la línea siguiente es otra etiqueta (pasaba "Modalidad: SNIES")
OTRAS_ETIQUETAS = {"snies", "horario", "schedule", "ver resolucion", "conoce otras tarifas",
                   "ver mas", "mas informacion", "registro calificado", "acreditacion"}
MAX_CHUNK_FICHA = 6   # la ficha propia está arriba; más abajo vienen las de otros programas
NIVELES = [  # (prefijo del nombre en la URL, nombre del nivel)
    ("maestria", "Maestrías"),
    ("especializacion", "Especializaciones"),
    ("doctorado", "Doctorados"),
]


def nivel_de(tipo, slug):
    for prefijo, nombre in NIVELES:
        if slug.startswith(prefijo):
            return nombre
    return "Otros posgrados"


def nombre_escuela(slug):
    return ESCUELAS.get(slug, slug.replace("escuela-", "Escuela de ").replace("-", " ").title())


def _normal(texto):
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c)).strip(" #*:")


def fichas(chunks):
    """url -> {"Duración": "3 semestres", ...}, leído de los primeros chunks de cada página."""
    resultado = {}
    numero = chunks["chunk_id"].str.extract(r"_(\d+)$")[0].astype(float)
    primeros = chunks[numero <= MAX_CHUNK_FICHA].assign(n=numero).sort_values(["url", "n"])
    for url, grupo in primeros.groupby("url"):
        lineas = [l.strip() for texto in grupo["page_content"] for l in texto.splitlines() if l.strip()]
        ficha = {}
        for i, linea in enumerate(lineas[:-1]):
            campo = CAMPOS_FICHA.get(_normal(linea))
            valor = lineas[i + 1].strip(" #*")
            etiqueta = _normal(valor) in CAMPOS_FICHA or _normal(valor) in OTRAS_ETIQUETAS
            if campo and campo not in ficha and len(valor) <= 60 and not etiqueta:
                ficha[campo] = valor
        if ficha:
            resultado[url] = ficha
    return resultado


def programas(chunks):
    """Una fila por página de programa: nivel, escuela, nombre y URL."""
    sin_catalogo = chunks[chunks["categoria"] != CATEGORIA]
    ficha_de = fichas(sin_catalogo)
    paginas = sin_catalogo.drop_duplicates("url")
    filas = []
    for url, titulo in zip(paginas["url"], paginas["titulo"]):
        ruta = url.replace(BASE, "").split("?")[0]
        m = RE_PROGRAMA.match(ruta)
        if not m:
            continue
        tipo, escuela, slug = m.groups()
        filas.append({
            "nivel": nivel_de(tipo, slug),
            "escuela": nombre_escuela(escuela),
            "programa": str(titulo).strip(),
            "url": url,
            "url_catalogo": f"{BASE}/{tipo}",
            "ficha": ficha_de.get(url, {}),
        })
    return (pd.DataFrame(filas).drop_duplicates(["nivel", "programa"])
            .sort_values(["nivel", "escuela", "programa"]).reset_index(drop=True))


def chunks_catalogo(df):
    """Un chunk por nivel y escuela, con el mismo formato de los demás ('Título > Sección' + texto)."""
    salida = []
    for (nivel, escuela), grupo in df.groupby(["nivel", "escuela"]):
        lineas = "\n".join(
            f"- {p} ({u})" + "".join(f". {campo}: {valor}" for campo, valor in f.items())
            for p, u, f in zip(grupo["programa"], grupo["url"], grupo["ficha"])
        )
        texto = (
            f"Catálogo de programas de EAFIT > {nivel} > {escuela}\n\n"
            f"{nivel} de la {escuela} de la Universidad EAFIT que tienen página en el sitio web:\n{lineas}"
        )
        sin_tildes = unicodedata.normalize("NFKD", f"{nivel}_{escuela}".lower()).encode("ascii", "ignore").decode()
        clave = re.sub(r"[^a-z0-9]+", "_", sin_tildes).strip("_")
        salida.append({
            "id": f"catalogo_{clave}",
            "texto": texto,
            "metadata": {
                "chunk_id": f"catalogo_{clave}",
                "pagina_id": "catalogo",
                "url": grupo["url_catalogo"].iloc[0],
                "titulo": f"Catálogo de {nivel.lower()} de EAFIT",
                "categoria": CATEGORIA,
                "idioma": "es",
                "fecha_extraccion": "",
            },
        })
    return salida


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("version", nargs="?", default="v3")
    parser.add_argument("--agregar", action="store_true", help="Agrega el catálogo al índice de Chroma")
    args = parser.parse_args()

    ruta_chunks = Path("data/processed") / args.version / "chunks.parquet"
    df = programas(pd.read_parquet(ruta_chunks))
    catalogo = chunks_catalogo(df)

    print(f"Programas encontrados: {len(df)} | con ficha: {(df['ficha'].str.len() > 0).sum()} "
          f"| con duración: {df['ficha'].apply(lambda f: 'Duración' in f).sum()}")
    print(df.groupby(["nivel", "escuela"]).size().to_string())
    print(f"\nChunks de catálogo: {len(catalogo)}. Ejemplo:\n")
    ejemplo = next((c for c in catalogo if "Maestrías" in c["texto"]), catalogo[0])
    print(ejemplo["texto"])

    if not args.agregar:
        print("\nSolo revisión. Para agregarlo al índice: --agregar")
        return

    col = chromadb.PersistentClient(path=f"vectorstore/orion_{args.version}").get_collection(f"orion_{args.version}")
    viejos = col.get(where={"categoria": CATEGORIA})["ids"]
    if viejos:
        col.delete(ids=viejos)
    titan = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="us-east-1",
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
    )
    col.add(
        ids=[c["id"] for c in catalogo],
        embeddings=titan.embed_documents([c["texto"] for c in catalogo]),
        documents=[c["texto"] for c in catalogo],
        metadatas=[c["metadata"] for c in catalogo],
    )
    print(f"\nCatálogo agregado: {len(catalogo)} chunks (se quitaron {len(viejos)} anteriores). "
          f"Índice con {col.count():,} chunks.\nSiguiente: python rag/publicar.py {args.version}")


if __name__ == "__main__":
    main()