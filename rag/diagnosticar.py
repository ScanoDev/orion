"""
Diagnóstico de una pregunta que ORION respondió mal:
  1. ¿Están en el corpus las páginas que debería encontrar?
  2. ¿Qué trajo cada modo de búsqueda y en qué puesto?

Uso (desde la raíz del repo):
    python rag/diagnosticar.py "Dame 5 maestrías relacionadas con la computación" maestria datos sistemas inteligencia
    (las palabras después de la pregunta se buscan en la URL y el título de las páginas del corpus)
"""
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "rag")
import cadena  # noqa: E402

pregunta = sys.argv[1]
palabras = [p.lower() for p in sys.argv[2:]]
indice = cadena.indice_corpus()

# 1. Páginas del corpus que tienen la primera palabra y alguna de las demás en la URL o el título
chunks = pd.read_parquet(cadena._ruta_chunks(indice))
paginas = chunks.drop_duplicates("url")[["url", "titulo", "categoria"]]
texto = (paginas["url"] + " " + paginas["titulo"]).str.lower()
filtro = texto.str.contains(palabras[0]) if palabras else texto.str.len() > 0
if len(palabras) > 1:
    filtro &= texto.str.contains("|".join(palabras[1:]))
encontradas = paginas[filtro]
print(f"CORPUS {indice}: {len(encontradas)} páginas con {palabras}")
for _, p in encontradas.head(40).iterrows():
    print(f"  {p['categoria']:<16} {p['titulo'][:60]} | {p['url']}")

# 2. Lo que trae cada modo (los 10 que salen de la búsqueda; al modelo le llegan 5)
for modo in cadena.MODOS:
    docs = cadena._buscador(modo, indice).invoke(pregunta)
    print(f"\nMODO {modo}:")
    for i, d in enumerate(docs, start=1):
        m = d.metadata
        marca = "  <- esperada" if m["url"] in set(encontradas["url"]) else ""
        print(f"  {i:>2}. {m['titulo'][:55]} | {m['url'].replace('https://www.eafit.edu.co', '')}{marca}")
