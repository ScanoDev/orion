"""
Qué páginas del universo de URLs no se descargaron, agrupadas por sección.

Correr desde la raíz del repo de los talleres (aws-medallion-streaming):
    python RUTA_A_ORION/medallion/scraping/faltantes.py

Lee:   medallion/scraping/eafit_urls.json  (universo: lista de {"url", "title"})
       data/raw/*.json                      (descargadas: nombre = md5 de la URL)
Deja:  faltantes.json                       (las URLs que faltan, para volver a correr el scraping)
"""
import collections
import hashlib
import json
import os
import sys

UNIVERSO = "medallion/scraping/eafit_urls.json"
RAW = "data/raw"

universo = json.load(open(UNIVERSO, encoding="utf-8"))
descargadas = {f[:-5] for f in os.listdir(RAW) if f.endswith(".json")}
faltan = [u for u in universo if hashlib.md5(u["url"].encode()).hexdigest() not in descargadas]

print(f"Universo: {len(universo):,} | descargadas: {len(descargadas):,} | faltan: {len(faltan):,}\n")


def seccion(url):
    partes = url.split("?")[0].split("/")
    primera = partes[3].lower() if len(partes) > 3 else ""
    return (partes[4].lower() if len(partes) > 4 else "") if primera == "en" else primera


conteo = collections.Counter(seccion(u["url"]) for u in faltan)
for nombre, n in conteo.most_common(int(sys.argv[1]) if len(sys.argv) > 1 else 40):
    print(f"{n:>5}  {nombre or '(raíz)'}")

json.dump(faltan, open("faltantes.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nLista completa en faltantes.json")