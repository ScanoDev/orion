"""
Arma la lista de URLs para la extracción v3 de ORION:
  1. Faltantes de secciones útiles (programas, calendario, becas, servicios, sedes...).
  2. Páginas que Firecrawl bajó traducidas al inglés aunque su URL es en español
     (se vuelven a pedir con location CO / es-CO).

Correr desde la raíz del repo ORION:
    python medallion/scraping/lista_v3.py RUTA/faltantes.json

Lee:   faltantes.json (de faltantes.py) y data/oro/*.parquet (oro v2)
Deja:  medallion/scraping/urls_v3.json  (lista de {"url", "title", "motivo"})
"""
import glob
import json
import sys

import pandas as pd

PRIORIDAD_ALTA = {
    "plan-de-estudio", "posgrados", "calendario-academico", "becas-y-financiacion", "idiomas",
    "servicios", "consultorio-juridico", "campus-eafit", "eafit-pereira", "eafit-llanogrande",
}
PRIORIDAD_MEDIA = {"bienestar-universitario", "institucional"}


def seccion(url):
    partes = url.split("?")[0].split("/")
    primera = partes[3].lower() if len(partes) > 3 else ""
    return (partes[4].lower() if len(partes) > 4 else "") if primera == "en" else primera


faltantes = json.load(open(sys.argv[1], encoding="utf-8"))
lista = []
for u in faltantes:
    s = seccion(u["url"])
    if s in PRIORIDAD_ALTA or s.startswith("escuela-"):
        lista.append({**u, "motivo": "faltante_alta"})
    elif s in PRIORIDAD_MEDIA:
        lista.append({**u, "motivo": "faltante_media"})

oro = pd.concat(pd.read_parquet(a) for a in glob.glob("data/oro/*.parquet"))
traducidas = oro[(oro["idioma"] == "en") & (~oro["url"].str.contains("/en/", regex=False))]
lista += [{"url": u, "title": t, "motivo": "traducida_al_ingles"} for u, t in zip(traducidas["url"], traducidas["titulo"])]

resumen = pd.Series([x["motivo"] for x in lista]).value_counts()
print(resumen.to_string(), f"\nTotal: {len(lista):,} URLs (aprox. {len(lista):,} créditos de Firecrawl)")
json.dump(lista, open("medallion/scraping/urls_v3.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("Lista en medallion/scraping/urls_v3.json")