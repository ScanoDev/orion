"""
Prueba con 3 páginas si Firecrawl las devuelve en español al pedirlas desde Colombia.
Gasta 3 créditos. Necesita FIRECRAWL_API_KEY en el .env de ORION.

Correr desde la raíz de ORION:  python medallion/scraping/probar_idioma.py
"""
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()
LLAVE = os.environ["FIRECRAWL_API_KEY"]
UBICACION = {"country": "CO", "languages": ["es-CO"]}

lista = json.load(open("medallion/scraping/urls_v3.json", encoding="utf-8"))
prueba = [x for x in lista if x["motivo"] == "traducida_al_ingles"][:3]

for x in prueba:
    r = requests.post(
        "https://api.firecrawl.dev/v2/scrape",
        headers={"Authorization": f"Bearer {LLAVE}"},
        json={"url": x["url"], "formats": ["markdown"], "onlyMainContent": True, "location": UBICACION},
        timeout=90,
    )
    datos = r.json().get("data", {})
    meta = datos.get("metadata", {})
    print("=" * 80)
    print("URL:", x["url"])
    print(f"Título antes: {x['title']}")
    print(f"Título ahora: {meta.get('title')} | idioma: {meta.get('language')} | HTTP {r.status_code}")
    print((datos.get("markdown") or "(sin markdown)")[:300])