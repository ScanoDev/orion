"""
Extracción v3 de ORION: baja con Firecrawl las URLs de urls_v3.json, pidiéndolas
desde Colombia en español, y las guarda con el mismo formato de bronce.

Correr desde la raíz de ORION:  python medallion/scraping/extraer_v3.py
Necesita FIRECRAWL_API_KEY en el .env.

Se puede interrumpir y volver a correr: salta las que ya están en data/raw_v3/.
Cada archivo se llama md5(url).json, igual que en bronce, así que al subirlos
reemplazan a las versiones traducidas al inglés.
"""
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()
LLAVE = os.environ["FIRECRAWL_API_KEY"]
ENTRADA = "medallion/scraping/urls_v3.json"
SALIDA = "data/raw_v3"
HILOS = 1
                     # peticiones al mismo tiempo (Firecrawl limita la concurrencia)
REINTENTOS = 5
CUERPO_BASE = {
    "formats": ["markdown"],
    "onlyMainContent": True,
    "timeout": 30000,         # milisegundos por página
    "location": {"country": "CO", "languages": ["es-CO"]},
}


def ruta_de(url):
    return os.path.join(SALIDA, hashlib.md5(url.encode()).hexdigest() + ".json")


def extraer(url):
    """Devuelve 'ok', 'error_pagina' o 'fallo' (fallo = no se guardó; se reintenta en la próxima corrida)."""
    for intento in range(REINTENTOS):
        try:
            r = requests.post(
                "https://api.firecrawl.dev/v2/scrape",
                headers={"Authorization": f"Bearer {LLAVE}"},
                json={"url": url, **CUERPO_BASE},
                timeout=120,
            )
            if r.status_code == 429 or r.status_code >= 500:     # demasiadas peticiones o error del servidor
                time.sleep(2 ** intento)
                continue
            registro = {
                "url_pedida": url,
                "fecha_extraccion": datetime.now(timezone.utc).isoformat(),
                "http_status": r.status_code,
                "respuesta": r.json(),
            }
            with open(ruta_de(url), "w", encoding="utf-8") as f:
                json.dump(registro, f, ensure_ascii=False)
            return "ok" if r.status_code == 200 else "error_pagina"
        except (requests.RequestException, ValueError):
            time.sleep(2 ** intento)
    return "fallo"


def main():
    os.makedirs(SALIDA, exist_ok=True)
    urls = [x["url"] for x in json.load(open(ENTRADA, encoding="utf-8"))]
    pendientes = [u for u in urls if not os.path.exists(ruta_de(u))]
    print(f"URLs: {len(urls):,} | ya bajadas: {len(urls) - len(pendientes):,} | pendientes: {len(pendientes):,}")

    conteo = {"ok": 0, "error_pagina": 0, "fallo": 0}
    inicio = time.time()
    with ThreadPoolExecutor(max_workers=HILOS) as pool:
        futuros = {pool.submit(extraer, u): u for u in pendientes}
        for i, futuro in enumerate(as_completed(futuros), start=1):
            conteo[futuro.result()] += 1
            if i % 25 == 0 or i == len(pendientes):
                print(f"  {i:,}/{len(pendientes):,} | {conteo} | {time.time() - inicio:.0f} s")

    print("\nListo:", conteo)
    if conteo["fallo"]:
        print("Hubo fallos: vuelve a correr el script y solo intentará esas.")


if __name__ == "__main__":
    main()