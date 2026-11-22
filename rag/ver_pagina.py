"""
Muestra qué pasó con un dato de una página en cada capa:
  1. Bronce (S3): el markdown que bajó Firecrawl, antes de limpiar
  2. Chunks del índice: lo que quedó después de plata y oro, y lo que ve ORION

Uso (desde la raíz del repo):
    python rag/ver_pagina.py URL palabra
    python rag/ver_pagina.py https://www.eafit.edu.co/posgrados/escuela-ciencias-aplicadas-ingenieria/maestria-ingenieria semestre
"""
import hashlib
import json
import sys
import unicodedata

import boto3
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "rag")
import cadena  # noqa: E402

BUCKET = "orion-eafit-2026"
url, palabra = sys.argv[1], sys.argv[2]


def normal(texto):
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def mostrar(lineas, titulo):
    clave = normal(palabra)
    hallazgos = [i for i, l in enumerate(lineas) if clave in normal(l)]
    print(f"{titulo}: {len(lineas)} líneas, {len(hallazgos)} con '{palabra}'")
    for i in hallazgos[:15]:
        contexto = [l.strip() for l in lineas[max(0, i - 2): i + 3] if l.strip()]
        print("   ...", " | ".join(contexto)[:300])


# 1. Bronce
s3 = boto3.client("s3")
llave = f"bronce/{hashlib.md5(url.encode()).hexdigest()}.json"
try:
    crudo = json.loads(s3.get_object(Bucket=BUCKET, Key=llave)["Body"].read())
    markdown = ((crudo.get("respuesta") or {}).get("data") or {}).get("markdown") or ""
    print(f"BRONCE {llave} | extraída {crudo.get('fecha_extraccion', '')[:10]}")
    mostrar(markdown.splitlines(), "  markdown")
except s3.exceptions.NoSuchKey:
    print(f"BRONCE: no existe {llave}")

# 2. Chunks del índice
indice = cadena.indice_corpus()
chunks = pd.read_parquet(cadena._ruta_chunks(indice))
de_la_pagina = chunks[chunks["url"].str.rstrip("/") == url.rstrip("/")]
print(f"\nCHUNKS {indice}: {len(de_la_pagina)} chunks de esta página")
for _, c in de_la_pagina.iterrows():
    mostrar(c["page_content"].splitlines(), f"  {c['chunk_id']}")
if len(de_la_pagina):
    print("\nPRIMER CHUNK COMPLETO:\n" + de_la_pagina.iloc[0]["page_content"][:1500])
