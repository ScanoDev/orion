"""
Revisa las preguntas de cobertura (eval/preguntas_cobertura.json) contra los dos corpus:
la página esperada debe estar en oro v3 y NO en oro v2. Así v2 no puede responderlas
desde esa página y la comparación v2 contra v3 mide lo que agregaron las páginas nuevas.
También muestra el comienzo del texto de cada página, para confirmar que la respuesta está ahí.

Uso (desde la raíz del repo, con data/oro/ = v2 y data/oro_v3/ = v3):
    python eval/verificar_cobertura.py
"""
import glob
import json
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


def leer(ruta):
    return pd.concat(pd.read_parquet(a) for a in glob.glob(ruta))


v2 = set(leer("data/oro/*.parquet")["url"].str.lower().str.rstrip("/"))
v3 = leer("data/oro_v3/*.parquet")
v3["clave"] = v3["url"].str.lower().str.rstrip("/")
por_url = v3.set_index("clave")

preguntas = json.load(open("eval/preguntas_cobertura.json", encoding="utf-8"))
sirven = 0
for p in preguntas:
    esperadas = [u.lower().rstrip("/") for u in p["urls_esperadas"]]
    en_v3 = [u for u in esperadas if u in por_url.index]
    en_v2 = [u for u in esperadas if u in v2]
    ok = bool(en_v3) and not en_v2
    sirven += ok
    print("=" * 90)
    print(f"[{p['id']}] {'SIRVE' if ok else 'NO SIRVE'} | en v3: {len(en_v3)}/{len(esperadas)} | en v2: {len(en_v2)}")
    print("PREGUNTA:", p["pregunta"])
    for u in en_v3:
        texto = str(por_url.loc[u, "texto"]) if not isinstance(por_url.loc[u], pd.DataFrame) else str(por_url.loc[u].iloc[0]["texto"])
        print(f"  {u}\n  {' '.join(texto.split())[:350]}")
print("=" * 90)
print(f"Sirven {sirven} de {len(preguntas)}")
