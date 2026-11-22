"""
¿Las URLs que espera la evaluación de Migue están en nuestro corpus?

Si una URL esperada no está en oro, la búsqueda nunca la puede encontrar: eso es
un problema de datos, no del retriever. También mira cuántas de las que faltan
quedarían cubiertas con la extracción v3 (urls_v3.json).

Correr desde la raíz de ORION:  python medallion/scraping/cobertura_eval.py
"""
import glob
import json

import pandas as pd


def norm(url):
    return url.lower().split("?")[0].split("#")[0].rstrip("/")


preguntas = json.load(open("eval/preguntas_orion_actualizadas.json", encoding="utf-8"))
oro = pd.concat(pd.read_parquet(a) for a in glob.glob("data/oro/*.parquet"))
en_oro = {norm(u) for u in oro["url"]}
en_v3 = {norm(x["url"]) for x in json.load(open("medallion/scraping/urls_v3.json", encoding="utf-8"))}

con_respuesta = [p for p in preguntas if p["tipo"] == "con_respuesta"]
cubiertas, solo_v3, sin_cobertura = [], [], []
for p in con_respuesta:
    urls = {norm(u) for u in p["urls_esperadas"]}
    if urls & en_oro:
        cubiertas.append(p)
    elif urls & en_v3:
        solo_v3.append(p)
    else:
        sin_cobertura.append(p)

total = len(con_respuesta)
print(f"Preguntas con respuesta: {total}")
print(f"  Alguna URL esperada está en oro v2:        {len(cubiertas):>3} ({len(cubiertas) / total:.0%})")
print(f"  Solo se cubrirían con la extracción v3:    {len(solo_v3):>3}")
print(f"  Ninguna URL esperada está en el corpus:    {len(sin_cobertura):>3}")

for titulo, grupo in [("Se cubrirían con v3", solo_v3), ("Sin cobertura", sin_cobertura)]:
    if grupo:
        print(f"\n{titulo}:")
        for p in grupo:
            print(f"  [{p['id']}] {p['pregunta'][:70]}")
            for u in p["urls_esperadas"]:
                print(f"       {u}")