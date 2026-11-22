"""
Sube a bronce las páginas de la extracción v3 (data/raw_v3/).

Qué revisa antes de subir:
  1. Solo sube archivos con status 200 y markdown con texto. Así, una página que
     falló en v3 no reemplaza a su versión buena en bronce.
  2. Cada archivo se llama md5(url).json, igual que en bronce. Las páginas que antes
     salieron traducidas al inglés deben REEMPLAZAR un archivo que ya existe, y las
     faltantes deben ser NUEVAS. El resumen muestra si eso se cumple.
  3. Con --subir, exige que exista el respaldo de bronce con al menos los mismos archivos.

Uso (desde la raíz del repo):
    python medallion/scraping/subir_bronce_v3.py            # solo revisa y muestra el resumen
    aws s3 sync s3://orion-eafit-2026/bronce/ s3://orion-eafit-2026/bronce_respaldo_v2/
    python medallion/scraping/subir_bronce_v3.py --subir    # sube
"""
import argparse
import glob
import hashlib
import json
import os
from collections import Counter

import boto3

BUCKET = "orion-eafit-2026"
BRONCE = "bronce/"
RESPALDO = "bronce_respaldo_v2/"
CARPETA_LOCAL = "data/raw_v3"
LISTA_V3 = "medallion/scraping/urls_v3.json"


def llaves(s3, prefijo):
    """Nombres de archivo (sin carpeta) que hay en S3 bajo un prefijo."""
    nombres = set()
    for pagina in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefijo):
        for obj in pagina.get("Contents", []):
            resto = obj["Key"][len(prefijo):]
            if resto and "/" not in resto:          # solo los archivos de ese nivel
                nombres.add(resto)
    return nombres


def es_buena(ruta):
    """(True, '') si la página sirve; (False, motivo) si no."""
    try:
        with open(ruta, encoding="utf-8") as f:
            r = json.load(f)
    except (OSError, ValueError):
        return False, "json_danado"
    data = (r.get("respuesta") or {}).get("data") or {}
    status = (data.get("metadata") or {}).get("statusCode") or r.get("http_status")
    if status != 200:
        return False, f"status_{status}"
    if len((data.get("markdown") or "").strip()) == 0:
        return False, "sin_markdown"
    return True, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subir", action="store_true", help="Sube a bronce (si no, solo revisa)")
    args = parser.parse_args()

    s3 = boto3.client("s3")
    en_bronce = llaves(s3, BRONCE)
    print(f"Archivos en s3://{BUCKET}/{BRONCE}: {len(en_bronce):,}")

    motivo_de = {
        hashlib.md5(x["url"].encode()).hexdigest() + ".json": x["motivo"]
        for x in json.load(open(LISTA_V3, encoding="utf-8"))
    }

    buenas, malas, cruce = [], Counter(), Counter()
    for ruta in sorted(glob.glob(os.path.join(CARPETA_LOCAL, "*.json"))):
        nombre = os.path.basename(ruta)
        ok, motivo = es_buena(ruta)
        if not ok:
            malas[motivo] += 1
            continue
        buenas.append((ruta, nombre))
        accion = "reemplaza" if nombre in en_bronce else "nueva"
        cruce[(motivo_de.get(nombre, "fuera_de_la_lista"), accion)] += 1

    print(f"\nArchivos en {CARPETA_LOCAL}: {len(buenas) + sum(malas.values()):,}")
    print(f"  Buenas (se suben): {len(buenas):,}")
    for motivo, n in malas.most_common():
        print(f"  No se suben ({motivo}): {n:,}")
    print(f"  Pendientes de la lista (aún sin bajar): {len(motivo_de) - len(buenas) - sum(malas.values()):,}")

    print("\nMotivo de la lista       Acción en bronce   Páginas")
    for (motivo, accion), n in sorted(cruce.items()):
        print(f"  {motivo:<22} {accion:<18} {n:,}")
    print("Esperado: traducida_al_ingles -> reemplaza;  faltante_alta / faltante_media -> nueva")

    if not args.subir:
        print("\nSolo revisión. Para subir: respalda bronce y corre con --subir")
        return

    en_respaldo = llaves(s3, RESPALDO)
    if len(en_respaldo) < len(en_bronce):
        raise SystemExit(
            f"\nEl respaldo tiene {len(en_respaldo):,} archivos y bronce {len(en_bronce):,}. Corre primero:\n"
            f"aws s3 sync s3://{BUCKET}/{BRONCE} s3://{BUCKET}/{RESPALDO}"
        )

    for i, (ruta, nombre) in enumerate(buenas, start=1):
        s3.upload_file(ruta, BUCKET, BRONCE + nombre)
        if i % 100 == 0 or i == len(buenas):
            print(f"  Subidas {i:,}/{len(buenas):,}")
    print(f"\nListo. Bronce queda con {len(llaves(s3, BRONCE)):,} archivos.")


if __name__ == "__main__":
    main()
