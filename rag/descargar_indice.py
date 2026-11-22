"""
Baja de S3 el índice publicado y lo deja listo para usar:

    indices/<version>/chroma_orion_<version>.zip -> vectorstore/orion_<version>/
    indices/<version>/chunks.parquet             -> data/processed/<version>/
    indices/<version>/manifiesto.json            -> data/processed/<version>/

Uso (desde la raíz del repo):
    python rag/descargar_indice.py             # versión v3 (la de la app)
    python rag/descargar_indice.py v2          # la línea base, para comparar
    python rag/descargar_indice.py --forzar    # vuelve a bajar aunque ya exista
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

BUCKET = "orion-eafit-2026"
PROCESADOS = Path("data/processed")


def descargar(version, forzar=False):
    s3 = boto3.client("s3")
    prefijo = f"indices/{version}/"
    carpeta_chroma = Path(f"vectorstore/orion_{version}")
    SALIDA = PROCESADOS / version   # cada versión en su carpeta, para poder comparar v2 y v3
    SALIDA.mkdir(parents=True, exist_ok=True)

    def bajar(nombre, destino):
        print(f"Bajando s3://{BUCKET}/{prefijo}{nombre} ...")
        s3.download_file(BUCKET, prefijo + nombre, str(destino))

    try:
        # 1. Chroma (zip -> carpeta)
        if carpeta_chroma.exists() and not forzar:
            print(f"Ya existe {carpeta_chroma} (usa --forzar para reemplazarla)")
        else:
            ruta_zip = SALIDA / f"chroma_orion_{version}.zip"
            bajar(f"chroma_orion_{version}.zip", ruta_zip)
            if carpeta_chroma.exists():
                shutil.rmtree(carpeta_chroma)
            with zipfile.ZipFile(ruta_zip) as z:
                z.extractall(carpeta_chroma)
            ruta_zip.unlink()
            print(f"Chroma lista en {carpeta_chroma}")

        # 2. chunks.parquet y manifiesto.json
        for nombre in ("chunks.parquet", "manifiesto.json"):
            destino = SALIDA / nombre
            if destino.exists() and not forzar:
                print(f"Ya existe {destino} (usa --forzar para reemplazarlo)")
            else:
                bajar(nombre, destino)
    except ClientError as e:
        raise SystemExit(f"No se pudo bajar el índice {version} de S3: {e}")

    # 3. Resumen del manifiesto
    with open(SALIDA / "manifiesto.json", encoding="utf-8") as f:
        m = json.load(f)
    print(f"\nÍndice {version}: {m['n_chunks']:,} chunks | {m['n_paginas']:,} páginas | "
          f"oro: {m['version_oro']} | {m['modelo_embeddings']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baja el índice de ORION desde S3")
    parser.add_argument("version", nargs="?", default="v3")
    parser.add_argument("--forzar", action="store_true")
    args = parser.parse_args()
    descargar(args.version, args.forzar)