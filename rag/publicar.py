"""
Empaqueta un índice de Chroma y lo publica en S3, en indices/<version>/:
  - chroma_orion_<version>.zip  (carpeta de Chroma comprimida)
  - chunks.parquet              (para el BM25; sale del mismo Chroma, así siempre coincide con el índice)
  - manifiesto.json             (modelo, parámetros de chunking, versión de oro, número de chunks)

Correr desde la raíz del repo:
    python rag/publicar.py          # v2
    python rag/publicar.py v3
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
import chromadb
import pandas as pd

sys.path.insert(0, "rag")
from fragmentar import SOLAPE_CHUNK, TAMANO_CHUNK  # noqa: E402

BUCKET = "orion-eafit-2026"
MODELO_EMBEDDINGS = "amazon.titan-embed-text-v2:0"


def leer_chroma(carpeta, coleccion):
    cliente = chromadb.PersistentClient(path=carpeta)
    col = cliente.get_collection(coleccion)
    datos = col.get(include=["documents", "metadatas", "embeddings"])
    dimensiones = len(datos["embeddings"][0])
    filas = [{"page_content": doc, **meta} for doc, meta in zip(datos["documents"], datos["metadatas"])]
    return pd.DataFrame(filas), dimensiones


def main():
    parser = argparse.ArgumentParser(description="Publica el índice de ORION en S3")
    parser.add_argument("version", nargs="?", default="v2")
    args = parser.parse_args()
    version = args.version

    carpeta_chroma = f"vectorstore/orion_{version}"
    coleccion = f"orion_{version}"
    prefijo_s3 = f"indices/{version}/"
    salida = Path("data/processed") / version
    salida.mkdir(parents=True, exist_ok=True)

    df, dimensiones = leer_chroma(carpeta_chroma, coleccion)
    print(f"Chunks en Chroma ({coleccion}): {len(df):,}")

    # 1. chunks.parquet
    ruta_parquet = str(salida / "chunks.parquet")
    df.to_parquet(ruta_parquet, index=False)

    # 2. Chroma comprimida
    ruta_zip = shutil.make_archive(str(salida / f"chroma_orion_{version}"), "zip", carpeta_chroma)

    # 3. manifiesto.json
    manifiesto = {
        "coleccion": coleccion,
        "modelo_embeddings": MODELO_EMBEDDINGS,
        "dimensiones": dimensiones,
        "chunking": {
            "tamano_chunk": TAMANO_CHUNK,
            "solape": SOLAPE_CHUNK,
            "splitters": ["MarkdownHeaderTextSplitter(#,##,###)", "RecursiveCharacterTextSplitter"],
        },
        "version_oro": f"oro_{version}",
        "n_paginas": int(df["pagina_id"].nunique()),
        "n_chunks": len(df),
        "creado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    ruta_manifiesto = str(salida / "manifiesto.json")
    with open(ruta_manifiesto, "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, ensure_ascii=False, indent=2)

    # 4. Subir a S3
    s3 = boto3.client("s3")
    for ruta, nombre in [(ruta_zip, f"chroma_orion_{version}.zip"),
                         (ruta_parquet, "chunks.parquet"),
                         (ruta_manifiesto, "manifiesto.json")]:
        s3.upload_file(ruta, BUCKET, prefijo_s3 + nombre)
        print(f"Subido: s3://{BUCKET}/{prefijo_s3}{nombre}")
    Path(ruta_zip).unlink()   # el zip ya está en S3; la carpeta de Chroma queda en vectorstore/


if __name__ == "__main__":
    main()
