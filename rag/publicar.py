"""
Empaqueta el índice de Chroma y lo publica en S3:
  - chroma_orion_v1.zip  (carpeta de Chroma comprimida)
  - chunks.parquet       (para el BM25; sale del mismo Chroma, así siempre coincide con el índice)
  - manifiesto.json      (modelo, parámetros de chunking, versión de oro, número de chunks)

Correr desde la raíz del repo:  python rag/publicar.py
"""
import json
import shutil
import sys
from datetime import datetime, timezone

import boto3
import chromadb
import pandas as pd

sys.path.insert(0, "rag")
from fragmentar import SOLAPE_CHUNK, TAMANO_CHUNK  # noqa: E402

BUCKET = "orion-eafit-2026"
PREFIJO_S3 = "indices/v2/"   # cuando indexes todo oro, cámbialo a "indices/v1/"
CARPETA_CHROMA = "vectorstore/orion_v2"
COLECCION = "orion_v2"
MODELO_EMBEDDINGS = "amazon.titan-embed-text-v2:0"
VERSION_ORO = "oro_v2"
SALIDA = "data/processed"


def leer_chroma():
    cliente = chromadb.PersistentClient(path=CARPETA_CHROMA)
    col = cliente.get_collection(COLECCION)
    datos = col.get(include=["documents", "metadatas", "embeddings"])
    dimensiones = len(datos["embeddings"][0])
    filas = [{"page_content": doc, **meta} for doc, meta in zip(datos["documents"], datos["metadatas"])]
    return pd.DataFrame(filas), dimensiones


def main():
    df, dimensiones = leer_chroma()
    print(f"Chunks en Chroma: {len(df):,}")

    # 1. chunks.parquet
    ruta_parquet = f"{SALIDA}/chunks.parquet"
    df.to_parquet(ruta_parquet, index=False)

    # 2. Chroma comprimida
    ruta_zip = shutil.make_archive(f"{SALIDA}/chroma_orion_v2", "zip", CARPETA_CHROMA)

    # 3. manifiesto.json
    manifiesto = {
        "coleccion": COLECCION,
        "modelo_embeddings": MODELO_EMBEDDINGS,
        "dimensiones": dimensiones,
        "chunking": {
            "tamano_chunk": TAMANO_CHUNK,
            "solape": SOLAPE_CHUNK,
            "splitters": ["MarkdownHeaderTextSplitter(#,##,###)", "RecursiveCharacterTextSplitter"],
        },
        "version_oro": VERSION_ORO,
        "n_paginas": int(df["pagina_id"].nunique()),
        "n_chunks": len(df),
        "creado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    ruta_manifiesto = f"{SALIDA}/manifiesto.json"
    with open(ruta_manifiesto, "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, ensure_ascii=False, indent=2)

    # 4. Subir a S3
    s3 = boto3.client("s3")
    for ruta, nombre in [(ruta_zip, "chroma_orion_v2.zip"),
                         (ruta_parquet, "chunks.parquet"),
                         (ruta_manifiesto, "manifiesto.json")]:
        s3.upload_file(ruta, BUCKET, PREFIJO_S3 + nombre)
        print(f"Subido: s3://{BUCKET}/{PREFIJO_S3}{nombre}")


if __name__ == "__main__":
    main()