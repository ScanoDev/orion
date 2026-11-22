"""
Arma el índice de Chroma de una versión nueva de oro reusando los embeddings de la anterior.

Por cada chunk de oro nuevo:
  - si el índice anterior tiene el mismo chunk_id con el mismo texto, copia su embedding (gratis)
  - si es nuevo o su texto cambió, lo calcula con Titan V2
Los chunks que ya no existen en oro nuevo simplemente no se copian.
El chunk_id sale de md5(url), así que una página conserva sus ids entre versiones.

Uso (desde la raíz del repo, con oro nuevo en data/oro_v3/):
    python rag/indexar_incremental.py                       # v2 -> v3
    python rag/indexar_incremental.py --anterior v2 --nueva v3 --oro "data/oro_v3/*.parquet"
Deja:  vectorstore/orion_v3/  (colección orion_v3). Después: python rag/publicar.py v3
"""
import argparse
import glob
import shutil
import sys
import time
from pathlib import Path

import chromadb
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError
from langchain_aws import BedrockEmbeddings

sys.path.insert(0, str(Path(__file__).parent))
from fragmentar import fragmentar_documentos  # noqa: E402

REGION = "us-east-1"
MODELO_EMBEDDINGS = "amazon.titan-embed-text-v2:0"
LOTE_TITAN = 50
LOTE_CHROMA = 2000
MAX_REINTENTOS = 6
REINTENTABLES = {"ThrottlingException", "ModelTimeoutException", "ServiceUnavailableException"}


def con_reintentos(llamada):
    for intento in range(MAX_REINTENTOS):
        try:
            return llamada()
        except ClientError as e:
            codigo = e.response["Error"]["Code"]
            if codigo not in REINTENTABLES or intento == MAX_REINTENTOS - 1:
                raise
            print(f"  {codigo}: reintento {intento + 1} en {2 ** intento}s...")
            time.sleep(2 ** intento)


def limpiar_metadata(m):
    """Chroma solo acepta str, int, float y bool en la metadata."""
    return {k: ("" if v is None else v) for k, v in m.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anterior", default="v2")
    parser.add_argument("--nueva", default="v3")
    parser.add_argument("--oro", default=None, help='Por defecto "data/oro_<nueva>/*.parquet"')
    args = parser.parse_args()

    ruta_oro = args.oro or f"data/oro_{args.nueva}/*.parquet"
    archivos = glob.glob(ruta_oro)
    if not archivos:
        raise SystemExit(f"No hay archivos en {ruta_oro}")
    oro = pd.concat(pd.read_parquet(a) for a in archivos)
    chunks = fragmentar_documentos(oro)
    print(f"Oro {args.nueva}: {len(oro):,} páginas | {len(chunks):,} chunks")

    # 1. Índice anterior
    anterior = chromadb.PersistentClient(path=f"vectorstore/orion_{args.anterior}")
    col_ant = anterior.get_collection(f"orion_{args.anterior}")
    datos = col_ant.get(include=["documents", "embeddings"])
    previo = {i: (doc, emb) for i, doc, emb in zip(datos["ids"], datos["documents"], datos["embeddings"])}
    print(f"Índice {args.anterior}: {len(previo):,} chunks")

    # 2. Qué se reusa y qué se calcula
    reusar, calcular = [], []
    for c in chunks:
        cid = c.metadata["chunk_id"]
        if cid in previo and previo[cid][0] == c.page_content:
            reusar.append((c, previo[cid][1]))
        else:
            calcular.append(c)
    print(f"Se reusan {len(reusar):,} embeddings | se calculan {len(calcular):,} con Titan "
          f"| quedan por fuera {len(set(previo) - {c.metadata['chunk_id'] for c in chunks}):,} chunks viejos")

    # 3. Colección nueva (si existe, se borra y se arma de cero)
    carpeta = Path(f"vectorstore/orion_{args.nueva}")
    if carpeta.exists():
        shutil.rmtree(carpeta)
    nueva = chromadb.PersistentClient(path=str(carpeta))
    col = nueva.create_collection(f"orion_{args.nueva}", metadata={"hnsw:space": "cosine"})

    for i in range(0, len(reusar), LOTE_CHROMA):
        lote = reusar[i:i + LOTE_CHROMA]
        col.add(
            ids=[c.metadata["chunk_id"] for c, _ in lote],
            embeddings=[list(e) for _, e in lote],
            documents=[c.page_content for c, _ in lote],
            metadatas=[limpiar_metadata(c.metadata) for c, _ in lote],
        )
    print(f"  Copiados {len(reusar):,} chunks del índice {args.anterior}")

    titan = BedrockEmbeddings(
        model_id=MODELO_EMBEDDINGS,
        region_name=REGION,
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
    )
    inicio = time.time()
    for i in range(0, len(calcular), LOTE_TITAN):
        lote = calcular[i:i + LOTE_TITAN]
        textos = [c.page_content for c in lote]
        vectores = con_reintentos(lambda: titan.embed_documents(textos))
        col.add(
            ids=[c.metadata["chunk_id"] for c in lote],
            embeddings=vectores,
            documents=textos,
            metadatas=[limpiar_metadata(c.metadata) for c in lote],
        )
        hechos = min(i + LOTE_TITAN, len(calcular))
        if hechos % 500 < LOTE_TITAN or hechos == len(calcular):
            print(f"  Titan {hechos:,}/{len(calcular):,} | {time.time() - inicio:.0f} s")

    print(f"\nÍndice {args.nueva} listo: {col.count():,} chunks en {carpeta}")
    print(f"Siguiente: python rag/publicar.py {args.nueva}")


if __name__ == "__main__":
    main()
