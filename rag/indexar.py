"""
Indexa los chunks de oro en Chroma (colección orion_v1) con embeddings Titan V2.

Correr desde la raíz del repo:  python rag/indexar.py
"""
import glob
import sys
import time

import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma

sys.path.insert(0, "rag")
from fragmentar import fragmentar_documentos  # noqa: E402

REGION = "us-east-1"
MODELO_EMBEDDINGS = "amazon.titan-embed-text-v2:0"
COLECCION = "orion_v1"
CARPETA_CHROMA = "vectorstore/orion_v1"
N_PAGINAS = None       # None = todas las páginas
TAMANO_LOTE = 50
MAX_REINTENTOS = 6
ERRORES_REINTENTABLES = {"ThrottlingException", "ModelTimeoutException", "ServiceUnavailableException"}

embeddings = BedrockEmbeddings(
    model_id=MODELO_EMBEDDINGS,
    region_name=REGION,
    config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),  # reintentos de boto3
)


def con_reintentos(llamada):
    """Ejecuta llamada() y, si Bedrock limita las peticiones, espera y reintenta (1s, 2s, 4s...)."""
    for intento in range(MAX_REINTENTOS):
        try:
            return llamada()
        except ClientError as e:
            codigo = e.response["Error"]["Code"]
            if codigo not in ERRORES_REINTENTABLES or intento == MAX_REINTENTOS - 1:
                raise
            espera = 2 ** intento
            print(f"  {codigo}: reintento {intento + 1} en {espera}s...")
            time.sleep(espera)


def main():
    archivos = glob.glob("data/oro/*.parquet")
    oro = pd.concat(pd.read_parquet(a) for a in archivos)
    if N_PAGINAS:
        oro = oro.head(N_PAGINAS)

    chunks = fragmentar_documentos(oro)
    print(f"Páginas: {len(oro):,} | Chunks: {len(chunks):,}")

    db = Chroma(
        collection_name=COLECCION,
        embedding_function=embeddings,
        persist_directory=CARPETA_CHROMA,
        collection_metadata={"hnsw:space": "cosine"},
    )

    inicio = time.time()
    for i in range(0, len(chunks), TAMANO_LOTE):
        lote = chunks[i:i + TAMANO_LOTE]
        ids = [c.metadata["chunk_id"] for c in lote]
        con_reintentos(lambda: db.add_documents(lote, ids=ids))
        print(f"  {min(i + TAMANO_LOTE, len(chunks)):,}/{len(chunks):,} chunks | {time.time() - inicio:.0f} s")

    # Chequeo rápido
    print("\nPrueba de búsqueda: 'matricula ingenieria de sistemas'")
    for doc in db.similarity_search("matricula ingenieria de sistemas", k=3):
        print(f"  {doc.metadata['categoria']:<16} | {doc.metadata['titulo'][:60]}")


if __name__ == "__main__":
    main()