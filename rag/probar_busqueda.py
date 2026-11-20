"""
Prueba de la búsqueda BM25 sobre todo oro, sin usar AWS.

Antes: aws s3 cp s3://orion-eafit-2026/publicado/oro_v1/ data/oro/ --recursive
Correr desde la raíz del repo:  python rag/probar_busqueda.py
"""
import glob
import sys
import time

import pandas as pd

sys.path.insert(0, "rag")
from bm25_espanol import BM25Espanol  # noqa: E402
from fragmentar import fragmentar_documentos  # noqa: E402

PREGUNTAS = [
    "requisitos de admision para pregrado",
    "matricula ingenieria de sistemas",
    "becas para estudiantes",
]

archivos = glob.glob("data/oro/*.parquet")
oro = pd.concat(pd.read_parquet(a) for a in archivos)
print(f"Páginas de oro: {len(oro):,}")

inicio = time.time()
chunks = fragmentar_documentos(oro)
largos = [len(c.page_content) for c in chunks]
print(f"Chunks: {len(chunks):,} | promedio {sum(largos) / len(largos):.0f} caracteres | "
      f"máximo {max(largos):,} | {time.time() - inicio:.1f} s")
print("\nEjemplo de chunk:\n" + "-" * 60)
print(chunks[0].page_content[:400])
print("-" * 60, chunks[0].metadata)

inicio = time.time()
buscador = BM25Espanol(documentos=chunks, k=5)
print(f"\nÍndice BM25 listo en {time.time() - inicio:.1f} s")

for pregunta in PREGUNTAS:
    print(f"\n=== {pregunta}")
    for doc in buscador.invoke(pregunta):
        m = doc.metadata
        print(f"  {m['puntaje_bm25']:>6} | {m['categoria']:<16} | {m['titulo'][:60]}")
        print(f"         {m['url']}")