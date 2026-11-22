"""
Buscador BM25 (por palabras clave) adaptado al español, como retriever de LangChain.

Por qué propio: el BM25Retriever de LangChain vive en langchain-community (en retiro)
y solo separa por espacios, así que "matricula" no encuentra "Matrícula".
Aquí se pasa todo a minúsculas, se quitan tildes y signos, se ignoran palabras vacías
y cada palabra se reduce a su raíz (stemming), para que "becas" encuentre "beca".

Uso:
    buscador = BM25Espanol(documentos=chunks, k=5)
    resultados = buscador.invoke("matricula ingenieria de sistemas")
"""
import re
import unicodedata
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr
from rank_bm25 import BM25Okapi
from snowballstemmer import stemmer

_stemmer = stemmer("spanish")


@lru_cache(maxsize=200_000)
def _raiz(palabra):
    """Raíz de una palabra, guardada en memoria porque el vocabulario se repite mucho."""
    return _stemmer.stemWord(palabra)

# Palabras muy comunes que no ayudan a distinguir documentos (español e inglés, porque el corpus es mixto)
PALABRAS_VACIAS = set("""
a al algo algun alguna algunas alguno algunos ante antes como con contra cual cuales cuando de del desde
donde durante e el ella ellas ello ellos en entre era es esa esas ese eso esos esta estas este esto estos
fue fueron ha han hasta hay la las le les lo los mas me mi mis muy ni no nos o os otra otras otro otros
para pero poco por porque que quien quienes se sea segun ser si sin sobre su sus tambien tan te tiene
tienen todo todos tu tus un una unas uno unos y ya yo cuanto cuanta cuantos cuantas cual
the of and to in for on with at by from is are was were be or as an this that it its
""".split())


def tokenizar(texto):
    texto = unicodedata.normalize("NFKD", (texto or "").lower())
    # quita tildes y caracteres invisibles (como el espacio de ancho cero U+200B que traen
    # algunas páginas dentro de las palabras: "financ\u200biación" debe ser "financiación")
    texto = "".join(
        c for c in texto
        if not unicodedata.combining(c) and unicodedata.category(c) != "Cf"
    )
    palabras = [t for t in re.findall(r"[a-z0-9]+", texto) if t not in PALABRAS_VACIAS]
    return [_raiz(p) for p in palabras]   # becas, beca -> bec


class BM25Espanol(BaseRetriever):
    documentos: list[Document]
    k: int = 5
    _bm25: Any = PrivateAttr()

    def model_post_init(self, __context):
        self._bm25 = BM25Okapi([tokenizar(d.page_content) for d in self.documentos])

    def _get_relevant_documents(self, query, *, run_manager=None):
        puntajes = self._bm25.get_scores(tokenizar(query))
        mejores = sorted(range(len(puntajes)), key=lambda i: puntajes[i], reverse=True)[: self.k]
        resultados = []
        for i in mejores:
            if puntajes[i] <= 0:
                break
            doc = self.documentos[i]
            resultados.append(Document(page_content=doc.page_content,
                                       metadata={**doc.metadata, "puntaje_bm25": round(float(puntajes[i]), 3)}))
        return resultados