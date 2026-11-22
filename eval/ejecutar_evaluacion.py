
import sys
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client, evaluate

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

load_dotenv(RAIZ / ".env", override=True)

from rag.cadena import responder, VERSION_CADENA
from eval.evaluadores import (
    evaluar_recuperacion,
    evaluar_abstencion,
)
from eval.evaluador_calidad import evaluar_calidad


client = Client()

DATASET = "ORION-DESARROLLO"


def ejecutar_orion(inputs):
    resultado = responder(inputs["pregunta"])

    return {
        "respuesta": resultado["respuesta"],
        "fuentes": resultado["fuentes"],
        "urls_contexto": resultado["urls_contexto"],
        "chunk_ids": resultado["chunk_ids"],
        "contexto": resultado["contexto"],
    }


if __name__ == "__main__":
    resultados = evaluate(
        ejecutar_orion,
        data=DATASET,
        evaluators=[
            evaluar_recuperacion,
            evaluar_abstencion,
            evaluar_calidad,
        ],
        experiment_prefix=VERSION_CADENA,
        description="Evaluación inicial de ORION con BM25",
        max_concurrency=1,
        client=client,
    )

    print("\nExperimento finalizado")
    print(resultados)
