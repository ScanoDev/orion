"""
Sube a LangSmith el dataset ORION-PARAFRASEADAS (eval/preguntas_parafraseadas.json).

Son 10 preguntas del set de desarrollo escritas como las haría un estudiante:
sin las palabras de la página, con sinónimos, sin tildes o en tono informal.
Por ejemplo "¿Qué es la reserva de cupo?" pasa a "ya me admitieron, ¿cómo aparto mi puesto?".
Ahí BM25 tiene menos palabras en común con la página y se ve si la búsqueda vectorial ayuda.
El campo parafrasea_id dice de qué pregunta original salió cada una.

Uso (desde la raíz del repo):  python eval/subir_parafraseadas.py
"""
import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env", override=True)

NOMBRE = "ORION-PARAFRASEADAS"
RUTA = RAIZ / "eval" / "preguntas_parafraseadas.json"


def main():
    client = Client()
    if client.has_dataset(dataset_name=NOMBRE):
        print(f"Ya existe: {NOMBRE}. No se modifica.")
        return

    preguntas = json.loads(RUTA.read_text(encoding="utf-8"))
    dataset = client.create_dataset(
        dataset_name=NOMBRE,
        description="Preguntas de desarrollo parafraseadas (sin las palabras de la página) - ORION EAFIT",
    )
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"pregunta": p["pregunta"]} for p in preguntas],
        outputs=[{"tipo": p["tipo"], "urls_esperadas": p["urls_esperadas"]} for p in preguntas],
        metadata=[{"id_pregunta": p["id"], "parafrasea_id": p["parafrasea_id"]} for p in preguntas],
    )
    print(f"{NOMBRE}: {len(preguntas)} preguntas cargadas")


if __name__ == "__main__":
    main()
