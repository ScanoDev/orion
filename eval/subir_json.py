"""
Sube a LangSmith un dataset de preguntas en el formato de preguntas_orion_actualizadas.json
(campos: id, pregunta, tipo, urls_esperadas). Si el dataset ya existe, no lo toca.

Uso (desde la raíz del repo):
    python eval/subir_json.py eval/preguntas_cobertura.json ORION-COBERTURA
    python eval/subir_json.py eval/preguntas_ciegas.json ORION-CIEGAS
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env", override=True)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Uso: python eval/subir_json.py <archivo.json> <NOMBRE-DATASET>")
    ruta, nombre = Path(sys.argv[1]), sys.argv[2]

    client = Client()
    if client.has_dataset(dataset_name=nombre):
        print(f"Ya existe: {nombre}. No se modifica.")
        return

    preguntas = json.loads(ruta.read_text(encoding="utf-8"))
    dataset = client.create_dataset(dataset_name=nombre, description=f"ORION EAFIT: {ruta.name}")
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"pregunta": p["pregunta"]} for p in preguntas],
        outputs=[{"tipo": p["tipo"], "urls_esperadas": p["urls_esperadas"]} for p in preguntas],
        metadata=[{"id_pregunta": p["id"]} for p in preguntas],
    )
    print(f"{nombre}: {len(preguntas)} preguntas cargadas")


if __name__ == "__main__":
    main()
