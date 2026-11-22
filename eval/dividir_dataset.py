
import json
import random
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env", override=True)

client = Client()

ruta = RAIZ / "eval" / "preguntas_orion_actualizadas.json"

with open(ruta, encoding="utf-8") as f:
    preguntas = json.load(f)

grupos = defaultdict(list)

for pregunta in preguntas:
    grupos[pregunta["tipo"]].append(pregunta)

random.seed(42)

# Cantidades reservadas para evaluación final
cantidades_final = {
    "con_respuesta": 10,
    "sin_respuesta": 3,
    "fuera_de_tema": 2,
}

desarrollo = []
final = []

for tipo, cantidad in cantidades_final.items():
    elementos = grupos[tipo].copy()
    random.shuffle(elementos)

    final.extend(elementos[:cantidad])
    desarrollo.extend(elementos[cantidad:])

def subir_dataset(nombre, preguntas):
    if client.has_dataset(dataset_name=nombre):
        print(f"Ya existe: {nombre}. No se modifica.")
        return

    dataset = client.create_dataset(
        dataset_name=nombre,
        description="Evaluación RAG de ORION - Universidad EAFIT",
    )

    client.create_examples(
        dataset_id=dataset.id,
        inputs=[
            {"pregunta": p["pregunta"]}
            for p in preguntas
        ],
        outputs=[
            {
                "tipo": p["tipo"],
                "urls_esperadas": p["urls_esperadas"],
            }
            for p in preguntas
        ],
        metadata=[
            {"id_pregunta": p["id"]}
            for p in preguntas
        ],
    )

    print(f"{nombre}: {len(preguntas)} preguntas cargadas")

subir_dataset("ORION-DESARROLLO", desarrollo)
subir_dataset("ORION-EVALUACION-FINAL", final)

print("\nDistribución:")
print("Desarrollo:", len(desarrollo))
print("Evaluación final:", len(final))
