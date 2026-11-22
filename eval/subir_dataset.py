
import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

# Cargar variables de entorno
load_dotenv(override=True)

client = Client()

# Leer las preguntas
ruta = Path(__file__).parent / "preguntas_orion_actualizadas.json"

with open(ruta, "r", encoding="utf-8") as archivo:
    preguntas = json.load(archivo)

# Nombre del dataset
nombre_dataset = "ORION-PREGUNTAS-EVALUACION"

# Evitar duplicados
if client.has_dataset(dataset_name=nombre_dataset):
    print(f"El dataset '{nombre_dataset}' ya existe.")
    print("No se crearon registros duplicados.")
    raise SystemExit(0)

# Crear dataset
dataset = client.create_dataset(
    dataset_name=nombre_dataset,
    description=(
        "50 preguntas para evaluar el sistema RAG ORION "
        "de la Universidad EAFIT."
    ),
)

# Preparar ejemplos
inputs = []
outputs = []
metadata = []

for item in preguntas:
    inputs.append({
        "pregunta": item["pregunta"]
    })

    outputs.append({
        "tipo": item["tipo"],
        "urls_esperadas": item["urls_esperadas"]
    })

    metadata.append({
        "id_pregunta": item["id"]
    })

# Subir ejemplos
client.create_examples(
    dataset_id=dataset.id,
    inputs=inputs,
    outputs=outputs,
    metadata=metadata,
)

print("Dataset creado correctamente.")
print("Nombre:", nombre_dataset)
print("Preguntas cargadas:", len(preguntas))
