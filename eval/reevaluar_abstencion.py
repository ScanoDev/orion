
from pathlib import Path
import sys

from dotenv import load_dotenv
from langsmith import Client

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

load_dotenv(RAIZ / ".env", override=True)

from eval.evaluadores import evaluar_abstencion

client = Client()

EXPERIMENTO_ID = "f60461ed-aad9-48f3-99b5-9d34ce868d66"

# Obtener las ejecuciones del experimento.
runs = list(
    client.list_runs(
        project_id=EXPERIMENTO_ID,
        is_root=True,
    )
)

print(f"Ejecuciones encontradas: {len(runs)}")

actualizados = 0
omitidos = 0

for run in runs:
    if not run.outputs:
        omitidos += 1
        continue

    # Obtener el ejemplo del dataset asociado al run.
    if not run.reference_example_id:
        omitidos += 1
        continue

    ejemplo = client.read_example(run.reference_example_id)

    if not ejemplo.outputs:
        omitidos += 1
        continue

    # Solo reevaluar preguntas de abstención.
    tipo = ejemplo.outputs.get("tipo")

    if tipo not in ("sin_respuesta", "fuera_de_tema"):
        omitidos += 1
        continue

    evaluacion = evaluar_abstencion(
        inputs=ejemplo.inputs,
        outputs=run.outputs,
        reference_outputs=ejemplo.outputs,
    )

    # Guardar la puntuación corregida como nueva métrica.
    # Usamos otra clave para conservar la evaluación original.
    client.create_feedback(
        run_id=run.id,
        key="abstencion_v2",
        score=evaluacion["score"],
        comment="Reevaluación con detección de 'no se encontró'.",
    )

    actualizados += 1

print("\nReevaluación finalizada")
print("Casos reevaluados:", actualizados)
print("Casos omitidos:", omitidos)
