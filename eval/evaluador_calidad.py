
import json
import re

from langchain_core.messages import HumanMessage
from rag.modelos import obtener_llm


def evaluar_calidad(inputs, outputs, reference_outputs):
    if reference_outputs["tipo"] != "con_respuesta":
        return {
            "key": "fidelidad",
            "score": None,
        }

    pregunta = inputs["pregunta"]
    respuesta = outputs["respuesta"]
    contexto = outputs["contexto"]

    prompt = f"""
Eres un evaluador de respuestas RAG.

Determina si la respuesta está respaldada por el contexto.

Criterios:
1. Todas las afirmaciones factuales deben estar respaldadas.
2. No se permite inventar información.
3. No uses conocimiento externo.
4. Devuelve únicamente un objeto JSON válido.

PREGUNTA:
{pregunta}

CONTEXTO:
{contexto}

RESPUESTA:
{respuesta}

Formato de salida:
{{
    "score": 0,
    "justificacion": "Explicación breve"
}}

Utiliza score 1 si la respuesta está respaldada por el
contexto y 0 en caso contrario.
"""

    llm = obtener_llm()
    resultado = llm.invoke([
        HumanMessage(content=prompt)
    ])

    texto = resultado.content

    if not isinstance(texto, str):
        texto = str(texto)

    coincidencia = re.search(
        r"\{.*\}",
        texto,
        re.DOTALL,
    )

    if not coincidencia:
        raise ValueError(
            f"El juez no devolvió JSON: {texto}"
        )

    datos = json.loads(coincidencia.group())

    score = int(datos["score"])

    if score not in (0, 1):
        raise ValueError("El score debe ser 0 o 1")

    return {
        "key": "fidelidad",
        "score": score,
        "comment": datos["justificacion"],
    }
