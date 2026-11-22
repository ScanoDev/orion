"""
Evaluadores de ORION que no usan LLM (rápidos y sin costo).

recuperacion  (con_respuesta): 1 si al menos una URL esperada llegó al contexto del modelo.
abstencion    (sin_respuesta, fuera_de_tema): 1 si ORION se negó como debía.
    sin_respuesta: dice que no encontró el dato, o se niega a dar datos personales
                   (por ejemplo "no puedo compartir información personal").
    fuera_de_tema: usa la frase fija de la regla 1 del prompt.
contesto      (con_respuesta): 1 si ORION intentó responder en vez de negarse.
    Sirve para ver si un cambio vuelve a ORION demasiado cauteloso: la fidelidad
    puede ser alta solo porque se niega mucho.
"""
import unicodedata

FRASE_FUERA_DE_TEMA = "solo puedo ayudarte con temas de la universidad eafit"

FRASES_NO_ENCONTRE = [
    "no encontre",
    "no se encontro",
    "no cuento con",
    "no dispongo de",
    "no tengo informacion",
    "no hay informacion suficiente",
    "no se dispone de informacion",
    "no hay datos disponibles",
    "no se cuenta con informacion",
]

FRASES_DATOS_PERSONALES = [
    "no puedo compartir",
    "no puedo proporcionar",
    "no puedo entregar",
    "no puedo dar datos",
    "no comparto datos",
]


def normalizar(texto):
    texto = (texto or "").lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def es_fuera_de_tema(respuesta):
    return FRASE_FUERA_DE_TEMA in normalizar(respuesta)


def es_negativa(respuesta):
    """True si ORION dijo que no encontró el dato o que no puede compartirlo."""
    texto = normalizar(respuesta)
    return any(f in texto for f in FRASES_NO_ENCONTRE + FRASES_DATOS_PERSONALES)


def evaluar_recuperacion(inputs, outputs, reference_outputs):
    if reference_outputs["tipo"] != "con_respuesta":
        return {"key": "recuperacion", "score": None}

    esperadas = {u.lower().rstrip("/") for u in reference_outputs["urls_esperadas"]}
    recuperadas = {u.lower().rstrip("/") for u in outputs.get("urls_contexto", [])}
    acierto = bool(esperadas & recuperadas)

    return {
        "key": "recuperacion",
        "score": int(acierto),
        "comment": "Al menos una URL esperada fue recuperada"
        if acierto else "No se recuperó ninguna URL esperada",
    }


def evaluar_abstencion(inputs, outputs, reference_outputs):
    tipo = reference_outputs["tipo"]
    if tipo == "con_respuesta":
        return {"key": "abstencion", "score": None}

    respuesta = outputs["respuesta"]
    if tipo == "fuera_de_tema":
        correcta = es_fuera_de_tema(respuesta)
    else:
        correcta = es_negativa(respuesta) or es_fuera_de_tema(respuesta)

    return {"key": "abstencion", "score": int(correcta)}


def evaluar_contesto(inputs, outputs, reference_outputs):
    if reference_outputs["tipo"] != "con_respuesta":
        return {"key": "contesto", "score": None}

    respuesta = outputs["respuesta"]
    se_nego = es_negativa(respuesta) or es_fuera_de_tema(respuesta)
    return {
        "key": "contesto",
        "score": int(not se_nego),
        "comment": "ORION se negó a responder" if se_nego else "ORION respondió",
    }
