
import unicodedata


def normalizar(texto):
    texto = texto.lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def evaluar_recuperacion(inputs, outputs, reference_outputs):
    if reference_outputs["tipo"] != "con_respuesta":
        return {"key": "recuperacion", "score": None}

    esperadas = {
        u.lower().rstrip("/")
        for u in reference_outputs["urls_esperadas"]
    }
    recuperadas = {
        u.lower().rstrip("/")
        for u in outputs.get("urls_contexto", [])
    }

    acierto = bool(esperadas.intersection(recuperadas))

    return {
        "key": "recuperacion",
        "score": int(acierto),
        "comment": "Al menos una URL esperada fue recuperada"
        if acierto else "No se recuperó ninguna URL esperada",
    }



def evaluar_abstencion(inputs, outputs, reference_outputs):
    tipo = reference_outputs["tipo"]

    if tipo == "con_respuesta":
        return {
            "key": "abstencion",
            "score": None,
        }

    respuesta = normalizar(outputs["respuesta"])

    if tipo == "fuera_de_tema":
        correcta = (
            "solo puedo ayudarte con temas de la universidad eafit"
            in respuesta
        )
    else:
        expresiones = [
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

        correcta = any(
            expresion in respuesta
            for expresion in expresiones
        )

    return {
        "key": "abstencion",
        "score": int(correcta),
    }
