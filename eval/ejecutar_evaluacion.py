"""
Corre la evaluación de ORION en LangSmith, para uno o varios modos de búsqueda.

Uso (desde la raíz del repo):
    python eval/ejecutar_evaluacion.py                                   # desarrollo, modo hibrido
    python eval/ejecutar_evaluacion.py --modos bm25 vector hibrido       # los tres modos, para la tabla
    python eval/ejecutar_evaluacion.py --dataset parafraseadas --modos bm25 vector hibrido
    python eval/ejecutar_evaluacion.py --dataset ciegas --modos bm25 hibrido --confirmar-final
    python eval/ejecutar_evaluacion.py --local eval/preguntas_parafraseadas.json   # sin subir nada

Cada modo queda como un experimento aparte en LangSmith, con nombre
"<version>-<dataset>" (por ejemplo "v0.3-hibrido-desarrollo"), y además se guarda
un CSV en eval/resultados/ con una fila por pregunta, listo para las gráficas.

ORION-CIEGAS es la prueba ciega: preguntas nuevas que nadie ha corrido. Se corre una sola
vez, con la versión final, y por eso pide --confirmar-final.
"""
import argparse
import json
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
load_dotenv(RAIZ / ".env", override=True)

from langsmith import Client, evaluate  # noqa: E402
from langsmith.schemas import Example  # noqa: E402

from eval.evaluador_calidad import evaluar_calidad  # noqa: E402
from eval.evaluadores import evaluar_abstencion, evaluar_contesto, evaluar_recuperacion  # noqa: E402
from rag.cadena import MODOS, responder, version_cadena  # noqa: E402

DATASETS = {
    "desarrollo": "ORION-DESARROLLO",
    "parafraseadas": "ORION-PARAFRASEADAS",
    "final": "ORION-EVALUACION-FINAL",   # las 15 ya usadas: línea base de v0.2
    "ciegas": "ORION-CIEGAS",             # preguntas nuevas: prueba ciega, se corre una sola vez
}
CARPETA_RESULTADOS = RAIZ / "eval" / "resultados"
EVALUADORES = [evaluar_recuperacion, evaluar_abstencion, evaluar_contesto, evaluar_calidad]
METRICAS = ["recuperacion", "fidelidad", "contesto", "abstencion"]


def crear_objetivo(modo):
    """La función que LangSmith llama con cada pregunta del dataset."""
    def ejecutar_orion(inputs):
        r = responder(inputs["pregunta"], modo=modo)
        return {
            "respuesta": r["respuesta"],
            "fuentes": r["fuentes"],
            "urls_contexto": r["urls_contexto"],
            "chunk_ids": r["chunk_ids"],
            "contexto": r["contexto"],
        }
    ejecutar_orion.__name__ = f"orion_{modo}"
    return ejecutar_orion


def ejemplos_locales(ruta):
    """Lee un JSON con el formato de preguntas_orion_actualizadas.json (para probar sin subir)."""
    preguntas = json.loads(Path(ruta).read_text(encoding="utf-8"))
    dataset_id = uuid.uuid4()
    return [
        Example(
            id=uuid.uuid4(),
            dataset_id=dataset_id,
            inputs={"pregunta": p["pregunta"]},
            outputs={"tipo": p["tipo"], "urls_esperadas": p["urls_esperadas"]},
            metadata={"id_pregunta": p["id"]},
        )
        for p in preguntas
    ]


def guardar_y_resumir(resultados, nombre):
    df = resultados.to_pandas()
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA_RESULTADOS / f"{nombre}.csv"
    df.to_csv(ruta, index=False, encoding="utf-8-sig")

    print(f"\n{nombre}  ({len(df)} preguntas)  ->  {ruta.relative_to(RAIZ)}")
    for metrica in METRICAS:
        col = f"feedback.{metrica}"
        if col in df.columns and df[col].notna().any():
            valores = df[col].dropna()
            print(f"  {metrica:<13} {valores.mean():.2f}  ({int(valores.sum())} de {len(valores)})")
    if "execution_time" in df.columns:
        t = df["execution_time"].dropna()
        print(f"  {'latencia':<13} P50 {t.quantile(0.5):.2f} s | P95 {t.quantile(0.95):.2f} s")


def main():
    parser = argparse.ArgumentParser(description="Evaluación de ORION")
    parser.add_argument("--dataset", choices=DATASETS, default="desarrollo")
    parser.add_argument("--modos", nargs="+", choices=MODOS, default=["hibrido"])
    parser.add_argument("--confirmar-final", action="store_true",
                        help="Obligatorio para el dataset ciegas (prueba ciega, se corre una vez)")
    parser.add_argument("--local", metavar="JSON",
                        help="Evalúa preguntas de un JSON sin subir resultados a LangSmith")
    args = parser.parse_args()

    if args.dataset == "ciegas" and not args.confirmar_final and not args.local:
        raise SystemExit("ORION-CIEGAS es la prueba ciega. Si de verdad es la corrida final, agrega --confirmar-final")

    if args.local:
        datos, etiqueta, subir, cliente = ejemplos_locales(args.local), "local", False, None
    else:
        datos, etiqueta, subir, cliente = DATASETS[args.dataset], args.dataset, True, Client()

    for modo in args.modos:
        nombre = f"{version_cadena(modo)}-{etiqueta}"
        print(f"\n=== {nombre} ===")
        resultados = evaluate(
            crear_objetivo(modo),
            data=datos,
            evaluators=EVALUADORES,
            experiment_prefix=nombre,
            description=f"ORION {version_cadena(modo)} sobre {etiqueta}",
            metadata={"version_cadena": version_cadena(modo), "modo_busqueda": modo, "dataset": etiqueta},
            max_concurrency=1,
            client=cliente,
            upload_results=subir,
        )
        guardar_y_resumir(resultados, nombre)


if __name__ == "__main__":
    main()
