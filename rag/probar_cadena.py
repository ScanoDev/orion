"""
Prueba de la cadena RAG con preguntas de distintos tipos.

Correr desde la raíz del repo:
    python rag/probar_cadena.py              # modo del .env (por defecto hibrido)
    python rag/probar_cadena.py bm25         # o vector, o hibrido
    python rag/probar_cadena.py bm25 > rag/resultados/v03_bm25.txt
Sin AWS (modelos falsos), en PowerShell:
    $env:ORION_MODO="falso"; python rag/probar_cadena.py
"""
import sys
import time

# Windows escribe en cp1252 al redirigir con ">", y se rompe con tildes o símbolos como "→"
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, "rag")
from cadena import MODOS, responder, version_cadena  # noqa: E402

PREGUNTAS = [
    "¿Qué necesito para inscribirme a un pregrado?",          # con respuesta
    "¿Qué becas hay para estudiantes de pregrado?",            # con respuesta
    "¿Cuál es el número de celular del rector?",               # datos personales: debe negarse
    "¿Quién ganó el mundial de fútbol de 2022?",               # fuera de tema
]

modo = sys.argv[1] if len(sys.argv) > 1 else None
if modo and modo not in MODOS:
    raise SystemExit(f"Modo {modo!r} no existe. Usa uno de: {', '.join(MODOS)}")

print(f"VERSIÓN: {version_cadena(modo)}")
for pregunta in PREGUNTAS:
    inicio = time.time()
    r = responder(pregunta, modo=modo)
    print("=" * 80)
    print("PREGUNTA:", pregunta)
    print(f"RESPUESTA ({time.time() - inicio:.1f} s):\n{r['respuesta']}")
    print("FUENTES:")
    for f in r["fuentes"]:
        print(f"  [{f['n']}] {f['categoria']:<16} {f['titulo'][:55]} | {f['url']}")
    print("PÁGINAS QUE VIO EL MODELO:")
    for url in r["urls_contexto"]:
        print("  ", url)