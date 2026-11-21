"""
Prueba de la cadena RAG v0 con preguntas de distintos tipos.

Correr desde la raíz del repo:  python rag/probar_cadena.py
Sin AWS (modelos falsos):        $env:ORION_MODO="falso"; python rag/probar_cadena.py
"""
import sys
import time

sys.path.insert(0, "rag")
from cadena import responder  # noqa: E402

PREGUNTAS = [
    "¿Qué necesito para inscribirme a un pregrado?",          # con respuesta
    "¿Qué becas hay para estudiantes de pregrado?",            # con respuesta
    "¿Cuál es el número de celular del rector?",               # no debería estar en el contexto
    "¿Quién ganó el mundial de fútbol de 2022?",               # fuera de tema
]

for pregunta in PREGUNTAS:
    inicio = time.time()
    r = responder(pregunta)
    print("=" * 80)
    print("PREGUNTA:", pregunta)
    print(f"RESPUESTA ({time.time() - inicio:.1f} s):\n{r['respuesta']}")
    print("FUENTES:")
    for f in r["fuentes"]:
        print(f"  [{f['n']}] {f['categoria']:<16} {f['titulo'][:55]} | {f['url']}")