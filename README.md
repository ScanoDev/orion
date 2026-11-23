# ORION

Asistente virtual de la Universidad EAFIT. Responde preguntas sobre el sitio de EAFIT con un RAG (búsqueda híbrida BM25 + vectorial, embeddings Titan V2 en Chroma y Nova 2 Lite en Amazon Bedrock) y cita las páginas de donde saca cada dato.

## Correr la app (índice v3)

Requisitos: Python 3.12 y credenciales de AWS con acceso a Bedrock y al bucket `orion-eafit-2026`.

```bash
pip install -r requirements.txt
cp .env.example .env                  # y completa lo que haga falta
python rag/descargar_indice.py        # baja el índice v3 desde S3 (v3 es el valor por defecto)
streamlit run app/app.py              # con Bedrock real
```

Modo de prueba, sin llamar a Bedrock (el modelo es de mentira):

```bash
ORION_MODO=falso streamlit run app/app.py
```

El índice de Chroma queda en `vectorstore/orion_v3/`, y `chunks.parquet` y `manifiesto.json` en `data/processed/v3/`. Para volver a bajarlo usa `--forzar`; para comparar con la línea base, `python rag/descargar_indice.py v2`.

## Variables de entorno

Ver `.env.example`.

| Variable | Valor | Para qué |
|---|---|---|
| `ORION_MODO` | `bedrock` | `bedrock` usa Nova 2 Lite y Titan V2; `falso` usa modelos de mentira, sin AWS |
| `ORION_BUSQUEDA` | `hibrido` | Tipo de búsqueda de la cadena |
| `ORION_INDICE` | `v3` | Versión del índice que baja el contenedor al arrancar |
| `AWS_DEFAULT_REGION` | `us-east-1` | Región de Bedrock y S3 |
| `LANGSMITH_*` | (opcional) | Trazas en LangSmith: `LANGSMITH_TRACING=true` y `LANGSMITH_API_KEY` |

## Docker

```bash
docker build -t orion .
docker run --rm -p 8501:8501 -v "$HOME/.aws:/root/.aws:ro" orion
```

Abrir http://localhost:8501. El índice no va dentro de la imagen: el contenedor lo baja de S3 al arrancar (`docker/arrancar.sh`) con las credenciales de AWS. En la EC2 usa el rol de la instancia (ver `infra/orion_ec2.yaml`).

## Reconstruir el índice

Cuando hay una versión nueva de `oro`, el índice se arma reusando los embeddings de la versión anterior: solo se calculan con Titan los chunks nuevos o cambiados.

```bash
python rag/descargar_indice.py v2                                   # índice anterior
aws s3 cp s3://orion-eafit-2026/publicado/oro_v3/ data/oro_v3/ --recursive
python rag/indexar_incremental.py --anterior v2 --nueva v3
python rag/publicar.py v3                                           # sube a s3://orion-eafit-2026/indices/v3/
```

## Limitaciones conocidas

- Las tarifas y la financiación están en documentos embebidos (Issuu y PDF) que no se extrajeron.