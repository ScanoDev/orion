#!/bin/sh
# Arranque del contenedor: baja el índice de S3 (si no está) y levanta Streamlit.
set -e
python rag/descargar_indice.py "${ORION_INDICE:-v3}"
exec streamlit run app/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
