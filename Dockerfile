# Imagen de la app de ORION (Streamlit + cadena RAG).
# El índice NO va dentro de la imagen: se baja de S3 al arrancar (docker/arrancar.sh),
# con las credenciales del rol de la EC2. Así la imagen no tiene datos ni llaves.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AWS_DEFAULT_REGION=us-east-1 \
    ORION_MODO=bedrock \
    ORION_BUSQUEDA=hibrido \
    ORION_INDICE=v3

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# Por si el script llega con saltos de línea de Windows (CRLF), que rompen /bin/sh
RUN sed -i "s/\r$//" docker/arrancar.sh && chmod +x docker/arrancar.sh

EXPOSE 8501
CMD ["docker/arrancar.sh"]