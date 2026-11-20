import boto3
from typing import Iterator

import config


def get_s3_client():
    return boto3.client("s3", region_name=config.AWS_REGION)


def list_objects(prefix: str = config.S3_PREFIX) -> Iterator[dict]:
    """Recorre todos los objetos del bucket (paginado) y los entrega uno a uno."""
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=config.S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("/"):  # saltar "carpetas" vacías
                continue
            yield {
                "key": obj["Key"],
                "size": obj["Size"],
                "last_modified": obj["LastModified"],
            }

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Parte el texto en pedazos de ~chunk_size caracteres, con solape entre ellos."""
    if overlap >= chunk_size:
        raise ValueError("overlap debe ser menor que chunk_size")

    text = " ".join(text.split())  # normaliza espacios y saltos de línea
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]
        if chunk:
            chunks.append(chunk)
    return chunks


if __name__ == "__main__":
    with open("data/raw/prueba.txt", encoding="utf-8") as f:
        texto = f.read()

    chunks = chunk_text(texto)
    print(f"{len(chunks)} chunks")
    print(chunks[0][:200])