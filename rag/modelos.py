"""
Modelos de ORION en un solo lugar.

ORION_MODO=bedrock (por defecto): Nova 2 Lite y Titan V2 en Amazon Bedrock.
ORION_MODO=falso: modelos de mentira de LangChain, para probar el código sin AWS.

Uso:
    from modelos import obtener_llm, obtener_embeddings, modo_modelos
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Lee el .env de la raíz del repo (llaves de LangSmith, ORION_MODO, ORION_BUSQUEDA).
# Lo que ya esté definido en la terminal tiene prioridad sobre el .env.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

REGION = "us-east-1"
MODELO_LLM = "us.amazon.nova-2-lite-v1:0"           # perfil de inferencia (empieza con us.)
MODELO_EMBEDDINGS = "amazon.titan-embed-text-v2:0"  # 1024 dimensiones


def modo_modelos():
    """'bedrock' o 'falso'."""
    return os.getenv("ORION_MODO", "bedrock").strip().lower()


def _cliente_bedrock():
    """Cliente con reintentos: Bedrock limita las llamadas por minuto."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        "bedrock-runtime",
        region_name=REGION,
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
    )


def obtener_llm():
    if modo_modelos() == "falso":
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        return FakeListChatModel(responses=["(respuesta de prueba del modelo falso) [1]"])

    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(model=MODELO_LLM, client=_cliente_bedrock(), temperature=0, max_tokens=800)


def obtener_embeddings():
    if modo_modelos() == "falso":
        from langchain_core.embeddings import DeterministicFakeEmbedding

        return DeterministicFakeEmbedding(size=1024)

    from langchain_aws import BedrockEmbeddings

    return BedrockEmbeddings(model_id=MODELO_EMBEDDINGS, client=_cliente_bedrock())