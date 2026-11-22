
import os
from dotenv import load_dotenv
from langsmith import Client

# Cargar el .env de la raíz del proyecto
load_dotenv(override=True)

api_key = os.getenv("LANGSMITH_API_KEY")

if not api_key:
    raise ValueError("No se encontró LANGSMITH_API_KEY en .env")

print("API Key encontrada:", api_key[:8] + "...")
print("Longitud:", len(api_key))

try:
    client = Client(api_key=api_key)

    datasets = list(client.list_datasets(limit=5))

    print("\nConexión exitosa con LangSmith")

    for dataset in datasets:
        print("-", dataset.name)

except Exception as e:
    print("\nError al conectar con LangSmith:")
    print(type(e).__name__)
    print(str(e))
