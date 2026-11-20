from langchain_aws import BedrockEmbeddings, ChatBedrockConverse

region = "us-east-1"
modelo_llm = "us.amazon.nova-2-lite-v1:0"
modelo_embeddings = "amazon.titan-embed-text-v2:0"

llm = ChatBedrockConverse(
    model=modelo_llm,
    region_name=region,
    temperature=0
)
response = llm.invoke("En una sola frase: ¿Qué es la Universidad EAFIT?")
print("LLM:", response)
print("Tokens usados:", response.usage_metadata)

embeddings = BedrockEmbeddings(
    model_id=modelo_embeddings, 
    region_name=region
)
vector = embeddings.embed_query("¿Cuanto cuesta la matrícula de ingeniería de sistemas?")
print("Embedding:", len(vector), "dimensiones. Primeros 5 valores:", [round(x,4) for x in vector[:5]])
