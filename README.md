## sdfsd
las tarifas y la financiación están en documentos embebidos (Issuu y PDF) que no se extrajeron

## Correr la app

```bash
pip install -r requirements.txt
python rag/descargar_indice.py          # baja el índice v2 desde S3 (necesita credenciales de AWS)
aws s3 cp s3://orion-eafit-2026/publicado/oro_v2/ data/oro/ --recursive
ORION_MODO=falso streamlit run app/app.py   # sin AWS, con modelo de mentira
streamlit run app/app.py                    # con Bedrock real
```