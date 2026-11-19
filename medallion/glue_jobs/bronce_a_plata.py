"""
Trabajo de Glue: bronce (JSON de Firecrawl en S3) a plata (tablas Iceberg).

Job details:
    Glue version 5.x, Spark, Python 3, G.1X, 2 workers
    Job parameters: --datalake-formats = iceberg
                    --BUCKET = nombre del bucket de ORION
    Advanced properties, Python library path: s3://<BUCKET>/codigo/limpieza_paginas.py

Salidas:
    glue_catalog.orion_db.plata_documentos   una fila por página limpia
    glue_catalog.orion_db.plata_rechazos     páginas descartadas con su motivo
    s3://<BUCKET>/plata/conteos/             CSV con el cuadre de conteos
"""
import sys
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from limpieza_paginas import leer_bronce, limpiar_paginas

args = getResolvedOptions(sys.argv, ["JOB_NAME", "BUCKET"])
BUCKET = args["BUCKET"]

conf = SparkConf()
conf.set("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
conf.set("spark.sql.catalog.glue_catalog", "org.apache.iceberg.spark.SparkCatalog")
conf.set("spark.sql.catalog.glue_catalog.warehouse", f"s3://{BUCKET}/warehouse/")
conf.set("spark.sql.catalog.glue_catalog.catalog-impl", "org.apache.iceberg.aws.glue.GlueCatalog")
conf.set("spark.sql.catalog.glue_catalog.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")

sc = SparkContext(conf=conf)
glue = GlueContext(sc)
spark = glue.spark_session
job = Job(glue)
job.init(args["JOB_NAME"], args)

# 1. Leer bronce
paginas = leer_bronce(spark, f"s3://{BUCKET}/bronce/").cache()

# 2. Limpiar
documentos, rechazadas, conteos = limpiar_paginas(paginas)

for concepto, n in conteos:
    print(f"{concepto}: {n:,}")

# 3. Guardar plata como Iceberg (se reemplaza en cada corrida)
documentos.writeTo("glue_catalog.orion_db.plata_documentos").using("iceberg").createOrReplace()
rechazadas.writeTo("glue_catalog.orion_db.plata_rechazos").using("iceberg").createOrReplace()

# 4. Conteos en S3 para verlos sin entrar a los logs
(
    spark.createDataFrame(conteos, ["concepto", "cantidad"])
    .coalesce(1)
    .write.mode("overwrite")
    .option("header", True)
    .csv(f"s3://{BUCKET}/plata/conteos/")
)

job.commit()