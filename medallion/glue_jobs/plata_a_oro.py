"""
Trabajo de Glue: plata (tabla Iceberg) a oro (tabla Iceberg + copia publicada).

Job details:
    Glue version 5.x, Spark, Python 3, G.1X, 2 workers
    Job parameters: --datalake-formats = iceberg
                    --BUCKET = nombre del bucket de ORION
                    --VERSION = versión que se publica (v1, v2...)
    Advanced properties, Python library path: s3://<BUCKET>/codigo/oro_documentos.py

Salidas:
    glue_catalog.orion_db.oro_documentos     una fila por página, con categoría
    s3://<BUCKET>/publicado/oro_<VERSION>/   copia fija en un solo parquet para el RAG
    s3://<BUCKET>/oro/conteos/               CSV con páginas por categoría e idioma
"""
import sys
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from oro_documentos import enriquecer_documentos

args = getResolvedOptions(sys.argv, ["JOB_NAME", "BUCKET", "VERSION"])
BUCKET = args["BUCKET"]
VERSION = args["VERSION"]

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

# 1. Leer plata
plata = spark.table("glue_catalog.orion_db.plata_documentos")

# 2. Agregar sección, categoría y título limpio
oro = enriquecer_documentos(plata).cache()

# 3. Guardar oro como Iceberg (se reemplaza en cada corrida; Iceberg guarda las versiones)
oro.writeTo("glue_catalog.orion_db.oro_documentos").using("iceberg").createOrReplace()

# 4. Publicar una copia fija para el RAG. Nunca leer los archivos internos de Iceberg
#    directamente: ahí quedan también los de versiones anteriores.
oro.coalesce(1).write.mode("overwrite").parquet(f"s3://{BUCKET}/publicado/oro_{VERSION}/")

# 5. Conteos por categoría e idioma
conteos = (
    oro.groupBy("categoria", "idioma").count()
    .withColumnRenamed("count", "paginas")
    .orderBy("categoria", "idioma")
)
conteos.show(50, truncate=False)
conteos.coalesce(1).write.mode("overwrite").option("header", True).csv(f"s3://{BUCKET}/oro/conteos/")

job.commit()