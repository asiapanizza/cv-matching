import json
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, sha2, struct, udf
from pyspark.sql.types import StringType, StructType, StructField, MapType

# --- ENVIRONMENT SETUP ---
home_dir = os.path.expanduser("~")
conda_env_name = "talent_matching_linux"
python_path = os.path.join(home_dir, "miniconda3", "envs", conda_env_name, "bin", "python")
os.environ['PYSPARK_PYTHON'] = python_path
os.environ['PYSPARK_DRIVER_PYTHON'] = python_path
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 pyspark-shell'

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from download_model import model_validator

def process_row(row):
    """Router: Processes Jobs if 'category' exists, otherwise processes CVs"""
    import json
    # Job Imports
    from cleaning_logic.clean_postings import schematize_posting
    from job_processing.job_formatting import job_text 
    
    # CV Imports
    from cv_processing.cv_formatting import cv_formatter
    from cv_processing.string_cvs_processing import CVParserNLP
    from cv_processing.string_dataset_processing import CVParserDATASET
    from cv_processing.linkedin_pdf_processing import extract_cv_data
    from cv_processing.json_dataset_processing import reprocess_json

    doc_id = row['id']
    source = row['source']
    raw_data = row['raw_data']
    data_type = row['type']
    category = row.get('category')  # This is the differentiator

    try:
        # --- PATH A: JOB PROCESSING ---
        if category == "job":
            print(f"[{doc_id}] Processing as JOB")
            job_dict = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
            
            # Step 1: Schematize
            schema_data = schematize_posting(job_dict)
            # Step 2: Extract Text
            text_out = job_text(schema_data)
            
            return {
                "id": doc_id, "source": source, "is_job": "True",
                "schema_json": json.dumps(schema_data),
                "text_output": json.dumps({"text": text_out, "id": doc_id}),
                "personal_info": "{}", "error": None
            }

        # --- PATH B: CV PROCESSING ---
        else:
            print(f"[{doc_id}] Processing as CV")
            schema_data = None
            if source == "new_texts" or "txt" in data_type:
                schema_data = CVParserNLP().parse(raw_data)
            elif source == "linkedin_pdf" or "pdf" in data_type:
                schema_data = extract_cv_data(raw_data)
            elif source == "string_dataset":
                schema_data = CVParserDATASET().parse(raw_data)
            elif source == "json_dataset":
                schema_data = reprocess_json(json.loads(raw_data))
            
            if schema_data and "Error" not in schema_data:
                # Internal helper for CV formatting
                from cv_processing.cv_formatting import cv_formatter
                text_out, pers_info = cv_formatter(schema_data)
                
                # Add ID to dictionaries for consistency
                schema_data["id"] = doc_id
                pers_info["id"] = doc_id

                return {
                    "id": doc_id, "source": source, "is_job": "False",
                    "schema_json": json.dumps(schema_data),
                    "text_output": json.dumps({"text": text_out, "id": doc_id}),
                    "personal_info": json.dumps(pers_info),
                    "error": None
                }
            else:
                return {"id": doc_id, "source": source, "is_job": "False", "schema_json": "{}", 
                        "text_output": "{}", "personal_info": "{}", "error": "Parser failed"}

    except Exception as e:
        return {"id": doc_id, "source": source, "is_job": "error", "schema_json": "{}", 
                "text_output": "{}", "personal_info": "{}", "error": str(e)}

process_udf = udf(process_row, MapType(StringType(), StringType()))

def run_spark_etl(num_workers=4):
    model_validator()
    
    spark = SparkSession.builder \
        .master(f"local[{num_workers}]") \
        .appName("CV_Job_Unified_Processor") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")

    # 1. READ FROM KAFKA (Listening to both topics)
    df_raw = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "raw_resumes,raw_jobs") \
        .option("startingOffsets", "earliest") \
        .load()

    input_schema = StructType([
        StructField("id", StringType(), True),
        StructField("raw_data", StringType(), True),
        StructField("source", StringType(), True),
        StructField("type", StringType(), True),
        StructField("category", StringType(), True) # Important: identifies Jobs
    ])

    parsed_df = df_raw.selectExpr("CAST(value AS STRING)") \
        .select(from_json("value", input_schema).alias("data")) \
        .select("data.*")

    # 2. DEDUP
    df_unique = parsed_df.withColumn("content_hash", sha2(col("raw_data"), 256)) \
        .dropDuplicates(["content_hash"])

    # 3. PROCESS AND ROUTE
    def process_and_write(batch_df, batch_id):
        if batch_df.isEmpty(): return
        
        print(f"\n=== Processing Batch {batch_id} ===")
        # Apply UDF
        processed_df = batch_df.withColumn("res", process_udf(struct([col(c) for c in batch_df.columns])))
        results_df = processed_df.select("res.*").cache()

        # Filter for Successful results
        success_df = results_df.filter((col("error").isNull()) | (col("error") == ""))

        # --- ROUTE JOBS ---
        jobs_df = success_df.filter(col("is_job") == "True")
        if not jobs_df.isEmpty():
            print(f"Batch {batch_id}: Writing {jobs_df.count()} Jobs to Kafka...")
            # Job Schema
            jobs_df.select(col("id").alias("key"), 
                struct(col("id"), col("schema_json").alias("schema"), col("source")).alias("v")) \
                .selectExpr("key", "to_json(v) AS value") \
                .write.format("kafka").option("topic", "processed_schema_job") \
                .option("kafka.bootstrap.servers", "localhost:9092").save()
            # Job Text
            jobs_df.select(col("id").alias("key"), 
                struct(col("id"), col("text_output").alias("text"), col("source")).alias("v")) \
                .selectExpr("key", "to_json(v) AS value") \
                .write.format("kafka").option("topic", "processed_text_job") \
                .option("kafka.bootstrap.servers", "localhost:9092").save()

        # --- ROUTE CVS ---
        cvs_df = success_df.filter(col("is_job") == "False")
        if not cvs_df.isEmpty():
            print(f"Batch {batch_id}: Writing {cvs_df.count()} CVs to Kafka...")
            # CV Schema
            cvs_df.select(col("id").alias("key"), 
                struct(col("id"), col("schema_json").alias("schema"), col("source")).alias("v")) \
                .selectExpr("key", "to_json(v) AS value") \
                .write.format("kafka").option("topic", "processed_schema_cv") \
                .option("kafka.bootstrap.servers", "localhost:9092").save()
            # CV Text
            cvs_df.select(col("id").alias("key"), 
                struct(col("id"), col("text_output").alias("text"), col("source")).alias("v")) \
                .selectExpr("key", "to_json(v) AS value") \
                .write.format("kafka").option("topic", "processed_text_cv") \
                .option("kafka.bootstrap.servers", "localhost:9092").save()
            # CV Personal Info
            cvs_df.select(col("id").alias("key"), 
                struct(col("id"), col("personal_info").alias("info"), col("source")).alias("v")) \
                .selectExpr("key", "to_json(v) AS value") \
                .write.format("kafka").option("topic", "processed_personal_info_cv") \
                .option("kafka.bootstrap.servers", "localhost:9092").save()

        results_df.unpersist()

    query = df_unique.writeStream \
        .foreachBatch(process_and_write) \
        .option("checkpointLocation", "checkpoints_unified_etl") \
        .trigger(processingTime='10 seconds') \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    run_spark_etl()