import json
import time
import os
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

from confluent_kafka import Consumer, KafkaError
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# --- CONFIGURATION ---
@dataclass
class Config:
    KAFKA_BOOTSTRAP_SERVERS: str = 'localhost:9092'
    KAFKA_GROUP_ID: str = 'unified_cv_job_consumer_group'
    KAFKA_TOPICS: tuple = (
        "processed_schema_cv", 
        "processed_text_cv", 
        "processed_personal_info_cv",
        "processed_schema_job", # Added
        "processed_text_job"   # Added
    )
    BATCH_SIZE: int = 50
    BATCH_TIMEOUT: int = 30
    OUTPUT_DIR: str = "output_cv_processing"
    SPARK_APP_NAME: str = "Unified_Kafka_Consumer"
    SPARK_MASTER: str = "local[*]"

# --- SCHEMI SPARK ---
class Schemas:
    # ... (Keep existing CV schemas) ...
    SCHEMA_CV = StructType([
        StructField("id", StringType(), False),
        StructField("source", StringType(), True),
        StructField("education", StringType(), True),
        StructField("experience", StringType(), True),
        StructField("skills", StringType(), True)
    ])

    # NEW: Job Schema
    SCHEMA_JOB = StructType([
        StructField("id", StringType(), False),
        StructField("source", StringType(), True),
        StructField("title", StringType(), True),
        StructField("company", StringType(), True),
        StructField("description", StringType(), True),
        StructField("skills", StringType(), True)
    ])
    
    # Generic text schema (used for both CV and Job text)
    TEXT_DATA = StructType([
        StructField("id", StringType(), False),
        StructField("source", StringType(), True),
        StructField("text", StringType(), True),
        StructField("text_length", IntegerType(), True)
    ])
    
    PERSONAL_INFO = StructType([
        StructField("id", StringType(), False),
        StructField("source", StringType(), True),
        StructField("name", StringType(), True),
        StructField("email", StringType(), True),
        StructField("linkedin", StringType(), True)
    ])

# --- PARSING LOGIC ---
class DataParser:
    # ... (Keep existing CV parse methods) ...

    @staticmethod
    def parse_job_schema(key: str, data: Dict) -> Optional[Dict]:
        try:
            # Assumes data['schema'] contains the dict from schematize_posting
            raw_schema = data.get('schema', '{}')
            job_data = json.loads(raw_schema) if isinstance(raw_schema, str) else raw_schema
            
            return {
                'id': data.get('id', key),
                'source': data.get('source'),
                'title': job_data.get('title'),
                'company': job_data.get('company'),
                'description': job_data.get('description'),
                'skills': json.dumps(job_data.get('skills', []))
            }
        except Exception as e:
            logger.error(f"Error parsing job schema: {e}")
            return None

    @staticmethod
    def parse_text(key: str, data: Dict) -> Dict:
        text_content = data.get('text', '')
        return {
            'id': data.get('id', key),
            'source': data.get('source'),
            'text': text_content,
            'text_length': len(text_content)
        }

# --- PROCESSOR CORE ---
class UnifiedProcessor:
    def __init__(self):
        self._init_spark()
        self._init_kafka()
        # Expanded buffers
        self.buffers = {
            'schema_cv': [], 'text_cv': [], 'info_cv': [],
            'schema_job': [], 'text_job': []
        }
        self.last_flush_time = time.time()
        
    def _init_spark(self):
        self.spark = SparkSession.builder \
            .appName(Config.SPARK_APP_NAME) \
            .master(Config.SPARK_MASTER) \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()

    def _init_kafka(self):
        conf = {'bootstrap.servers': Config.KAFKA_BOOTSTRAP_SERVERS,
                'group.id': Config.KAFKA_GROUP_ID,
                'auto.offset.reset': 'earliest'}
        self.consumer = Consumer(conf)
        self.consumer.subscribe(list(Config.KAFKA_TOPICS))

    def _save_buffer(self, buffer_data: List[Dict], schema: StructType, folder_name: str):
        if not buffer_data: return
        df = self.spark.createDataFrame(buffer_data, schema=schema)
        output_path = os.path.join(Config.OUTPUT_DIR, folder_name)
        df.write.mode("append").partitionBy("id").parquet(output_path)
        logger.info(f"✓ Saved {len(buffer_data)} records to {folder_name}")

    def process_batch(self):
        if not any(self.buffers.values()): return
        
        # Save CVs
        self._save_buffer(self.buffers['schema_cv'], Schemas.SCHEMA_CV, "schema_cv")
        self._save_buffer(self.buffers['text_cv'], Schemas.TEXT_DATA, "text_cv")
        self._save_buffer(self.buffers['info_cv'], Schemas.PERSONAL_INFO, "info_cv")
        
        # Save JOBS
        self._save_buffer(self.buffers['schema_job'], Schemas.SCHEMA_JOB, "job_schema")
        self._save_buffer(self.buffers['text_job'], Schemas.TEXT_DATA, "job_text")

        for k in self.buffers: self.buffers[k] = []
        self.last_flush_time = time.time()

    def _handle_message(self, msg):
        topic = msg.topic()
        key = msg.key().decode('utf-8') if msg.key() else "None"
        try:
            value = json.loads(msg.value().decode('utf-8'))
            
            # ROUTING
            if topic == "processed_schema_cv":
                res = DataParser.parse_schema(key, value)
                if res: self.buffers['schema_cv'].append(res)
            elif topic == "processed_text_cv":
                self.buffers['text_cv'].append(DataParser.parse_text(key, value))
            elif topic == "processed_personal_info_cv":
                self.buffers['info_cv'].append(DataParser.parse_personal_info(key, value))
            
            # JOB ROUTING
            elif topic == "processed_schema_job":
                res = DataParser.parse_job_schema(key, value)
                if res: self.buffers['schema_job'].append(res)
            elif topic == "processed_text_job":
                self.buffers['text_job'].append(DataParser.parse_text(key, value))
                
        except Exception as e:
            logger.error(f"Error in handle_message: {e}")

    def run(self):
        try:
            while True:
                msg = self.consumer.poll(timeout=1.0)
                if (time.time() - self.last_flush_time) >= Config.BATCH_TIMEOUT:
                    self.process_batch()
                if msg is None: continue
                if msg.error(): continue
                
                self._handle_message(msg)
                
                if max(len(b) for b in self.buffers.values()) >= Config.BATCH_SIZE:
                    self.process_batch()
        except KeyboardInterrupt:
            self.process_batch()
        finally:
            self.consumer.close()
            self.spark.stop()

if __name__ == "__main__":
    UnifiedProcessor().run()