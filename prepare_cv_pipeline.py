import os, sys
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(script_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from give_inputs import give_inputs, select_integer
from run_encoder import encoder
from ingest_cv.cv_spark_pipeline.cv_spark_producer import ingest_data
import subprocess
from ingest_cv.cv_spark_pipeline.cv_spark_ingestion import run_spark_etl
from ingest_cv.cv_spark_pipeline.cv_spark_consumer import run_consumer

import json
import numpy as np
import pandas as pd

BATCH = [
    {"path": "ingest_cv/master_resumes.jsonl", "source": "json_dataset", "type": "jsonl"},
    {"path": "ingest_cv/Resume.csv", "source": "string_dataset", "type": "csv", "col" : "Resume_str"},
    {"path": "data_jobs.csv", "source": "jobs_dataset", "type": "csv"}
]

import subprocess
import platform
import sys
import os

def run_in_terminal(script_name):
    os_name = platform.system().lower()
    python_exe = sys.executable
    script_path = os.path.abspath(script_name)
    project_root = os.path.dirname(os.path.abspath(__file__))

    if os_name == 'windows':
        subprocess.Popen(['start', 'cmd', '/k', f'"{python_exe}" "{script_path}"'], shell=True)
    
    elif os_name == 'linux':
        # Costruiamo un comando che imposta il PYTHONPATH e avvia lo script
        # Aggiungiamo 'read' alla fine per tenere il terminale aperto se lo script crasha
        command = f'export PYTHONPATH="{project_root}"; "{python_exe}" "{script_path}"; echo "---"; echo "Processo terminato. Premi Invio per chiudere."; read'
        try:
            subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', command])
        except FileNotFoundError:
            subprocess.Popen(['xterm', '-e', f'bash -c \'{command}\''])
            
    elif os_name == 'darwin':
        command = f'export PYTHONPATH="{project_root}"; "{python_exe}" "{script_path}"'
        subprocess.Popen(['osascript', '-e', f'tell application "Terminal" to do script "{command}"'])
        print(f"Sistema operativo {os_name} non supportato.")



# runs for the first time with all the datasets to build up the database
def create_parquet():
    spark_etl_script = "ingest_cv/cv_spark_pipeline/cv_spark_ingestion.py"
    consumer_script = "ingest_cv/cv_spark_pipeline/cv_spark_consumer.py"
    # builds up the cv database
    print("Pipeline starting...")
    ingest_data(BATCH)
    print("Two separate terminals for Spark Streaming are about to open, please do not close them")
    run_in_terminal(spark_etl_script)
    run_in_terminal(consumer_script)


if __name__  == "__main__":
    create_parquet()