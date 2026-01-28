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
from faiss_matching import matching
import json
import numpy as np
import pandas as pd

BATCH = [
    {"path": "ingest_cv/master_resumes.jsonl", "source": "json_dataset", "type": "jsonl"},
    {"path": "ingest_cv/Resume.csv", "source": "string_dataset", "type": "csv", "col" : "Resume_str"}
]

import subprocess
import platform
import sys
import os

def run_in_terminal(script_name):
    os_name = platform.system().lower()
    python_exe = sys.executable
    if os_name == 'windows':
        subprocess.Popen(['start', 'cmd', '/k', f'"{python_exe}" {script_name}'], shell=True)  
    elif os_name == 'linux':
        try:
            subprocess.Popen(['gnome-terminal', '--', python_exe, script_name])
        except FileNotFoundError:
            subprocess.Popen(['xterm', '-e', python_exe, script_name])     
    elif os_name == 'darwin':
        script_path = os.path.abspath(script_name)
        command = f'tell application "Terminal" to do script "{python_exe} {script_path}"'
        subprocess.Popen(['osascript', '-e', command])
        
    else:
        print(f"Sistema operativo {os_name} non supportato.")



# runs for the first time with all the datasets to build up the database
def order():
    text_path = ".../cv-job-matcher-project/ingest_cv/cv_spark_pipeline/output_cv_processing/text_cv/"
    schema_path = ".../cv-job-matcher-project/ingest_cv/cv_spark_pipeline/output_cv_processing/schema_cv/"
    info_path = ".../cv-job-matcher-project/ingest_cv/cv_spark_pipeline/output_cv_processing/info_cv/"
    job_schema_path = ".../cv-job-matcher-project/ingest_cv/cv_spark_pipeline/output_cv_processing/schema_job/"
    job_text_path = ".../cv-job-matcher-project/ingest_cv/cv_spark_pipeline/output_cv_processing/text_cv/"
    text_elements = os.listdir(text_path)
    data_frames = []

    # collects all the processed texts and saves the text database
    for element in text_elements:
        full_element_path = os.path.join(text_path, element)
        if os.path.isdir(full_element_path) and element.startswith("id=A"):
            try:
                cv = pd.read_parquet(full_element_path)
                data_frames.append(cv)
                print(f"{element} has been read correctly")
            except Exception as e:
                print(f"Error in {element}: {e}")
    if data_frames:
        cv_text_df = pd.concat(data_frames, ignore_index=True)
    else:
        cv_text_df = pd.DataFrame(columns=['id', 'text'])
    cv_text_df.to_parquet("cv_datasets/cv_text")

    # saves and encodes the cv embedding files

    encoder(cv_text_df, "cv", is_query= False)
    encoder(cv_text_df, "cv", is_query= True)

    #collects all the processed cv schemas and saves the schema database

    schema_elements = os.listdir(schema_path)
    cv_schema = []
    for element in schema_elements:
        full_element_path = os.path.join(schema_path, element)
        if os.path.isdir(full_element_path) and element.startswith("id=A"):
            for file_name in os.listdir(full_element_path):
                if file_name.endswith(".json"):
                    file_path = os.path.join(full_element_path, file_name)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            data = json.loads(line)
                            cv_schema.append(data)
    with open("cv_datasets/cv_schema", "w") as f:
        json.dump(cv_schema,f)

    #collects all the processed personal infos and saves the personal info database
    
    data_frames = []
    info_elements = os.listdir(info_path)
    for element in info_elements:
        full_element_path = os.path.join(text_path, element)
        if os.path.isdir(full_element_path) and element.startswith("id=A"):
            try:
                cv = pd.read_parquet(full_element_path)
                data_frames.append(cv)
                print(f"{element} has been read correctly")
            except Exception as e:
                print(f"Error in {element}: {e}")
    if data_frames:
        cv_info_df = pd.concat(data_frames, ignore_index=True)
    else:
        cv_info_df = pd.DataFrame(columns=['id', 'name', 'email', 'linkedin'])
    cv_info_df.to_parquet("cv_datasets/cv_info")
    print("The full datasets can be found in the folder cv_datasets")

    job_schema_elements = os.listdir(job_schema_path)
    job_schema = []
    for element in job_schema_elements:
        full_element_path = os.path.join(schema_path, element)
        if os.path.isdir(full_element_path) and element.startswith("id=B"):
            for file_name in os.listdir(full_element_path):
                if file_name.endswith(".json"):
                    file_path = os.path.join(full_element_path, file_name)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            data = json.loads(line)
                            job_schema.append(data)
    with open("job_datasets/job_schema", "w") as f:
        json.dump(job_schema,f)
    
    job_text_elements = os.listdir(job_text_path)
    data_frames = []

    # collects all the processed texts and saves the text database
    for element in job_text_elements:
        full_element_path = os.path.join(text_path, element)
        if os.path.isdir(full_element_path) and element.startswith("id=A"):
            try:
                cv = pd.read_parquet(full_element_path)
                data_frames.append(cv)
                print(f"{element} has been read correctly")
            except Exception as e:
                print(f"Error in {element}: {e}")
    if data_frames:
        job_text_df = pd.concat(data_frames, ignore_index=True)
    else:
        job_text_df = pd.DataFrame(columns=['id', 'text'])
    job_text_df.to_parquet("job_datasets/job_text")
    encoder(job_text_df, "job", is_query= False)
    encoder(job_text_df, "job", is_query= True)

    print("The full datasets can be found in the folder job_datasets")

def main():
    spark_etl_script = "ingest_cv/cv_spark_pipeline/cv_spark_ingestion.py"
    consumer_script = "ingest_cv/cv_spark_pipeline/cv_spark_consumer.py"

    # ETL pipeline for the newly added file
    inputs, query_with_cv = give_inputs()
    print("Pipeline starting...")
    ingest_data(inputs)
    print("Please, open your Spark Streaming terminals. Are they still open?")
    while(True):
        terminal = input("Yes [Y] or no [N]?")
        terminal = terminal.lower()
        if terminal == "n":
            print("If one is still open, please close it. They will both be restarted")
            run_in_terminal(spark_etl_script)
            run_in_terminal(consumer_script)
        if terminal != 'n' and terminal != 'y':
            print("Invalid input")
        else:
            break
    # finds the most recent resume and adds it to the databases
    # loads the existing databases
    database_text = pd.read_parquet("cv_datasets/cv_text")
    database_info = pd.read_parquet("cv_datasets/cv_info")
    job_database_text = pd.read_parquet("job_datasets/job_text")
    with open("cv_datasets/cv_schema", "r") as f:
        database_schema = json.load(f)
    with open("job_datasets/job_schema", "r") as f:
        job_database_schema = json.load(f)
    
    # if querying with a cv, it finds the most recent (= last added) cv
    if query_with_cv == True:
        path = ".../cv-job-matcher-project/ingest_cv/cv_spark_pipeline/output_cv_processing/text_cv/"
        folder_id = [
            d for d in os.listdir(path) 
            if os.path.isdir(os.path.join(path, d)) and d.startswith("id=A")
        ]
        if folder_id:
            # id of the most recent added file
            most_recent = max(folder_id, key=lambda x: int(x.split('=A')[1]))

            # adds new record to the text database
            df_most_recent_text = pd.read_parquet(os.path.join(path, most_recent))
            database_text = pd.concat([database_text, df_most_recent_text], ignore_index=True)
            database_text.to_parquet("cv_datasets/cv_text")

            # adds new record to the personal info database
            info_add_path = f".../cv-job-matcher-project/ingest_cv/cv_spark_pipeline/output_cv_processing/info_cv/{most_recent}"
            to_add_info = pd.read_parquet(info_add_path)
            database_info = pd.concat([database_info, to_add_info], ignore_index=True)
            database_info.to_parquet("cv_datasets/cv_info")

            # adds new record to the schema database
            schema_folder = f".../cv-job-matcher-project/ingest_cv/cv_spark_pipeline/output_cv_processing/schema_cv/{most_recent}"
            for f_name in os.listdir(schema_folder):
                if f_name.endswith(".json"):
                    with open(os.path.join(schema_folder, f_name), "r") as f:
                        for line in f:
                            database_schema.append(json.loads(line))
            
            with open("cv_datasets/cv_schema", "w") as f:
                json.dump(database_schema, f)
        else:
            return
        
    else:
        path = ".../cv-job-matcher-project/ingest_cv/cv_spark_pipeline/output_cv_processing/text_job/"
        folder_id = [
            d for d in os.listdir(path) 
            if os.path.isdir(os.path.join(path, d)) and d.startswith("id=B")
        ]
        if folder_id:
            # id of the most recent added file
            most_recent = max(folder_id, key=lambda x: int(x.split('=B')[1]))

            # adds new record to the text database
            df_most_recent_text_job = pd.read_parquet(os.path.join(path, most_recent))
            database_text = pd.concat([database_text, df_most_recent_text], ignore_index=True)
            database_text.to_parquet("job_datasets/job_text")

            # adds new record to the schema database
            schema_folder = f".../cv-job-matcher-project/ingest_cv/cv_spark_pipeline/output_cv_processing/schema_job/{most_recent}"
            for f_name in os.listdir(schema_folder):
                if f_name.endswith(".json"):
                    with open(os.path.join(schema_folder, f_name), "r") as f:
                        for line in f:
                            database_schema.append(json.loads(line))
            
            with open("job_datasets/job_schema", "w") as f:
                json.dump(database_schema, f)
        else:
            return

    # start query and matching process with fass
    k = select_integer()
    if query_with_cv:
        query = encoder(df_most_recent_text, "cv")

        matches = matching(query, k)
    else:
        query = encoder(df_most_recent_text_job, "job")
        matches = matching(query, k, False)
    match_texts = [diz["match_text"] for diz in matches["matches"]]
    match_scores = [diz["match_distance"] for diz in matches["matches"]]
    match_scores = np.array(match_scores)
    match_id = [diz["match_id"] for diz in matches["matches"]]
    
    indices = np.argsort(match_scores)[::-1]
    match_df = pd.DataFrame({
    "id": match_id, 
    "match_text": match_texts,
    "score": match_scores
    })
    match_df = match_df.iloc[indices].reset_index(drop=True)
    
    # if querying with a job posting, returns also the personal information of the candidate
    if not query_with_cv:
        match_df =pd.merge(match_df, database_info, on='id', how='left')
        match_df.to_parquet(f"matches/query_with_{most_recent}")
    # prints top 5 matches
    print("TOP MATCHES FOUND:")
    print(match_df.head())
    
if __name__ == "__main__":
    order()
    main()






    
    
    




    


    





    
        
    
    




    


