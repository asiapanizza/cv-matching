import json
import os
import base64
import logging
from confluent_kafka import Producer
import pandas as pd

# --- CONFIGURATION ---
RESUME_STATE_FILE = 'id_counter_resumes.txt'
JOB_STATE_FILE = 'id_counter_jobs.txt'

TOPIC_RESUMES = 'raw_resumes'
TOPIC_JOBS = 'raw_jobs'

KAFKA_CONF = {
    'bootstrap.servers': 'localhost:9092', 
    'linger.ms': 10,
    'message.max.bytes': 10485760 
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def delivery_report(err, msg):
    if err is not None:
        print(f'Invio fallito: {err}')
    else:
        print(f'Messaggio inviato a {msg.topic()} [Partition: {msg.partition()}]')

# --- STATE MANAGEMENT ---
def get_next_id(state_file):
    """Reads the counter for a specific state file."""
    if not os.path.exists(state_file):
        return 1
    with open(state_file, 'r') as f:
        try:
            return int(f.read().strip()) + 1
        except ValueError:
            return 1

def update_state_file(state_file, last_id):
    """Saves the last used ID to the specific state file."""
    with open(state_file, 'w') as f:
        f.write(str(last_id))

# --- FILE HANDLERS ---
def yield_jsonl_records(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def yield_json_records(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            for item in data: yield item
        else: yield data

def yield_pdf_record(path):
    with open(path, "rb") as f:
        pdf_bytes = f.read()
        base64_string = base64.b64encode(pdf_bytes).decode('utf-8')
    yield base64_string

def yield_text_record(path):
    with open(path, 'r', encoding='utf-8') as f:
        yield f.read()

def yield_csv_records(path, column_name='Resume_str'):
    try:
        df = pd.read_csv(path)
        for text in df[column_name]:
            yield str(text) if pd.notna(text) else ""
    except Exception as e:
        print(f"Errore lettura CSV: {e}")

# --- MAIN LOGIC ---
def ingest_data(files_to_process):
    p = Producer(KAFKA_CONF)
    
    # Initialize both counters
    resume_id = get_next_id(RESUME_STATE_FILE)
    job_id = get_next_id(JOB_STATE_FILE)
    
    total_sent = 0
    print(f"--- Starting Ingestion | Resumes: A{resume_id} | Jobs: B{job_id} ---")

    for file_info in files_to_process:
        f_path = file_info['path']
        f_source = file_info['source']
        f_type = file_info['type']
        f_category = file_info.get('category', 'cv') 

        if not os.path.exists(f_path):
            print(f"Skipping missing: {f_path}")
            continue
        
        # Select iterator
        if f_type == 'jsonl': iterator = yield_jsonl_records(f_path)
        elif f_type == 'json': iterator = yield_json_records(f_path)
        elif f_type == 'pdf': iterator = yield_pdf_record(f_path)
        elif f_type == 'txt': iterator = yield_text_record(f_path)
        elif f_type == "csv":
            col = file_info.get('col', 'Resume_str') 
            iterator = yield_csv_records(f_path, column_name=col)
        else: continue

        for raw_content in iterator:
            # Logic for separate ID prefix and counter
            if f_category == 'job':
                unique_id = f"B{job_id}"
                target_topic = TOPIC_JOBS
                job_id += 1
            else:
                unique_id = f"A{resume_id}"
                target_topic = TOPIC_RESUMES
                resume_id += 1
            
            payload = {
                "id": unique_id,
                "raw_data": raw_content, 
                "source": f_source,
                "type": f_type,
                "category": f_category 
            }

            try:
                p.produce(
                    target_topic,
                    key=unique_id,
                    value=json.dumps(payload).encode('utf-8'),
                    callback=delivery_report
                )
                p.poll(0)
                total_sent += 1
            except Exception as e:
                print(f"Failed to produce {unique_id}: {e}")

    p.flush()
    
    # Save both states
    update_state_file(RESUME_STATE_FILE, resume_id - 1)
    update_state_file(JOB_STATE_FILE, job_id - 1)
    
    print(f"--- Finished. Sent {total_sent} records. ---")

if __name__ == "__main__":
    MIXED_BATCH = [
        {"path": "ingest_cv/master_resumes.jsonl", "source": "json_dataset", "type": "jsonl", "category": "cv"},
        {"path": "ingest_job/job_postings.json", "source": "linkedin_jobs", "type": "json", "category": "job"}
    ]
    ingest_data(MIXED_BATCH)