#!/bin/bash

# Configuration
KAFKA_DIR="${1:-/usr/local/kafka}"
PORT=9092
YML_FILE="cv_processing_conda_environment.yml" 
ENV_NAME="cv_processing_conda_environment"
PROJECT_DIR="$(pwd)"
REAL_USER=${SUDO_USER:-$(whoami)}
REAL_HOME=$(getent passwd "$REAL_USER" | cut -d: -f6)

# Conda Path discovery
CONDA_BASE_PATH=""
for p in "$REAL_HOME/anaconda3" "$REAL_HOME/miniconda3" "/opt/anaconda3" "/opt/miniconda3"; do
    if [ -f "$p/etc/profile.d/conda.sh" ]; then
        CONDA_BASE_PATH="$p/etc/profile.d/conda.sh"
        break
    fi
done

# Pulizia file locali (non legati a Kafka)
rm -rf checkpoints_nuovo_test id_counter.txt
echo "Pipeline starter"

# --- GESTIONE CONDA ---
source "$CONDA_BASE_PATH"
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "Installing environment from $YML_FILE"
    conda env create -f "$YML_FILE"
else
    echo "Conda environment '$ENV_NAME' already present"
fi

# --- GESTIONE DOCKER KAFKA ---
echo "Resetting Kafka via Docker..."
# Spegne e rimuove volumi vecchi per pulire i dati (equivalente al tuo vecchio rm -rf $DATA_DIR)
docker compose down -v 2>/dev/null

echo "Starting Kafka container..."
docker compose up kafka -d

# --- WAIT DINAMICO (Invece di sleep 30) ---
echo "Waiting for Kafka to be ready on port $PORT..."
RETRIES=0
while ! (timeout 1 bash -c "</dev/tcp/localhost/$PORT" 2>/dev/null); do
    ((RETRIES++))
    if [ $RETRIES -gt 60 ]; then
        echo "Error: Kafka took too long to start."
        exit 1
    fi
    sleep 1
done
echo "Kafka is READY!"

# --- CREAZIONE TOPIC ---
echo "Creating topics on port $PORT..."
TOPICS=(
    "raw_resumes" "json_dataset" "linkedin_pdf" 
    "processed_schema_cv" "processed_text_cv" "processed_personal_info_cv"
)

for TOPIC in "${TOPICS[@]}"; do
    # Usiamo il binario locale puntando al broker Docker
    $KAFKA_DIR/bin/kafka-topics.sh --create --topic "$TOPIC" \
        --bootstrap-server localhost:$PORT \
        --if-not-exists # Evita errori se già presente
done
echo "Created topics."

# --- START PYTHON SCRIPTS ---
echo "Starting python scripts"
# Nota: assicurati che i tuoi file .py puntino a localhost:10000 nei loro bootstrap_servers
PYTHON_CMD="source $CONDA_BASE_PATH && conda activate $ENV_NAME && cd $PROJECT_DIR && python"

gnome-terminal --tab --title="INGESTER" -- bash -c "$PYTHON_CMD cv_spark_ingestion.py; exec bash"
sleep 1
gnome-terminal --tab --title="SPARK-CONSUMER" -- bash -c "$PYTHON_CMD cv_spark_consumer.py; exec bash"
sleep 5 # Ridotto lo sleep, ora che Kafka è sicuramente attivo
gnome-terminal --tab --title="PRODUCER" -- bash -c "$PYTHON_CMD cv_spark_producer.py; exec bash"

echo "Pipeline started successfully on port $PORT"