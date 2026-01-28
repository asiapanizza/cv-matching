#!/bin/bash

# Configuration
KAFKA_DIR="${1:-/usr/local/kafka}"
PORT=9092
# Ci serve ancora il binario locale per eseguire il comando kafka-topics
# se non vuoi entrare ogni volta nel container.

echo "Connecting to Kafka Docker instance on port: $PORT"

# 1. Avvio del container Docker (solo se non è già attivo)
if [ ! "$(docker ps -q -f name=kafka-broker)" ]; then
    echo "Starting Kafka via Docker Compose..."
    docker compose up kafka -d
else
    echo "Kafka container is already running."
fi

# 2. Verifica che Kafka sia pronto sulla porta 10000
echo "Waiting for Kafka to be ready on port $PORT..."
while ! (timeout 1 bash -c "</dev/tcp/localhost/$PORT" 2>/dev/null); do
    sleep 1
done

echo "Kafka is READY! Creating Topics..."

# 3. Lista dei Topic
TOPICS=(
    "raw_resumes"
    "json_dataset"
    "linkedin_pdf"
    "string_dataset"
    "new_texts"
    "processed_schema_cv"
    "processed_text_cv"
    "processed_personal_info_cv"
)

# 4. Creazione Topic (eseguita tramite i binari locali puntando al Docker)
for TOPIC in "${TOPICS[@]}"; do
    # Verifichiamo se il topic esiste già per evitare errori nell'output
    EXISTING=$($KAFKA_DIR/bin/kafka-topics.sh --list --bootstrap-server localhost:$PORT | grep "^$TOPIC$")
    
    if [ -z "$EXISTING" ]; then
        echo "Creating topic: $TOPIC"
        $KAFKA_DIR/bin/kafka-topics.sh --create \
            --topic "$TOPIC" \
            --bootstrap-server localhost:$PORT \
            --partitions 1 \
            --replication-factor 1
    else
        echo "Topic $TOPIC already exists, skipping."
    fi
done

echo "--- All topics verified ---"

# 5. Mostra i log del container invece di un file locale
echo "Streaming logs from docker container 'kafka-broker'..."
docker logs -f kafka-broker