
import pandas as pd
import os

def print_parquet_table(file_path):
    # Verifica se il file esiste
    if not os.path.exists(file_path):
        print(f"Errore: Il file {file_path} non esiste.")
        return

    try:
        # Legge il file Parquet
        # Nota: richiede 'pip install pyarrow' o 'pip install fastparquet'
        df = pd.read_parquet(file_path)

        # Configurazione Pandas per visualizzare meglio la tabella nel terminale
        pd.set_option('display.max_columns', None)  # Mostra tutte le colonne
        pd.set_option('display.width', 1000)        # Evita che la tabella vada a capo
        pd.set_option('display.colheader_justify', 'center') # Centra le intestazioni

        print(f"\n--- Contenuto del file: {os.path.basename(file_path)} ---")
        
        # Se il file è molto grande, stampiamo solo le prime 20 righe
        if len(df) > 20:
            print(df.head(20))
            print(f"\n... mostrate 20 righe su un totale di {len(df)} ...")
        else:
            print(df)
            
        print("-" * (len(os.path.basename(file_path)) + 24))

    except Exception as e:
        print(f"Si è verificato un errore durante la lettura: {e}")

# --- CONFIGURAZIONE ---
# Inserisci qui il percorso del tuo file parquet
percorso_file = "/home/davide/Desktop/cv-job-matcher-project/ingest_cv/cv_spark_pipeline/output_cv_processing/text_cv/id=A1/part-00007-e5adcca7-1b91-4315-9ff3-3edddab6d6eb.c000.snappy.parquet"

if __name__ == "__main__":
    print_parquet_table(percorso_file)