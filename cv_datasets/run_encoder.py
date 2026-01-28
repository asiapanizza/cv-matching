import json
import os

DIMENSION = 64
def encoder(input_df, input_type, model_path = f"trained_biencoders/trained_biencoder_2e-05", is_query = True):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.xpu.is_available():
        device = "xpu"
    from sentence_transformers import SentenceTransformer, util
    import torch
    import pandas as pd
    model_path = f"trained_biencoders/trained_biencoder_2e-05"
    model = SentenceTransformer(model_path, device = device, truncate_dim=dim)
    ids = input_df["id"].tolist()
    texts = input_df["text"].tolist()
    if is_query:
        texts= ["Query: " + text for text in texts]
        kind = "query"
    else:
        texts = ["Passage: " + text for text in texts]
        kind = "passage"
    # Generiamo gli embeddings troncati alla dimensione specifica
    embeddings = model.encode(texts, truncate_dim=DIMENSION, convert_to_tensor=True)
    diz =  { "id": ids, 
            "embedding":embeddings,
            "text": texts
        }


    folder_path = "embeddings"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = f"{folder_path}/{input_type}_{kind}_embedding.parquet"

    if os.path.exists(file_path):
        df_existing = pd.read_parquet(file_path)
        df_new = pd.DataFrame([diz])
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = pd.DataFrame([diz])

    df_final.to_parquet(file_path, engine='pyarrow', index=False)
    

