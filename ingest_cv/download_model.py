from huggingface_hub import snapshot_download
import os
#nhanv/cv_parser  string
#facebook/bart-large-mnli  cvs1
#urchade/gliner_base     cvs2
# Definiamo dove salvare il modelloc
def model_downloader(model_name, save_path_name):
    local_model_path = os.path.join(os.getcwd(), f"models/{save_path_name}")

    print(f"Inizio download del modello in: {local_model_path}")

    snapshot_download(
        repo_id=model_name, 
        local_dir=local_model_path,
        local_dir_use_symlinks=False, # Importante per evitare problemi di puntatori su Windows/Linux
        revision="main"
    )

    print("Download completato! Ora puoi disconnetterti da internet.")
    
def model_validator():
    model_list = [
        {"model_name" : "nhanv/cv_parser", "save_path_name" : "cv_parser_string_dataset_model"},
        {"model_name" : "facebook/bart-large-mnli", "save_path_name" : "cv_parser_cvs_model_1"},
        {"model_name" : "urchade/gliner_base", "save_path_name" : "cv_parser_cvs_model_2"}
    ]
    base_folder="models"
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    for model in model_list:
        full_path = os.path.join(current_script_dir, base_folder, model["save_path_name"])
        exists = os.path.exists(full_path)
        if not exists:
            model_downloader(**model)

if __name__ == "__main__":
    model_validator()