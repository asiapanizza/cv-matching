from scores.cosine_distance import recommend_cvs

# FOR CV RETRIEVAL

import pandas as pd

from data_loader import load_job_metadata_pandas

cv_df = pd.read_json("extraction/full_resume_dataset.json")
cv_df = cv_df[["resume_id", "skill_ids"]]

job_df = load_job_metadata_pandas()

#we must ensure skill_ids is a list (parquet might load it as array)!!!!!

def jaccard_similarity(job_id, cv_id):
    #retrieve skills for specific IDs
    job_skills = job_df[job_df["job_id"] == job_id]["skill_ids"].values[0]
    cv_skills = cv_df[cv_df["resume_id"] == cv_id]["skill_ids"].values[0]
    
    #convert to sets and handle cases where skills might be none/empty
    if job_skills is None: job_skills = []
    if cv_skills is None: cv_skills = []
    
    job_skills = set(list(job_skills))
    cv_skills = set(list(cv_skills))
    
    return compute_jaccard(job_skills, cv_skills)

def compute_jaccard(setA, setB):
    if len(setA.union(setB)) == 0: return 0.0
    int_c = len(setA.intersection(setB))
    un_c = len(setA.union(setB))
    return round(int_c/un_c, 4)

if __name__ == "__main__":
    
    #dynamically pick a valid ID
    valid_job_id = job_df["job_id"].iloc[0]
    
    print(f"Testing with Job ID: {valid_job_id}")
    print(jaccard_similarity(valid_job_id, "A1"))
    




