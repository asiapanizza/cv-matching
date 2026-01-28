import pandas as pd
from py_stringmatching import PartialRatio
from data_loader import load_job_metadata_pandas


occupations = pd.read_csv("extraction/occupations_en.csv")
occupations = occupations[["preferredLabel", "altLabels", "definition", "description"]]
occupation_diz = dict()


# occupation dictionary creation

for i in range(len(occupations)):
    key = occupations.loc[i]["preferredLabel"]
    values = occupations.loc[i]["altLabels"]
    if isinstance(values, str): #safety check
        values = values.split("\n")
    else:
        values = []
    values.append(key)
    values = list(set(values))
    occupation_diz[key] = values


cv_df = pd.read_json("extraction/full_resume_dataset.json")
cv_df = cv_df[["resume_id", "titles"]]


job_df = load_job_metadata_pandas()


occupation_groups = list(occupation_diz.values())

def title_category(job_id, cv_id):
    job_title = job_df[job_df["job_id"]== job_id]["title"].values[0]
    cv_titles = cv_df[cv_df["resume_id"] == cv_id]["titles"].values[0]
    pt=0
    for el in cv_titles:
        if el == job_title:
            return 1
        else:
            for i in range(len(occupation_groups)):
                if el in occupation_groups[i] and job_title in occupation_groups[i]:
                    pt +=1
    return round(pt,4)


def title_similarity(job_id, cv_id):
    s = PartialRatio()
    job_title = job_df[job_df["job_id"]== job_id]["title"].values[0]
    cv_titles = cv_df[cv_df["resume_id"] == cv_id]["titles"].values[0]
    pt = 0
    for title in cv_titles:
        score = s.get_raw_score(title, job_title)/100
        pt = max(pt, score)
    return pt
     


if __name__ == "__main__":

    #dynamically pick a valid ID
    valid_job_id = job_df["job_id"].iloc[0]
    
    print(f"Testing with Job ID: {valid_job_id}")
    print(title_category(valid_job_id, "A1"))
    print(title_similarity(valid_job_id, "A1"))


