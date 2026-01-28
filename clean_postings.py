import pandas as pd
import json
postings = pd.read_csv("data_jobs.csv")

def schematize_posting(postings):
    diz = dict()
    diz["title"] = postings["job_title_short"]
    diz["category"] = "job"
    diz["location"] = str(postings["job_location"]) +", " + str(postings["job_country"])
    diz["skills"] = set()
    if isinstance(postings["job_skills"], list) or isinstance(postings["job_skills"], set):
        diz["skills"] = set(postings["job_skills"].values())
    diz["company"] = postings["company_name"]
    if isinstance(postings["job_type_skills"], dict):
        for el in list(postings["job_type_skills"].values()):
            if el not in diz["skills"]:
                diz["skills"]= diz["skills"].update(el)
    elif isinstance(postings["job_type_skills"], list):
        for el in postings["job_type_skills"]:
            if el not in diz["skills"]:
                diz["skills"]= diz["skills"].update(el)
    diz["skills"] = list(diz["skills"])
    return diz

def job_formatting(diz):
    text = diz["company"]
    if diz["location"]:
        text += " in " + diz["location"]
    text += " is looking for a "
    text += diz["title"]
    if len(diz["skills"])>0:
        text += "whose experienced in"
        for skill in diz["skills"]:
            text += " " + skill
    return text



    
if __name__ == "__main__":
    jobs = []
    for i in range(len(postings)):
        diz = schematize_posting(postings.iloc[i])
        jobs.append(diz)
    with open("cleaned_jobs.json", "w") as f:
        json.dump(jobs,f, indent = 4)
    print(jobs[0])

