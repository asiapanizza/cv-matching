import pandas as pd
import json

import pandas as pd
import os



import pandas as pd
import os

import pandas as pd
import os



def schematize_posting(postings):
    diz = dict()
    
    # Inseriamo il tuo ID personalizzato (B1, B2...)
    
    diz["title"] = postings["job_title_short"] if pd.notna(postings["job_title_short"]) else ""
    diz["category"] = "job"
    diz["company"] = postings["company_name"] if pd.notna(postings["company_name"]) else ""
    
    # Gestione Location: evita "nan, Italy"
    loc = str(postings["job_location"]) if pd.notna(postings["job_location"]) else ""
    country = str(postings["job_country"]) if pd.notna(postings["job_country"]) else ""
    
    if loc and country:
        diz["location"] = f"{loc}, {country}"
    else:
        diz["location"] = loc or country or ""

    # Gestione Skills
    skills_set = set()
    
    # 1. job_skills
    js = postings.get("job_skills")
    if pd.notna(js):
        if isinstance(js, (list, set)):
            skills_set.update(js)
        elif isinstance(js, dict):
            skills_set.update(js.values())
        elif isinstance(js, str):
            if js[0] == "[":
                skill = eval(js)
                skills_set.update(skill)
            elif js[0] == "{":
                skill = eval(js)
                skills_set.update(skill.values())

    # 2. job_type_skills
    jts = postings.get("job_type_skills")
    if pd.notna(jts):
        if isinstance(jts, dict):
            for el in jts.values():
                # .update() aggiunge elementi se el è una lista, .add() se è singolo
                if isinstance(el, list): skills_set.update(el)
                else: skills_set.add(el)
        elif isinstance(jts, list):
            skills_set.update(jts)

    # Convertiamo in lista e rimuoviamo eventuali NaN rimasti dentro le liste
    diz["skills"] = [str(s) for s in skills_set if pd.notna(s)]
    
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
    postings = pd.read_csv("data_jobs.csv")
    jobs = []
    for i in range(len(postings)):
        diz = schematize_posting(postings.iloc[i])
        jobs.append(diz)
    with open("cleaned_jobs.json", "w") as f:
        json.dump(jobs,f, indent = 4)
    print(jobs[0])

