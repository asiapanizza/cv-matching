# this script converts the dataset "master_resumes.jsonl" into our chosen schema
# it then saves the json file


import json
from datetime import datetime, timedelta
import re
import pandas as pd
import numpy as np

def reprocess_json(resume):
    diz = dict()
    diz["personal information"] = {}
    
    # --- Personal Info Guards ---
    if "personal_info" in resume:
        # Name
        val_name = resume['personal_info'].get("name", "")
        if val_name and val_name.lower() != "unknown" and val_name.lower() != "not provided":
            diz["personal information"]["name"] = val_name
        else:
            diz["personal information"]["name"] = ""
            
        # LinkedIn
        val_link = resume['personal_info'].get("linkedin", "")
        if val_link and val_link.lower() != "unknown" and val_link.lower() != "not provided":
            link = val_link
        else:
            link = ""
        diz["personal information"]["linkedin"] = link
        
        # Email
        val_email = resume['personal_info'].get("email", "")
        if val_email and val_email.lower() != "unknown" and val_email.lower() != "not provided":
            mail = val_email
        else:
            mail = ""
        diz["personal information"]["email"] = mail 
        
        # Location (Fixed .lower() bug)
        diz["schema"] = {}
        city = resume['personal_info'].get("location", {}).get("city", "")
        if city and city.lower() != "unknown" and city.lower() != "not provided":
            loc = [city, resume['personal_info']["location"].get("country", "")]
            loc = ", ".join([l for l in loc if l])
        else:
            loc = ""
        diz["schema"]["location"] = loc
        
        # Summary/Title
    else:
        diz["schema"] = {"title": "", "location": ""}
        diz["personal information"] = {"email": "", "name": "", "linkedin": ""}

    # --- Experience Guards ---
    job_list = []
    total_years = []
    skills_to_add = []
    diz["schema"]["title"] = resume["experience"][0]["title"]
    for el in resume.get("experience", []):
        period = []
        job = dict()
        
        # Title
        val_t = el.get("title", "")
        if val_t and val_t.lower() != "unknown" and val_t.lower() != "not provided":
            job["title"] = val_t
        else:
            job["title"] = ""
            
        # Company
        val_c = el.get("company", "")
        if val_c and val_c.lower() != "unknown" and val_c.lower() != "not provided":
            job["company"] = val_c
        else:
            job["company"] = ""
            
        # Dates
        if "dates" in el:
            start_date = str(el["dates"].get("start", ""))
            if start_date and start_date.lower() != "unknown" and start_date.lower() != "not provided":
                period.append(start_date[:4])
                if start_date[:4].isdigit():
                    total_years.append(int(start_date[:4]))
            
            end = str(el["dates"].get("end", ""))
            if end.lower() == "present":
                end = "2026"
            
            if end and end.lower() != "unknown" and end.lower() != "not provided":
                period.append(end[:4])
                if end[:4].isdigit():
                    total_years.append(int(end[:4]))
            
            dur = el["dates"].get("duration", "")
            if dur and dur.lower() != "unknown" and dur.lower() != "not provided":
                period.append(f"({dur})")
                
            job["period"] = " ".join(period)
        else: 
            job["period"] = ""

        # Responsibilities
        if "responsibilities" in el:
            job["description"] = ", ".join([r for r in el["responsibilities"] if r.lower() not in ["unknown", "not provided"]])
        else:
            job["description"] = ""
            
        # Technical Environment
        if "technical_environment" in el:
            for skill_type in el["technical_environment"]:
                for skill in el["technical_environment"][skill_type]:
                    if skill and skill.lower() != "unknown" and skill.lower() != "not provided":
                        skills_to_add.append(skill)
        job_list.append(job)

    # Education 
    edu_list = []
    for el in resume.get("education", []):
        edu = dict()
        inst = el.get("institution", {}).get("name", "")
        if inst and inst.lower() != "unknown" and inst.lower() != "not provided":
            edu["institution"] = inst
        else:
            edu["institution"] = ""
            
        edu["degree"] = []
        deg = el.get("degree", {})
        for field in ["level", "field"]:
            val_f = deg.get(field, "")
            if val_f and val_f.lower() != "unknown" and val_f.lower() != "not provided":
                edu["degree"].append(val_f)
        edu["degree"] = " ".join(edu["degree"])

        period = []
        dates = el.get("dates", {})
        for d_key in ["start", "expected_graduation", "end"]:
            d_val = str(dates.get(d_key, ""))
            if d_val and d_val.lower() != "unknown" and d_val.lower() != "not provided":
                period.append(d_val[:4])
        edu["period"] = " - ".join(list(dict.fromkeys(period))) # distinct years

        details = []
        achievements = el.get("achievements", {})
        for voice in achievements:
            val = achievements[voice]
            if isinstance(val, str):
                if val.lower() != "unknown" and val.lower() != "not provided":
                    details.append(val)
            elif isinstance(val, list):
                for item in val:
                    if str(item).lower() != "unknown" and str(item).lower() != "not provided":
                        details.append(str(item))
        edu["details"] = ", ".join(details)
        edu_list.append(edu)

    # Skills Extraction 
    if "skills" in resume:
        for group in resume["skills"].values():
            if isinstance(group, dict):
                for sublist in group.values():
                    for s in sublist:
                        name = s.get("name", "") if isinstance(s, dict) else s
                        if name and str(name).lower() != "unknown" and str(name).lower() != "not provided":
                            skills_to_add.append(str(name))
            elif isinstance(group, list):
                for item in group:
                    if isinstance(item, dict):
                        for v in item.values():
                            if v and str(v).lower() != "unknown" and str(v).lower() != "not provided":
                                skills_to_add.append(str(v))
                    elif item and str(item).lower() != "unknown" and str(item).lower() != "not provided":
                        skills_to_add.append(str(item))

    diz["schema"]["experience"] = job_list
    diz["schema"]["education"] = edu_list
    diz["schema"]["skills"] = list(set(skills_to_add))
    
    # Experience calculation
    if total_years:
        diz["schema"]["total_experience"] = max(total_years) - min(total_years)
    else:
        diz["schema"]["total_experience"] = 0 if not job_list else None
        
    return diz



if __name__ == "__main__":
    resumes = []
    with open('ingest_cv/master_resumes.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    resume = json.loads(line)
                    resumes.append(resume)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line.strip()}. Error: {e}")
    
    i = 1
    to_save = []
    for el in resumes:
        schema = reprocess_json(el)["schema"]
        schema["id"] = "A" + str(i)
        to_save.append(schema)
        i +=1
    with open("new_training/datasets.json", "w") as f:
        json.dump(to_save, f, indent=4)
    print(to_save[0])

        

