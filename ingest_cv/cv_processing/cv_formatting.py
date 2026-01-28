import re
import json
import pandas as pd

def cv_formatter(resume):
    diz = dict()
    parts = ["I am a"] 
    content = resume.get("schema", {})      
    pers = resume.get("personal information", {})
    
    # 1. Title
    title = content.get("title", "")
    if title:
        parts.append(title)
    
    # 2. Experience Level
    exp = content.get("total_experience")
    if exp is not None:
        try:
            exp_val = int(exp)
            level = "principal level"
            if exp_val == 0: level = "internship level"
            elif exp_val < 3: level = "junior level"
            elif exp_val < 6: level = "mid-level"
            elif exp_val < 8: level = "senior level"
            parts.append(f"with {exp_val} years of experience ({level}).")
        except ValueError:
            pass

    # 3. Skills
    skills = content.get("skills", [])
    if skills:
        parts.append("My skills include: " + ", ".join(skills) + ".")

    # 4. Work Experience
    experience = content.get("experience", [])
    for job in experience:
        job_title = job.get('title', '')
        if not job_title: continue
        
        job_str = f"I worked as {job_title}"
        
        company = job.get("company")
        if company and company not in ["Company", ""]:
            job_str += f" at {company}"
            
        period = job.get("period")
        if period:
            job_str += f" from {period}"
            
        location = job.get("location")
        if location and location not in ["Unknown", "City , State", ""]:
            job_str += f" in {location}"
            
        desc = job.get("description")
        if desc:
            job_str += f", {desc}"
            
        parts.append(job_str + ".")

    # 5. Education
    education = content.get("education", [])
    for edu in education:
        degree = edu.get("degree", "")
        if degree and degree not in ["Degree Not Found", ""]:
            text = f"I completed my {degree}"
            
            inst = edu.get("institution", "")
            if inst and inst not in ["Institution Not Found", ""]:
                text += f" at {inst}"
            
            period = edu.get("period")
            if period:
                text += f" around {period}"
                
            details = edu.get("details")
            if details:
                text += f", {details}"
            
            parts.append(text + ".")
            
    # Join e pulizia regex
    full_text = " ".join(parts)   
    full_text = re.sub(r"\s+", " ", full_text).strip() 
    diz["text"] = full_text
    return diz, pers

if __name__ == "__main__":
    # Caricamento dati
    with open("new_training/datasets.json") as f:
        resumes = json.load(f)
    
    formatted_data = []
    for res in resumes:
        # Spacchettamento corretto della tupla
        diz_formatted, personal_info = cv_formatter(res)
        
        # Aggiunta prefisso Query
        diz_formatted["text"] = "Query: " + diz_formatted["text"]
        
        # Opzionale: puoi unire info personali se ti servono nel dataframe
        # diz_formatted.update(personal_info) 
        
        formatted_data.append(diz_formatted)
    
    # Creazione DataFrame e salvataggio
    df = pd.DataFrame(formatted_data)
    df.to_parquet("new_training/cv_query_text.parquet", engine="pyarrow", index=False)
    print(f"Successfully processed {len(df)} CV")