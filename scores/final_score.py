# FUNNEL STRATEGY

# use FAISS to obtain the best k potential matches
# use the final score to rank them and find the best cv

from scores.cosine_distance import recommend_cvs, recommend_jobs, df_job_query # <--- !!! IMPORT DATAFRAME TO GET VALID IDS
from scores.jaccard import jaccard_similarity
from scores.title_analyser import title_category, title_similarity
from scores.years_experience import experience_computer

alpha = 1.0
beta = 0.5
gamma1 = 0.1
gamma2 = 0.1

def dynamic_penalty(gap, from_job_to_cv):
    if from_job_to_cv == False:
        return 0    
    if gap < 0:
        return -0.05
    else:
        return 0.15


def compute_final_score(cosine_sim, job_id, cv_id, from_job_to_cv = True):
    jaccard_sim = jaccard_similarity(job_id, cv_id)
    title_cat = title_category(job_id, cv_id)
    title_dist = title_similarity(job_id, cv_id)
    years_gap = experience_computer(job_id, cv_id)
    final_score = (alpha   * cosine_sim
                 + beta    * jaccard_sim
                 + gamma1  * title_cat
                 + gamma2  * title_dist
                 - dynamic_penalty(years_gap, from_job_to_cv) * years_gap
    )
    return final_score

# CV EVALUATION

def funneling_cvs(job_id):
    ret = recommend_cvs(job_id, k = 50)
    
    #handle case where no candidates are found
    if not ret:
        return None, 0.0

    max_pt = -100 #start lower than 0 in case of penalties
    winner = None # initialize variable safely
    
    for element in ret:
        cos_sim = element[1]
        cv_id = element[0]
        final_score = compute_final_score(cos_sim, job_id, cv_id)
        
        if final_score > max_pt:
            max_pt = final_score
            winner = cv_id
            
    return winner, max_pt

# JOB POSTING EVALUATION

def funneling_postings(cv_id):
    ret = recommend_jobs(cv_id, k = 50)

    #handle case where no jobs are found
    if not ret:
        return None, 0.0
        
    max_pt = -100
    winner = None
    
    for element in ret:
        cos_sim = element[1]
        job_id = element[0]
        final_score = compute_final_score(cos_sim, job_id, cv_id, False)
        
        if final_score > max_pt:
            max_pt = final_score
            winner = job_id
            
    return winner, max_pt

# return all candidates ranked, not just the winner

def funneling_cvs_ranked(job_id, k=50):
    #returns all candidates with scores, sorted by score descending
    ret = recommend_cvs(job_id, k=k)

    if not ret:
        return []

    scored = []
    for cv_id, cos_sim in ret:
        final_score = compute_final_score(cos_sim, job_id, cv_id)
        scored.append((cv_id, final_score))

    #sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def funneling_postings_ranked(cv_id, k=50):
    #returns all jobs with scores, sorted by score descending
    ret = recommend_jobs(cv_id, k=k)

    if not ret:
        return []

    scored = []
    for job_id, cos_sim in ret:
        final_score = compute_final_score(cos_sim, job_id, cv_id, from_job_to_cv=False)
        scored.append((job_id, final_score))

    #sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


if __name__ == "__main__":
    #dynamically pick a Job ID that exists in sample
    if not df_job_query.empty:
        valid_job_id = df_job_query["job_id"].iloc[0]
        print(f"Testing Funnel with Job: {valid_job_id}")

        result = funneling_cvs(valid_job_id)
        print(f"Winner: {result[0]}, Score: {result[1]}")

        #also test ranked version
        ranked = funneling_cvs_ranked(valid_job_id, k=5)
        print(f"Top 5 candidates: {ranked}")
    else:
        print("Error: No job embeddings found to test with.")