import faiss
import numpy as np
import pandas as pd

# HERE: LOOKING TO MATCH CV WITH JOBS
CV_QUERY_PATH = "embeddings/cv_query_embedding.parquet"
JOB_PASSAGE_PATH = "embeddings/job_embeddings_passage.parquet"

 
df_job_passage = pd.read_parquet(JOB_PASSAGE_PATH)
df_cv_query = pd.read_parquet(CV_QUERY_PATH)
cv_query_lookup = df_cv_query.set_index('embedding_text')['embedding'].to_dict()
job_passage_lookup = df_job_passage.set_index('embedding_text')['embedding'].to_dict()
cv_id_to_text_query = df_cv_query.set_index('cv_id')['embedding_text'].to_dict()
job_id_to_text_passage = df_job_passage.set_index('job_id')['embedding_text'].to_dict()

job_dim = len(df_job_passage["embedding"][0])

# HERE: LOOKING TO MATCH JOB WITH CVS

# CV_PASSAGE_PATH = "training_embeddings/cv_embeddings_passage.parquet"
# JOB_QUERY_PATH = "training_embeddings/job_embeddings_query.parquet"

CV_PASSAGE_PATH = CV_QUERY_PATH
JOB_QUERY_PATH = JOB_PASSAGE_PATH
df_job_query = pd.read_parquet(JOB_QUERY_PATH)
df_cv_passage= pd.read_parquet(CV_PASSAGE_PATH)
cv_passage_lookup = df_cv_passage.set_index('embedding_text')['embedding'].to_dict()
job_query_lookup = df_job_query.set_index('embedding_text')['embedding'].to_dict()
cv_id_to_text_passage = df_cv_passage.set_index('cv_id')['embedding_text'].to_dict()
job_id_to_text_query = df_job_query.set_index('job_id')['embedding_text'].to_dict()

cv_dim = len(df_cv_passage["embedding"][0]) 



def matching(new_query, k = 10, search_jobs_for_cv = True):
    if search_jobs_for_cv == True:
        print("Looking to match your cvs with our dataset of jobs...")
        # we query with cv to find jobs
        job_matrix = np.stack(df_job_passage["embedding"].tolist()).astype('float32')
        job_ids = df_job_passage["job_id"].str[1:].astype("int64")
        index_job = faiss.IndexFlatIP(job_dim)
        index_job = faiss.IndexIDMap(index_job)
        index_job.add_with_ids(job_matrix, job_ids)
        if isinstance(new_query, str):
            new_query = [new_query]
        embeddings = [cv_query_lookup[text] for text in new_query]
        query_vec = np.stack(embeddings).astype('float32')
        D, I = index_job.search(query_vec, k)
        match_list = []
        for i in range(len(new_query)):
            job_texts = []
            for j in range(len(I[i])):
                if I[i][j] != -1:
                    job_id = "B" + str(I[i][j])
                    dist = D[i][j]
                    job_text = job_id_to_text_passage.get(job_id, None)
                    matched = dict()
                    matched["match_text"] = job_text
                    matched["match_id"] = job_id
                    matched["match_distance"] = dist
            
            matches = {
                "anchor": new_query[i],  
                "matches": job_texts
            }
            match_list.append(matches)
        
        return match_list
    else:
        print("Looking to match your jobs with our dataset of cvs...")
        cv_matrix = np.stack(df_cv_passage["embedding"].tolist()).astype('float32')
        cv_ids = df_cv_passage["cv_id"].str[1:].astype("int64")
        index_cv = faiss.IndexFlatIP(cv_dim)
        index_cv = faiss.IndexIDMap(index_cv)
        index_cv.add_with_ids(cv_matrix, cv_ids)
        if isinstance(new_query, str):
            new_query = [new_query]
        embeddings = [job_query_lookup[text] for text in new_query]
        query_vec = np.stack(embeddings).astype('float32')
        D, I = index_cv.search(query_vec, k)
        match_list = []
        for i in range(len(new_query)):
            cv_texts = []
            for j in range(len(I[i])):
                if I[i][j] != -1:
                    cv_id = "A" + str(I[i][j])
                    dist = D[i][j]
                    cv_text = cv_id_to_text_passage.get(cv_id, None)
                    matched = dict()
                    matched["match_text"] = cv_text
                    matched["match_id"] = cv_id
                    matched["match_distance"] = dist
            matches = {
                "anchor": new_query[i],  
                "matches": matched
            }
            match_list.append(matches)
        
        return match_list

# TODO: write data into index



if __name__ == "__main__":
    # looking for jobs to match my cv
    print(matching(df_cv_query["embedding_text"].iloc[0], k = 1))

    # # looking for cvs to match my job
    print(matching(df_job_query["embedding_text"].iloc[0], k = 1, search_jobs_for_cv= False))
    print(df_job_query["embedding_text"].iloc[0])










