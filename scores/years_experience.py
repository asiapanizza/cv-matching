import pandas as pd
import math
from data_loader import load_job_metadata_pandas

QUALIF_PENALTY = - 0.15
QUALIF_PRIZE = 0.05

def experience_computer(one_job, one_cv):
    # it's slightly better if skills exceed requirements min(3)
    # it's much worse if skills exceed requirements (max (3))
    diz = {1: "internship level", 2: "junior level", 3: "mid-level", 4:"senior level"}
    maximum = (max(diz.keys()) - min(diz.keys()))
    for key in diz.keys():
        if diz[key] in one_job.lower():
            for other_key in diz.keys():
                if other_key in one_cv.lower():
                    diff = other_key - key
                    return max(0,diff)*QUALIF_PRIZE/maximum - min(0,diff)*QUALIF_PENALTY/maximum
                

    

    
    