
## Cv-Job Matcher

Cv-Job Matcher is a Python program to match job postings and cvs.
Please note that the program is intended to be used on Linux, performance on Windows/Mac is not ensured.

## Packages and software required

Kindly ensure you have installed Java 17.

## Usage

# 1. Initiate Kafka with KRaft

It's possibile to start Kafka with KRaft (so without Zookeeper) on Linux in a single use mode.
The "start_single_use_kafka_session.sh" file has been created in order to
start the kafka server and create the topics.
By using it, the kafka files will be put into /tmp, so after each PC shutdown all the 
files from a previous run are trashed.

In order to use it, the user must open a console inside the folder and execute:
./start_single_use_kafka_session.sh


# 2. Activate the pipeline automatically

The "run_pipeline.sh" file has been created in order to automatically execute the entire cv
pipeline. It works by creating the conda environmnet, starting the kafka server and the 
opening a terminal for each part of the pipeline (producer, consumer and ingestion).
In this way the user can monitor the entire pipeline processing on different terminals.
The pipeline outputs are inside the "output_cv_processing" folder

In order to use it, the user must open a console inside the folder and execute:
./run_pipeline.sh


# 3. Prepare the dataset

On first usage, please run "prepare_cv_pipeline.py", which will perform ETL of the main datasets, both for resumes and job postings. Wait for the process to be over, then open "run.py" and execute the function "order()".

At the end of the ingestion process, as visible in the terminal, it will be possible to have the program ingest new documents, by running the function main() in run.py. 

# 4. Find a match for your file

Please, upload the files you want to ingest in the repository, preferably in the folder "new_entries".
Formats accepted for resumes are ".txt" and ".pdf", preferably a LinkedIn pdf in English.

Then, run the function "main()": answer the questions that appear in the terminal. This will start the new ingestion. 
Kindly note that the same file will be processed only once.
The program will match your query with the chosen number of resumes or job postings.

## Contributions

Cv_job_matcher is a project by Asia Panizza.
