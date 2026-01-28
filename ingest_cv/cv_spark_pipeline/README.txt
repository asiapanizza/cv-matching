################
#STARTING KAFKA#
################
It's possibile to start Kafka with KRaft (so without Zookeeper) on Linux in a single use mode.
The "start_single_use_kafka_session.sh" file has been created in order to
start the kafka server and create the topics.
By using it, the kafka files will be put into /tmp, so after each PC shutdown all the 
files from a previous run are trashed.

In order to use it, the user must open a console inside the folder and execute:
./start_single_use_kafka_session.sh

##############################
#AUTOMATIC PIPELINE EXECUTION#
##############################
The "run_pipeline.sh" file has been created in order to automatically execute the entire cv
pipeline. It works by creating the conda environmnet, starting the kafka server and the 
opening a terminal for each part of the pipeline (producer, consumer and ingestion).
In this way the user can monitor the entire pipeline processing on different terminals.
The pipeline outputs are inside the "output_cv_processing" folder

In order to use it, the user must open a console inside the folder and execute:
./run_pipeline.sh