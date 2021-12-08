
from airflow.decorators import dag
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from os.path import dirname, join
from datetime import datetime as dt

default_args = {
    'owner': 'spike',
    #'retries': 1,
    'venv_folder': join(dirname(__file__), "..", "venv" ),
    'csv_folder':join(dirname(__file__), "..", "data" ),
    'pickles_folder': join(dirname(__file__), "..", "pickles"),
    'start_date': dt(2021,1,1)
}

# define the DAG
dag = DAG(
    'SpikeModelGenerator',
    default_args=default_args,
    description='Training Models 1/2 for ML Spike challenge',
    # schedule_interval=timedelta(days=1),
)


task_1 = BashOperator(
    task_id='pre-process',
    params = {'venv_folder' : default_args["venv_folder"] },
    bash_command= "{{params.venv_folder}}/bin/python sope.py",
    dag =dag
)

task_1
