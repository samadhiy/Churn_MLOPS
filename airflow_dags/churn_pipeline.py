from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
}

with DAG(
    dag_id="churn_ml_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False
) as dag:

    ingestion = BashOperator(
        task_id="data_ingestion",
        bash_command="python src/data_ingestion.py"
    )

    preprocessing = BashOperator(
        task_id="preprocessing",
        bash_command="python src/preprocessing.py"
    )

    training = BashOperator(
        task_id="training",
        bash_command="python src/train.py"
    )

    evaluation = BashOperator(
        task_id="evaluation",
        bash_command="python src/evaluate.py"
    )

    ingestion >> preprocessing >> training >> evaluation