from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator

# import tasks
from m1_script import task1
from m2_script import task2
from postgres_script import task3
from dash_script import task4

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 1,
}

dag = DAG(
    'accidents_etl_pipeline',
    default_args=default_args,
    description='accidents etl pipeline',
)
with DAG(
        dag_id='accidents_etl_pipeline',
        schedule_interval='@once',
        default_args=default_args,
        tags=['accidents-pipeline'],
) as dag:
    cleaning = PythonOperator(
        task_id='cleaning',
        python_callable=task1,
    )
    add_feature = PythonOperator(
        task_id='add_feature',
        python_callable=task2,
    )
    load_to_postgres_task = PythonOperator(
        task_id='load_to_postgres',
        python_callable=task3,
    )
    create_dashboard_task = PythonOperator(
        task_id='create_dashboard_task',
        python_callable=task4,

    )

    cleaning >> add_feature >> create_dashboard_task
    add_feature >> load_to_postgres_task
