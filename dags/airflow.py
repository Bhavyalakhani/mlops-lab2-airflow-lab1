# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, build_save_model, load_model_elbow

# Define default arguments for your DAG
default_args = {
    'owner': 'your_name',
    'start_date': datetime(2025, 1, 15),
    'retries': 0,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5),  # Delay before retries
}

with DAG(
    'wine_dataset_clustering_airflow',
    default_args=default_args,
    description='Wine Dataset Clustering with DAG',
    catchup=False,
) as dag:

    load_data_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data,
    )

    data_preprocessing_task = PythonOperator(
        task_id='data_preprocessing_task',
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
    )

    build_save_model_task = PythonOperator(
        task_id='build_save_model_task',
        python_callable=build_save_model,
        op_args=[data_preprocessing_task.output, "wine_dataset_classification_model.pkl"],
    )

    load_model_task = PythonOperator(
        task_id='load_model_task',
        python_callable=load_model_elbow,
        op_args=["wine_dataset_classification_model.pkl", build_save_model_task.output],
    )

    load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task

if __name__ == "__main__":
    dag.test()
