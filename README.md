# MLOps-Lab2-Airflow-Lab1

This repository contains a small Airflow DAG and helper tasks that demonstrate a simple ML workflow:

- load the Wine dataset
- preprocess (MinMax scaling)
- run KMeans across a range of K to compute SSE (inertia) and determine an elbow
- save scaler and model artifacts under `dags/model/` on docker image 

The code uses a JSON-safe XCom pattern (base64-encoded pickled payloads) so complex objects can be passed between PythonOperator tasks.

## Project structure (brief)

- `dags/` — Airflow DAG and helper code
  - `airflow.py` — DAG definition wiring the PythonOperator tasks
- `dags/src/` — Python task implementations used by the DAG
  - `lab.py` — load/preprocess/train/predict functions (also runnable as a script for quick local tests)
- `dags/model/` — where scaler/wine_dataset_classification_model.pkl artifacts are written at runtime (created by tasks) in the virtual machine started by docker
- `config/airflow.cfg` — Airflow configuration used for local development

## Setup (macOS / zsh)

- Please setup docker as per the OS you are using.
- [Docker Installation Guide](https://docs.docker.com/engine/install/)

1) Create and activate a virtual environment (recommended):

```bash
# create mlops-airflow-lab1 virtual environment (from repo root)
python3 -m venv mlops-airflow-lab1
source mlops-airflow-lab1/bin/activate
```

2) Install Python dependencies locally:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3) Start Airflow (docker-compose) — this repo expects you to run Airflow with Docker Compose.

```bash
# from repo root
docker compose up -d
# wait for Airflow webserver/scheduler to be ready
```

## Running via Airflow

Once the Airflow stack is running you can open the Airflow web UI at:

- http://localhost:8080
- username : airflow2
- password : airflow2

The DAG in this repository is named `wine_dataset_clustering_airflow`. In the UI you can:

- Find and enable (or trigger) `wine_dataset_clustering_airflow` on the DAGs page.
- Trigger a run and then open the DAG graph or tree view to click into task instances.
- View the logs for each PythonOperator task (load_data_task, data_preprocessing_task, build_save_model_task, load_model_task) to see the logger output and artifact paths.

Logs and task output are available in the UI by clicking a task instance and choosing "View Log".

## Shut Down

1) Once the pipeline has completed successfully run the following command :

```bash
docker compose down
```

2) This command shuts down the container that was launched.