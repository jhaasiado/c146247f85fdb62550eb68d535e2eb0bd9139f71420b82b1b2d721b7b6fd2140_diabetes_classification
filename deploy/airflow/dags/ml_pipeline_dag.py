"""Skeleton Airflow DAG for the diabetes pipeline.

This DAG demonstrates how you might orchestrate the pipeline steps.
Fill in operators and connections as needed for your environment.
"""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="diabetes_ml_pipeline",
    description="Run preprocessing, training, and evaluation",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,  # trigger manually or set a cron
    catchup=False,
) as dag:

    # Example using the same command you run locally
    preprocess = BashOperator(
        task_id="preprocess",
        bash_command='uv run python -c "from src.data_preprocessing import preprocess_data; preprocess_data()"',
    )

    train = BashOperator(
        task_id="train",
        bash_command='uv run python -c "from src.training import run_training; run_training()"',
    )

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command=(
            'uv run python -c "'
            "from src.data_preprocessing import preprocess_data; "
            "from src.training import train_model; "
            "from src.evaluation import evaluate_model; "
            "import pandas as pd; from pathlib import Path; "
            "X = pd.read_csv('data/preprocessed/diabetes_scaled.csv'); "
            "y = X.pop('Outcome'); "
            "models = train_model(X, y); evaluate_model(models, X, y)"
            '"'
        ),
    )

    preprocess >> train >> evaluate
