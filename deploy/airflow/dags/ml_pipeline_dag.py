"""Airflow DAG using PythonOperator to orchestrate the ML pipeline.

Notes
- This DAG assumes the Airflow worker has the project code and dependencies
  available/importable. We extend sys.path for convenience when the DAG lives
  under `deploy/airflow/dags`.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# Ensure project root is importable (…/deploy/airflow/dags → root is parents[3])
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def task_preprocess() -> None:
    """Generate the preprocessed dataset and train/test split artifacts."""
    from src.data_preprocessing import preprocess_data

    preprocess_data()


def task_train() -> None:
    """Train models and persist them under `models/`."""
    from src.training import run_training

    run_training()


def task_evaluate() -> None:
    """Evaluate models and write metrics/reports under `results/`."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from src.data_preprocessing import DEFAULT_PREPROCESSED_PATH, DEFAULT_TARGET
    from src.evaluation import evaluate_model
    from src.training import train_model

    df = pd.read_csv(DEFAULT_PREPROCESSED_PATH)
    y = df[DEFAULT_TARGET]
    X = df.drop(columns=[DEFAULT_TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = train_model(X_train, y_train, random_state=42)
    evaluate_model(models, X_test, y_test)


with DAG(
    dag_id="diabetes_ml_pipeline",
    description="Run preprocessing, training, and evaluation using PythonOperator",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,  # trigger manually or set a cron
    catchup=False,
) as dag:

    preprocess = PythonOperator(task_id="preprocess", python_callable=task_preprocess)
    train = PythonOperator(task_id="train", python_callable=task_train)
    evaluate = PythonOperator(task_id="evaluate", python_callable=task_evaluate)

    preprocess >> train >> evaluate
