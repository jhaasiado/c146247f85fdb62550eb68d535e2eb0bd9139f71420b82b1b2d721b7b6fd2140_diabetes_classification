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
from typing import Dict

from airflow import DAG
from airflow.operators.python import PythonOperator

# Ensure project root is importable (…/deploy/airflow/dags → root is parents[3])
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def task_preprocess() -> str:
    """Generate the preprocessed dataset and return its path via XCom."""
    from src.data_preprocessing import DEFAULT_PREPROCESSED_PATH, preprocess_data

    preprocess_data()
    return str(DEFAULT_PREPROCESSED_PATH)


def task_train(ti) -> str:
    """Train models and persist them under `models/`.

    Returns the models directory path via XCom for downstream tasks.
    """
    from src.training import MODELS_DIR, run_training

    _ = ti.xcom_pull(task_ids="preprocess")
    run_training()
    return str(MODELS_DIR)


def task_evaluate(ti) -> str:
    """Evaluate saved models and write metrics under `results/`.

    Loads the preprocessed CSV and the persisted models; uses a deterministic
    split for comparability. Returns the results directory path.
    """
    import joblib
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from src.data_preprocessing import DEFAULT_PREPROCESSED_PATH, DEFAULT_TARGET
    from src.evaluation import evaluate_model

    _ = ti.xcom_pull(task_ids="train")

    df = pd.read_csv(DEFAULT_PREPROCESSED_PATH)
    y = df[DEFAULT_TARGET]
    X = df.drop(columns=[DEFAULT_TARGET])
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models_dir = Path("models")
    models: Dict[str, object] = {}
    for mf in models_dir.glob("model_*.pkl"):
        models[mf.stem.replace("model_", "")] = joblib.load(mf)

    results_dir = evaluate_model(models, X_test, y_test)
    return str(results_dir)


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
