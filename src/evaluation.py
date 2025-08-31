from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .utils import ensure_dir

RESULTS_DIR = Path("results")


def evaluate_model(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    results_dir: Path | str = RESULTS_DIR,
) -> Path:
    """Evaluate models on test data, store metrics and reports.

    Saves a `metrics.json` (accuracy, f1) and a `results.txt` with
    metrics and sklearn's classification report per model.
    Returns the results directory path.
    """

    results: Dict[str, Dict[str, float]] = {}
    text_lines = ["Model evaluation results:\n"]

    for name, model in models.items():
        # Prefer probabilities when available to derive class predictions
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        results[name] = {"accuracy": acc, "f1": f1}

        report = classification_report(y_test, y_pred)
        text_lines.append(f"- {name}: accuracy={acc:.4f}, f1={f1:.4f}")
        text_lines.append("\n" + report + "\n")

    out_dir = Path(results_dir)
    ensure_dir(out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "results.txt").write_text("\n".join(text_lines), encoding="utf-8")

    # Print a quick summary to stdout
    print("\n".join(text_lines))
    return out_dir
