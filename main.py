import logging

# Pipeline steps
from src.data_preprocessing import preprocess_data
from src.evaluation import evaluate_model
from src.training import save_models, train_model


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main() -> None:
    setup_logging()
    logger = logging.getLogger("pipeline")

    # 1) Preprocess raw data (scale + split)
    logger.info("Starting preprocessing...")
    X_train, X_test, y_train, y_test = preprocess_data()
    logger.info(
        "Preprocessing complete. Shapes -> X_train: %s, X_test: %s", X_train.shape, X_test.shape
    )

    # 2) Train models (RF, XGBoost, Logistic Regression, KNN) on the training split
    logger.info("Training models (RF, XGBoost, LR, KNN)...")
    models = train_model(X_train, y_train)
    logger.info("Training complete: %s", ", ".join(models.keys()))

    # 2.1) Save trained models as pickle files
    models_dir = save_models(models)
    logger.info("Saved trained models to: %s", models_dir)

    # 3) Evaluate on the test split and write artifacts
    logger.info("Evaluating models and writing artifacts...")
    out_dir = evaluate_model(models, X_test, y_test)
    logger.info("Artifacts written to: %s", out_dir)


if __name__ == "__main__":
    main()
