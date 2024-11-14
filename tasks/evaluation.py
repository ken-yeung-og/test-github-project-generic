import json
import logging
import pathlib
import pickle
import tarfile

import numpy as np
import pandas as pd
import xgboost

from sklearn.metrics import mean_squared_error

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def load_model(model_path):
    """Load XGBoost model from the given tarball."""
    try:
        logger.debug(f"Opening model tarball from {model_path}")
        with tarfile.open(model_path) as tar:
            tar.extractall(path=".")
            logger.info(f"Model extracted successfully from {model_path}")
    except Exception as e:
        logger.error(f"Error extracting model: {str(e)}")
        raise

    try:
        model = pickle.load(open("xgboost-model", "rb"))
        logger.info("XGBoost model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading the model: {str(e)}")
        raise
    return model


def load_data(test_path):
    """Load test data from the given path."""
    try:
        logger.info(f"Reading test data from {test_path}")
        df_test = pd.read_csv(
            test_path,
            header=None,
        )
        df_test = df_test.iloc[np.random.permutation(len(df_test))]
        df_test.columns = ["target"] + [
            f"feature_{x}" for x in range(df_test.shape[1] - 1)
        ]
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise
    return df_test


def evaluate_model(model, X_test, y_test):
    """Evaluate the model with RMSE and MSE."""
    try:
        logger.info("Performing predictions on test data.")
        y_pred = model.predict(xgboost.DMatrix(X_test))
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        logger.info(f"Test RMSE: {rmse}")
    except Exception as e:
        logger.error(f"Error during model prediction: {str(e)}")
        raise

    return y_pred


if __name__ == "__main__":
    logger.debug("Starting evaluation.")

    model_path = "/opt/ml/processing/model/model.tar.gz"
    test_path = "/opt/ml/processing/test/test.csv"
    output_dir = "/opt/ml/processing/evaluation"

    # Load model
    model = load_model(model_path)

    # Load test data
    df_test = load_data(test_path)
    X_test, y_test = df_test.iloc[:, 1:].values, df_test.iloc[:, 0].values.flatten()

    # Evaluate model and predict
    y_pred = evaluate_model(model, X_test, y_test)

    logger.debug("Calculating mean squared error and standard deviation.")
    mse = mean_squared_error(y_test, y_pred)
    std = np.std(y_test - y_pred)

    # Generate evaluation report
    report_dict = {
        "regression_metrics": {
            "mse": {"value": mse, "standard_deviation": std},
        },
    }

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    try:
        with open(evaluation_path, "w") as f:
            json.dump(report_dict, f)
        logger.info(f"Evaluation report saved to {evaluation_path}")
    except Exception as e:
        logger.error(f"Error writing the evaluation report: {str(e)}")
        raise
