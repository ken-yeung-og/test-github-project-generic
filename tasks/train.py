import argparse
import logging
import os
import pickle as pkl
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Set up logging
logging.basicConfig(level=logging.INFO)

TRAIN_VALIDATION_FRACTION = 0.2
RANDOM_STATE_SAMPLING = 200


def prepare_data(train_dir, validation_dir):
    """Read data from train and validation channels, return features and target variables."""
    # Read and shuffle train data
    df_train = pd.read_csv(os.path.join(train_dir, "train.csv"), header=None)
    df_train = df_train.sample(frac=1, random_state=RANDOM_STATE_SAMPLING)
    df_train.columns = ["target"] + [
        f"feature_{x}" for x in range(df_train.shape[1] - 1)
    ]
    logging.info(f"df_train shape: {df_train.shape}")

    try:
        # Read validation data if available
        df_validation = pd.read_csv(
            os.path.join(validation_dir, "validation.csv"), header=None
        )
        df_validation.columns = ["target"] + [
            f"feature_{x}" for x in range(df_validation.shape[1] - 1)
        ]
        logging.info(f"df_validation shape: {df_validation.shape}")
    except FileNotFoundError:
        # If no validation data, sample from train data
        logging.info(
            f"Validation data not found. Sampling {TRAIN_VALIDATION_FRACTION * 100}% of training data as validation."
        )
        df_validation = df_train.sample(
            frac=TRAIN_VALIDATION_FRACTION, random_state=RANDOM_STATE_SAMPLING
        )
        df_train = df_train.drop(df_validation.index).reset_index(drop=True)
        df_validation.reset_index(drop=True, inplace=True)

    # Split into features and target
    X_train, y_train = df_train.iloc[:, 1:], df_train.iloc[:, :1]
    X_val, y_val = df_validation.iloc[:, 1:], df_validation.iloc[:, :1]
    logging.info(f"Train shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    logging.info(f"Validation shapes: X_val={X_val.shape}, y_val={y_val.shape}")

    return X_train.values, y_train.values, X_val.values, y_val.values


def main():
    """Run the training pipeline."""
    parser = argparse.ArgumentParser()

    # Add hyperparameters and file paths
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--lambda", type=float, dest="lambda_param")
    parser.add_argument("--num_round", type=int)
    parser.add_argument("--max_depth", type=int)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--min_child_weight", type=int)
    parser.add_argument("--subsample", type=float)
    parser.add_argument("--objective", type=str)
    parser.add_argument("--tree_method", type=str, default="auto")
    parser.add_argument("--learning_rate", type=str, default="auto")
    parser.add_argument(
        "--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR")
    )
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument(
        "--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION")
    )

    args = parser.parse_args()

    # Prepare data
    X_train, y_train, X_val, y_val = prepare_data(args.train, args.validation)

    # XGBoost parameters
    params = {
        "booster": "gbtree",
        "objective": args.objective,
        "learning_rate": args.eta,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "reg_lambda": args.lambda_param,
        "reg_alpha": args.alpha,
        "eval_metric": "rmse",
    }

    # Train the model
    bst = xgb.XGBRegressor(**params)
    bst.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = bst.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    logging.info(f"Validation RMSE: {rmse}")

    model_location = args.model_dir + "/xgboost-model"
    pkl.dump(bst.get_booster(), open(model_location, "wb"))
    logging.info("Stored trained model at {}".format(model_location))


if __name__ == "__main__":
    main()
