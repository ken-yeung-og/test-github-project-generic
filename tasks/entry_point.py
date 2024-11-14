import os
import pickle as pkl

import numpy as np
import xgboost

def model_fn(model_dir):
    """
    Deserialize and return the fitted model.
    """
    model_file = "xgboost-model"
    model = pkl.load(open(os.path.join(model_dir, model_file), "rb"))
    return model


def input_fn(request_body, request_content_type):
    """
    Process the request data body and content type, and return a NumPy array.
    """
    if request_content_type == "text/csv":
        # Clean the input data and remove any newline characters or extra spaces
        data = request_body.strip().split("\n")
        data = [line.strip().split(",") for line in data]
        data = np.array(data, dtype=float)
        print(f"len(data): {len(data)}")
        return data
    else:
        raise ValueError(
            "Content type {} is not supported.".format(request_content_type)
        )


def predict_fn(input_data, model):
    """
    Make a prediction using the model and input data.
    """
    prediction = model.predict(xgboost.DMatrix(input_data))
    print(f"len(prediction): {len(prediction)}")
    return prediction


def output_fn(predictions, content_type):
    """
    Convert the prediction result to the desired output format.
    """
    if content_type == "text/csv":
        results = "\n".join(map(str, predictions))
        return results
    else:
        raise ValueError("Content type {} is not supported.".format(content_type))
