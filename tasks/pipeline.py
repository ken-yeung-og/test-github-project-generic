# pylint: disable=E1101

import os
import yaml
import pathlib as pl
import sagemaker
import sagemaker.session
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.inputs import TrainingInput
from sagemaker.metadata_properties import MetadataProperties
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_experiment_config import PipelineExperimentConfig
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, CacheConfig, TuningStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner
from sagemaker.xgboost import XGBoostPredictor

BASE_DIR = pl.Path(__file__).resolve().parent


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def is_file_empty(file_path):
    """Check if the given file is empty."""
    if not os.path.exists(file_path):
        return True
    elif os.stat(file_path).st_size == 0:
        return True
    return False


def get_pipeline(config):
    # Load configuration from config.yaml
    instance_type = config["INSTANCE_TYPE"]  # ml.m5.xlarge
    instance_count = config["INSTANCE_COUNT"]  # 1
    base_job_name = config["BASE_JOB_NAME"]  # "XGBTraffic"
    input_data = config[
        "INPUT_DATA_S3_URI"
    ]  # "s3://og-407-temp-test-data/austin-traffic/Radar_Traffic_Counts_20240528.csv"
    output_path = config["OUTPUT_PATH"]  # "/opt/ml/processing"
    xgboost_framework_version = config["XGBOOST_FRAMEWORK_VERSION"]  # "1.7-1"
    xgboost_py_version = config["XGBOOST_PY_VERSION"]  # "py3"
    evaluation_threshold = config["EVALUATION_THRESHOLD"]  # "1"
    hyperparameters = config["HYPERPARAMETERS"]
    cache_preprocess = config["CACHE_PREPROCESS"]
    cache_training = config["CACHE_TRAINING"]
    max_jobs = config["TUNE_MAX_JOBS"]
    max_parallel_jobs = config["TUNE_MAX_PARALLEL_JOBS"]
    strategy = config["TUNE_STRATEGY"]

    # Define session
    pipeline_session = PipelineSession()

    # Define env variables
    aws_region = os.environ.get("AWS_DEFAULT_REGION", pipeline_session.boto_region_name)
    role = os.environ.get(
        "SM_PIPELINE_ROLE_ARN", sagemaker.session.get_execution_role()
    )
    kms_key_alias = os.environ.get("SM_STUDIO_KMS_KEY_ALIAS", None)
    sm_project_name = os.environ.get("SM_PROJECT_NAME", "local_project")
    sm_project_id = os.environ.get("SM_PROJECT_ID", "p-012345")
    mg_name = os.environ.get("MPG_NAME", "local_model_group")
    commit = os.environ.get("COMMIT_ID", "0123456789")
    repo_name = os.environ.get("REPO_NAME", "model-local")

    cache_config = CacheConfig(enable_caching=True, expire_after="PT8H")

    # Preprocessing Step
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=instance_type,
        instance_count=instance_count,
        base_job_name=f"{sm_project_name}/{base_job_name}-preprocess",
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_process = ProcessingStep(
        name=f"{base_job_name}Preprocess",
        processor=sklearn_processor,
        outputs=[
            ProcessingOutput(output_name="train", source=f"{output_path}/train"),
            ProcessingOutput(
                output_name="validation", source=f"{output_path}/validation"
            ),
            ProcessingOutput(output_name="test", source=f"{output_path}/test"),
        ],
        code=str(BASE_DIR.joinpath("preprocess.py")),
        job_arguments=["--input-data", input_data],
        cache_config=cache_config if cache_preprocess else None,
    )

    # Training Step
    model_prefix = f"{sm_project_name}/{base_job_name}-train"
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=aws_region,
        version=xgboost_framework_version,
        py_version=xgboost_py_version,
        instance_type=instance_type,
    )

    xgb_train = Estimator(
        entry_point=str(BASE_DIR.joinpath("train.py"))
        if not is_file_empty(str(BASE_DIR.joinpath("train.py")))
        else None,
        image_uri=image_uri,
        instance_type=instance_type,
        instance_count=instance_count,
        output_path=f"s3://{pipeline_session.default_bucket()}/{model_prefix}",
        output_kms_key=kms_key_alias,
        base_job_name=f"{sm_project_name}/{base_job_name}-train",
        sagemaker_session=pipeline_session,
        role=role,
    )

    xgb_train.set_hyperparameters(
        objective="reg:squarederror",  # Define the object metric for the training job
    )

    # Tuning Step
    objective_metric_name = "validation:rmse"
    hyperparameter_ranges = {
        "alpha": ContinuousParameter(
            hyperparameters["alpha"]["min"],
            hyperparameters["alpha"]["max"],
            scaling_type=hyperparameters["alpha"]["scaling_type"],
        ),
        "lambda": ContinuousParameter(
            hyperparameters["lambda"]["min"],
            hyperparameters["lambda"]["max"],
            scaling_type=hyperparameters["lambda"]["scaling_type"],
        ),
        "num_round": IntegerParameter(
            hyperparameters["num_round"]["min"],
            hyperparameters["num_round"]["max"],
            scaling_type=hyperparameters["num_round"]["scaling_type"],
        ),
        "max_depth": IntegerParameter(
            hyperparameters["max_depth"]["min"],
            hyperparameters["max_depth"]["max"],
            scaling_type=hyperparameters["max_depth"]["scaling_type"],
        ),
        "eta": ContinuousParameter(
            hyperparameters["eta"]["min"],
            hyperparameters["eta"]["max"],
            scaling_type=hyperparameters["eta"]["scaling_type"],
        ),
        "gamma": ContinuousParameter(
            hyperparameters["gamma"]["min"],
            hyperparameters["gamma"]["max"],
            scaling_type=hyperparameters["gamma"]["scaling_type"],
        ),
        "min_child_weight": IntegerParameter(
            hyperparameters["min_child_weight"]["min"],
            hyperparameters["min_child_weight"]["max"],
            scaling_type=hyperparameters["min_child_weight"]["scaling_type"],
        ),
        "subsample": ContinuousParameter(
            hyperparameters["subsample"]["min"],
            hyperparameters["subsample"]["max"],
            scaling_type=hyperparameters["subsample"]["scaling_type"],
        ),
    }
    metric_definitions = [
        {"Name": objective_metric_name, "Regex": "Validation RMSE: ([0-9\\.]+)"}
    ]
    tuner_log = HyperparameterTuner(
        xgb_train,
        objective_metric_name,
        hyperparameter_ranges,
        metric_definitions,
        max_jobs=max_jobs,
        max_parallel_jobs=max_parallel_jobs,
        strategy=strategy,
        objective_type="Minimize",
        early_stopping_type="Auto",
    )

    hpo_args = tuner_log.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        }
    )

    step_tuning = TuningStep(
        name=f"{base_job_name}Tuning",
        step_args=hpo_args,
        cache_config=cache_config if cache_training else None,
    )

    best_model = Model(
        image_uri=image_uri,
        model_data=step_tuning.get_top_model_s3_uri(
            top_k=0, s3_bucket=pipeline_session.default_bucket(), prefix=model_prefix
        ),
        predictor_cls=XGBoostPredictor,
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_create_first = ModelStep(
        name="CreateBestModel",
        step_args=best_model.create(instance_type="ml.m5.xlarge"),
    )
    # Evaluation Step
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=instance_type,
        instance_count=instance_count,
        base_job_name=f"{sm_project_name}/{base_job_name}-eval",
        sagemaker_session=pipeline_session,
        role=role,
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )
    step_eval = ProcessingStep(
        name=f"{base_job_name}Eval",
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=step_tuning.get_top_model_s3_uri(
                    top_k=0,
                    s3_bucket=pipeline_session.default_bucket(),
                    prefix=model_prefix,
                ),
                destination=f"{output_path}/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination=f"{output_path}/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation", source=f"{output_path}/evaluation"
            ),
        ],
        code=str(BASE_DIR.joinpath("evaluation.py")),
        property_files=[evaluation_report],
    )

    # Create ModelMetrics and pass to RegisterModel
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0][
                        "S3Output"
                    ]["S3Uri"],
                    "evaluation.json",
                ],
            ),
            content_type="application/json",
        )
    )

    step_register_best = RegisterModel(
        name=f"{base_job_name}RegisterBestModel",
        estimator=xgb_train,
        model=best_model,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=f"{mg_name}-{base_job_name}",
        approval_status="PendingManualApproval",
        description=f"commit_id={commit}",
        customer_metadata_properties={
            "SM_PROJECT_NAME": sm_project_name,
            "SM_PROJECT_ID": sm_project_id,
            "INSTANCE_TYPE": instance_type,
            "INSTANCE_COUNT": str(instance_count),
            "ENV_EXAMPLE": "VariablesStartWithENV_PassedToEndpointRuntime",
        },
        model_metrics=model_metrics,
        metadata_properties=MetadataProperties(
            commit_id=commit, project_id=sm_project_id, repository=repo_name
        ),
    )

    # Conditional Registration Step
    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.mse.value",
        ),
        right=evaluation_threshold,
    )

    step_cond_registration = ConditionStep(
        name=f"{base_job_name}MSECond",
        conditions=[cond_lte],
        if_steps=[step_register_best],
        else_steps=[],
    )

    # Define Pipeline instance
    pipeline = Pipeline(
        name=f"{mg_name}-{base_job_name}",
        steps=[
            step_process,
            step_tuning,
            step_create_first,
            step_eval,
            step_cond_registration,
        ],
        sagemaker_session=pipeline_session,
        pipeline_experiment_config=PipelineExperimentConfig(
            experiment_name=f"{mg_name}-{base_job_name}",
            trial_name=f"{commit}-{base_job_name}",
        ),
    )

    return pipeline


if __name__ == "__main__":
    config_path = "config.yaml"
    config = load_config(config_path)
    pipeline = get_pipeline(config)
