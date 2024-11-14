# pylint: disable=E1101

import json
import logging
import os
import pathlib as pl
import time
from datetime import datetime

import boto3
import sagemaker
import sagemaker.session
import yaml

from sagemaker.model import Model
from sagemaker.transformer import Transformer

from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import TransformStep

from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.workflow.check_job_config import CheckJobConfig

from sagemaker.model_monitor import CronExpressionGenerator
from sagemaker.model_monitor import BatchTransformInput
from sagemaker.model_monitor import MonitoringDatasetFormat

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

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


def get_approved_model_arn_and_image_uri(session, mpg_name):
    approved_model_response = session.sagemaker_client.list_model_packages(
        ModelPackageGroupName=mpg_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
    )
    print(approved_model_response)

    model_package_arn = approved_model_response["ModelPackageSummaryList"][-1][
        "ModelPackageArn"
    ]
    model_package = session.sagemaker_client.describe_model_package(
        ModelPackageName=model_package_arn
    )

    model_data_url = model_package["InferenceSpecification"]["Containers"][0][
        "ModelDataUrl"
    ]
    image_uri = model_package["InferenceSpecification"]["Containers"][0]["Image"]

    return model_data_url, image_uri


def split_s3_uri(s3_uri):
    # Remove the "s3://" prefix
    if s3_uri.startswith("s3://"):
        s3_uri = s3_uri[5:]

    # Split the remaining string into bucket and key (path + file name)
    bucket, key = s3_uri.split("/", 1)

    # Split the key into prefix and file name
    *prefix_parts, file_name = key.split("/")
    prefix = "/".join(prefix_parts)

    return bucket, prefix, file_name


def get_pipeline(config):
    pipeline_session = PipelineSession()
    mpg_name = os.environ.get("MPG_NAME", "local_model_group")
    instance_type = config["INSTANCE_TYPE"]

    # Define env variables
    role = os.environ.get(
        "SM_PIPELINE_ROLE_ARN", sagemaker.session.get_execution_role()
    )
    batch_transform_input_s3_uri = config["BATCH_INPUT_S3_URI"]
    print(f"BATCH_INPUT_S3_URI: {batch_transform_input_s3_uri}")

    # Provided by deployment lambda
    model_group_name = mpg_name
    print(f"model_group_name: {model_group_name}")
    model_data_url, image_uri = get_approved_model_arn_and_image_uri(
        pipeline_session, model_group_name
    )
    model_package_name = model_data_url.split("/")[-2]

    sm_bucket_prefix = f"{pipeline_session.default_bucket()}/transform/{model_package_name}/{int(time.time())}"
    batch_transform_output_path = f"s3://{sm_bucket_prefix}/output"
    create_baseline = config["CREATE_BASELINE"]
    baseline_data_uri = config["INPUT_DATA_S3_URI"]
    baseline_results_uri = f"{baseline_data_uri}/data-quality-monitor-reports"
    bucket, prefix, file_name = split_s3_uri(baseline_data_uri)

    # Get the current date and time in GMT
    current_gmt = datetime.utcnow()

    # Format the new time as YYYY/MM/DD/HH
    formatted_time = current_gmt.strftime("%Y/%m/%d/%H")

    # Prepare JSON for S3 upload
    data_captured_destination_s3_json = [
        {"prefix": f"s3://{bucket}/{prefix}"},
        f"/{file_name}",
    ]
    print(f"Data to be captured in JSON: {data_captured_destination_s3_json}")
    # Define bucket and prefix for the data capture destination
    data_captured_destination_s3_uri_bucket = bucket
    data_captured_destination_s3_uri_prefix = f"{prefix}/datacapture"
    print(f"Data captured S3 URI bucket: {data_captured_destination_s3_uri_bucket}")
    print(f"Data captured S3 URI prefix: {data_captured_destination_s3_uri_prefix}")
    # Specify the file path where you want to save the JSON
    output_file_path = "data_captured_destination.json"
    data_captured_destination_s3_uri = f"s3://{data_captured_destination_s3_uri_bucket}/{data_captured_destination_s3_uri_prefix}"
    data_captured_destination_json_s3_uri = f"s3://{data_captured_destination_s3_uri_bucket}/{data_captured_destination_s3_uri_prefix}/input/{formatted_time}/{output_file_path}"

    print(
        f"Final S3 URI for data capture destination: {data_captured_destination_s3_uri}"
    )
    print(
        f"Final S3 URI for data capture json: {data_captured_destination_json_s3_uri}"
    )

    with open(output_file_path, "w") as json_file:
        json.dump(data_captured_destination_s3_json, json_file, indent=4)
    print(f"JSON data successfully written to {output_file_path}")

    # Define S3 client
    s3_client = boto3.client("s3")

    # Extract bucket name and object key from the S3 URI
    bucket_name = data_captured_destination_s3_uri_bucket
    object_key = f"{data_captured_destination_s3_uri_prefix}/input/{formatted_time}/{output_file_path}"
    print(f"Prepared S3 object key for upload: {object_key}")

    # Upload the file to S3
    try:
        s3_client.upload_file(output_file_path, bucket_name, object_key)
        print(f"File successfully uploaded to {data_captured_destination_json_s3_uri}")
    except Exception as e:
        print(f"An error occurred while uploading the file: {e}")

    print(f"Baseline results Output path: {baseline_results_uri}")
    print("Batch Transform Output path: {}".format(batch_transform_output_path))
    print(
        "Batch Transform Data Drift Monitoring Output path: {}".format(
            baseline_results_uri
        )
    )
    print(f"Identified the latest approved model package: {model_package_name}")

    transform_input_param = ParameterString(
        name="transform_input",
        default_value=batch_transform_input_s3_uri,
    )

    # Define the repacked model
    model = Model(
        model_data=model_data_url,
        image_uri=image_uri,
        role=role,
        sagemaker_session=pipeline_session,
        entry_point=str(BASE_DIR.joinpath("entry_point.py")),
    )

    model_step = ModelStep(
        name="RepackModelStep", step_args=model.create(instance_type=instance_type)
    )

    transformer = Transformer(
        model_name=model_step.properties.ModelName,
        instance_count=1,
        instance_type=instance_type,
        assemble_with="Line",
        accept="text/csv",
        output_path=batch_transform_output_path,
        sagemaker_session=pipeline_session,
    )

    transform_args = transformer.transform(
        batch_transform_input_s3_uri,
        content_type="text/csv",
        split_type="Line",
        input_filter="$[1:-1]",
        join_source="Input",
    )

    step_transform = TransformStep(
        name=f"{model_group_name}-transform", step_args=transform_args
    )

    # Initialize DefaultModelMonitor
    model_monitor = DefaultModelMonitor(
        role=role,
        instance_count=1,
        instance_type=instance_type,
        volume_size_in_gb=20,
    )
    if create_baseline:
        model_monitor.suggest_baseline(
            baseline_dataset=baseline_data_uri,
            dataset_format=DatasetFormat.csv(header=True),
            output_s3_uri=baseline_results_uri,
            wait=True,
            logs=False,
        )

    statistics_path = "{}/statistics.json".format(baseline_results_uri)
    constraints_path = "{}/constraints.json".format(baseline_results_uri)

    # Logging the statistics and constraints paths
    print(f"Statistics JSON path: {statistics_path}")
    print(f"Constraints JSON path: {constraints_path}")

    mon_schedule_name = f"{model_group_name}-data-drift-monitor-schedule"
    model_monitor.create_monitoring_schedule(
        monitor_schedule_name=mon_schedule_name,
        batch_transform_input=BatchTransformInput(
            data_captured_destination_s3_uri=data_captured_destination_s3_uri,
            destination="/opt/ml/processing/input",
            dataset_format=MonitoringDatasetFormat.csv(header=False),
        ),
        output_s3_uri=baseline_results_uri,
        statistics=statistics_path,
        constraints=constraints_path,
        schedule_cron_expression=CronExpressionGenerator.daily(),
        enable_cloudwatch_metrics=True,
    )

    desc_schedule_result = model_monitor.describe_schedule()
    print(desc_schedule_result)

    pipeline_name = f"{model_group_name}-inference"

    batch_inference_pipeline = Pipeline(
        name=pipeline_name,
        parameters=[transform_input_param],
        steps=[model_step, step_transform],
        sagemaker_session=pipeline_session,
    )

    return batch_inference_pipeline
