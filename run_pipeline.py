import os
import yaml
import logging
from tasks.pipeline import get_pipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load the configuration from a YAML file."""
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML: {e}")
        raise


def merge_configs(base_config, model_config):
    """Merge the base configuration with the model-specific configuration."""
    merged_config = base_config.copy()
    merged_config.update(model_config)
    return merged_config


def execute_pipeline(pipeline, merged_config, execution_display_name, status_dict):
    """Start and monitor pipeline execution."""
    try:
        execution = pipeline.start(execution_display_name=execution_display_name)
        logger.info(f"Execution started with Name: {execution_display_name}")
        status_dict[execution.arn] = "Executing"

    except Exception as e:
        logger.error(f"Failed to execute pipeline {execution_display_name}: {e}")
        status_dict[execution.arn] = "Failed"


def process_pipeline(model_config, base_config, status_dict):
    """Merge configurations, set up the pipeline, and start execution."""
    merged_config = merge_configs(base_config, model_config)
    pipeline = get_pipeline(merged_config)

    # Get required environment variables
    role_arn = os.getenv("SM_PIPELINE_ROLE_ARN")
    project_name = os.getenv("SM_PROJECT_NAME")
    project_id = os.getenv("SM_PROJECT_ID")
    commit_id = os.getenv("COMMIT_ID", "unknown")

    if not role_arn or not project_name or not project_id:
        logger.error(
            "Environment variables SM_PIPELINE_ROLE_ARN, SM_PROJECT_NAME, or SM_PROJECT_ID are missing."
        )
        return

    # Upsert the pipeline
    try:
        pipeline.upsert(
            role_arn=role_arn,
            tags=[
                {"Key": "sagemaker:project-name", "Value": project_name},
                {"Key": "sagemaker:project-id", "Value": project_id},
            ],
        )
    except Exception as e:
        logger.error(f"Failed to upsert pipeline: {e}")
        return

    execution_display_name = f"commit-{commit_id}-{merged_config['BASE_JOB_NAME']}"

    # Start the pipeline execution
    execute_pipeline(pipeline, merged_config, execution_display_name, status_dict)


def main():
    # Load the configuration file
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    config = load_config(config_path)

    base_config = config["base_config"]
    models_config = config["models"]
    num_pipeline = len(models_config)

    logger.info(f"Total number of pipelines: {num_pipeline}")

    max_parallel_pipelines = base_config.get("MAX_PARALLEL_PIPELINES", 2)

    if num_pipeline > max_parallel_pipelines:
        # Print an error message and stop execution if too many pipelines
        logger.error(
            f"Too many pipelines to execute: {num_pipeline}. Maximum allowed parallel pipelines: {max_parallel_pipelines}."
        )
        return
    else:
        # Sequential execution if pipelines are fewer than the parallel limit
        for model_config in models_config:
            process_pipeline(model_config, base_config, {})


if __name__ == "__main__":
    main()
