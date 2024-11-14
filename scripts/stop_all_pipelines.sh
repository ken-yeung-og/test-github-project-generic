#!/bin/bash

project=$1
# Get a list of all SageMaker pipelines
pipelines=$(aws sagemaker list-pipelines --query 'PipelineSummaries[].PipelineName' --output text)
filtered_pipelines=$(printf "%s\n" "${pipelines[@]}" | grep -E "^$project")

# Loop through each pipeline
for pipeline in $filtered_pipelines; do
    echo "Pipeline: $pipeline"

    # Get a list of all executions for the current pipeline
    executions=$(aws sagemaker list-pipeline-executions --pipeline-name "$pipeline" --query 'PipelineExecutionSummaries[?PipelineExecutionStatus==`Executing`].PipelineExecutionArn' --output text)

    # Loop through each execution and stop the running executions
    for execution in $executions; do
        echo "    Stopping execution: $execution"
        stopped=$(aws sagemaker stop-pipeline-execution --pipeline-execution-arn "$execution" --query 'PipelineExecutionArn')
        echo "    Stopped execution: $stopped"
    done
done
