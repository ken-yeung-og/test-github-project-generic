# Generic Project Template

## Project Structure

```
├── config.yaml                # Configuration file for the pipeline
├── run_pipeline.py            # Main script to run the entire pipeline
├── run_inference.py           # Main script to run the inference pipeline
├── requirements.txt           # Dependencies required for the project
├── tasks                      # Directory for different pipeline tasks
│   ├── inference.py           # Batch inference pipeline definition
│   ├── pipeline.py            # Script to orchestrate the pipeline tasks
│   ├── preprocess.py          # Preprocessing steps
│   ├── entry_point.py         # Batch inference steps
│   ├── train.py               # Training steps (Optional)
│   ├── hyptune.py             # Hyperparameter tuning steps (Optional)
│   ├── evaluation.py          # Evaluation steps
├── custom_image               # Scripts for building custom Docker image (Optional)
│   ├── Dockerfile             # Dockerfile to define the custom image (Optional)
│   ├── build_image.sh         # Script to build and push the Docker image (Optional)
├── local_pipeline.ipynb       # Notebook to run pipeline locally
└── README.md                  # Project documentation
```

### Explanation

- **config.yaml**: Remains as the configuration file for the pipeline, centralizing all configurations for easy adjustments.

- **run_pipeline.py**: Main script to run the entire pipeline, serving as the entry point for executing the full pipeline workflow.

- **run_inference.py**: Man script manages the executions of batch inference pipeline. It loads configurations from a YAML file and starts pipeline executions.

- **requirements.txt**: Lists all dependencies required for the project, ensuring reproducibility and easy setup of the environment.

- **tasks**: Directory housing scripts for each stage of the pipeline:
  - **pipeline.py**: Orchestrates the entire pipeline, ensuring tasks run in the correct sequence.
  - **preprocess.py**: Contains data preprocessing logic.
  - **train.py**: Handles model training procedures (Optional).
  - **hyptune.py**: Manages hyperparameter tuning (Optional).
  - **evaluation.py**: Evaluates model performance using defined metrics.
  - **inference.py**: Orchestrates the batch transform pipeline, ensuring tasks run in the correct sequence.
  - **entry_point.py**: Model serving script for SageMaker endpoint predictions and output formatting.

- **custom_image**: Directory for Docker-related scripts to build a custom Docker image (Optional):
  - **Dockerfile**: Defines the custom Docker image (Optional).
  - **build_image.sh**: Script to build and push the Docker image to a container registry (Optional).

- **local_pipeline.ipynb**: A Jupyter notebook for running and testing the pipeline locally. 

- **README.md**: Comprehensive project documentation, providing setup instructions, an overview of the project structure, and usage guidelines.
