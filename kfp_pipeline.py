from kfp import dsl
from kfp import compiler

import os

# Define the base docker image built from Dockerfile.training
# Lấy image từ Github Action pass qua. Mặc định là stock-trainer:latest nếu test thủ công.
BASE_IMAGE = os.environ.get('TRAINER_IMAGE', 'stock-trainer:latest')

@dsl.container_component
def train_stock_model(
    symbol: str
):
    """
    KFP Component to run the stock predictor training container
    """
    return dsl.ContainerSpec(
        image=BASE_IMAGE,
        command=["python", "src/training/final_ensemble_train.py"],
        args=["--symbol", symbol]
    )

@dsl.pipeline(
    name="stock-prediction-training-pipeline",
    description="Trains the ensemble models for multiple stock symbols concurrently."
)
def stock_training_pipeline(
    symbols: list = ["VNM", "VCB", "HPG", "FPT"]
):
    # Iterate over symbols to launch parallel training tasks
    with dsl.ParallelFor(symbols) as item:
        train_task = train_stock_model(symbol=item)
        train_task.set_env_variable('MLFLOW_TRACKING_URI', 'http://mlflow-local.mlops-infra.svc.cluster.local:5000')
        train_task.set_env_variable('MLFLOW_S3_ENDPOINT_URL', 'http://minio-svc.mlops-infra.svc.cluster.local:9000')
        train_task.set_env_variable('AWS_ACCESS_KEY_ID', 'minioadmin')
        train_task.set_env_variable('AWS_SECRET_ACCESS_KEY', 'nammoadidaphat')

if __name__ == '__main__':
    # Compile the pipeline definition into a YAML file
    compiler.Compiler().compile(
        pipeline_func=stock_training_pipeline,
        package_path='pipeline.yaml'
    )
    print("Pipeline successfully compiled to pipeline.yaml")
