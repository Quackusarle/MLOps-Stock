FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY services/ensemble_api/ /app/

# Weights sẽ được tải từ MLflow/MinIO khi container khởi động
ENV PYTHONPATH="/app"
ENV MLFLOW_TRACKING_URI="http://mlflow-local.mlops-infra.svc.cluster.local:5000"
ENV MLFLOW_S3_ENDPOINT_URL="http://minio-svc.mlops-infra.svc.cluster.local:9000"

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
