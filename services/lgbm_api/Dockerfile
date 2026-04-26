# ============================================================
# LightGBM API — Dự đoán bằng GBDT, tải weights từ MLflow/MinIO
# ============================================================
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app"

# Defaults cho in-cluster communication (override bằng K8s env khi deploy)
ENV MLFLOW_TRACKING_URI="http://mlflow-local.mlops-infra.svc.cluster.local:5000"
ENV MLFLOW_S3_ENDPOINT_URL="http://minio-svc.mlops-infra.svc.cluster.local:9000"

WORKDIR /app

# Layer 1: Dependencies (cached — lightgbm nhẹ hơn torch rất nhiều)
COPY services/lgbm_api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Layer 2: Shared source modules
COPY src/ /app/src/

# Layer 3: Service code
COPY services/lgbm_api/ /app/

# Security: Non-root user
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

EXPOSE 8003
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
