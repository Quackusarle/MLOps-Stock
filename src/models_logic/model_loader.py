"""
Model Loader — Tải weights từ MLflow (MinIO backend) khi container khởi động.

Cách hoạt động:
1. Kết nối MLflow Tracking Server
2. Tìm Run mới nhất cho symbol cần dự đoán
3. Download artifacts (weights, scalers) về /tmp/models/<SYMBOL>/
4. Cache lại — không download lần 2 nếu đã có
"""

import os
import mlflow
from mlflow.tracking import MlflowClient


# Thư mục cache mặc định bên trong container
CACHE_DIR = os.getenv("MODELS_CACHE_DIR", "/tmp/models")


def download_model_artifacts(symbol: str) -> str:
    """
    Download tất cả artifacts của một symbol từ MLflow.
    
    Args:
        symbol: Mã cổ phiếu (vd: "FPT")
    
    Returns:
        Đường dẫn thư mục chứa các file model đã download.
        Ví dụ: /tmp/models/FPT/models/
    
    Raises:
        FileNotFoundError: Nếu không tìm thấy run nào cho symbol này.
        ConnectionError: Nếu không kết nối được MLflow.
    """
    sym = symbol.upper()
    dest_dir = os.path.join(CACHE_DIR, sym)
    artifacts_dir = os.path.join(dest_dir, "models")

    # Nếu đã cache, không download lại
    manifest_path = os.path.join(artifacts_dir, f"{sym}_artifact_manifest.json")
    if os.path.exists(manifest_path):
        return artifacts_dir

    os.makedirs(dest_dir, exist_ok=True)

    # Kết nối MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Tìm experiment
    experiment = client.get_experiment_by_name("stock_ensemble_training")
    if experiment is None:
        raise FileNotFoundError(
            f"Experiment 'stock_ensemble_training' not found on MLflow server {tracking_uri}"
        )

    # Tìm run mới nhất cho symbol này
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.symbol = '{sym}'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise FileNotFoundError(
            f"No MLflow run found for symbol '{sym}' in experiment 'stock_ensemble_training'"
        )

    run = runs[0]
    run_id = run.info.run_id
    print(f"[ModelLoader] Found run {run_id} for {sym}, downloading artifacts...")

    # Download toàn bộ artifact_path="models" về dest_dir
    # Kết quả: dest_dir/models/FPT_lgbm_model.pkl, ...
    client.download_artifacts(run_id, "models", dest_dir)

    print(f"[ModelLoader] Artifacts saved to {artifacts_dir}")
    return artifacts_dir
