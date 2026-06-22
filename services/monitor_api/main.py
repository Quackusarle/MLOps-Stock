import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import threading
import numpy as np
import pandas as pd
from fastapi import FastAPI
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.data_pipeline.yahoo_data import YahooData

app = FastAPI(title="Model Monitor - Drift Detection Service")

# ============================================================
# Prometheus Metrics (Gauges vì giá trị drift có thể tăng/giảm)
# ============================================================
drift_score_gauge = Gauge(
    "model_drift_score",
    "Overall data drift score (0-1, higher = more drift)",
    ["ticker"]
)
drift_detected_gauge = Gauge(
    "model_drift_detected",
    "Whether data drift was detected (1=yes, 0=no)",
    ["ticker"]
)
drifted_features_gauge = Gauge(
    "model_drifted_features_count",
    "Number of features that have drifted",
    ["ticker"]
)
prediction_spread_gauge = Gauge(
    "model_prediction_spread",
    "Spread between TFT and LGBM predictions (uncertainty indicator)",
    ["ticker"]
)

# ============================================================
# Symbols to monitor
# ============================================================
MONITOR_SYMBOLS = os.getenv("MONITOR_SYMBOLS", "VNM,FPT,VCB,HPG").split(",")
REFERENCE_DAYS = int(os.getenv("REFERENCE_DAYS", "120"))
CURRENT_DAYS = int(os.getenv("CURRENT_DAYS", "30"))
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL_SECONDS", "3600"))  # Mặc định 1 giờ

FEATURES = [
    'open', 'high', 'low', 'close', 'volume',
    'sma_10', 'sma_20', 'rsi', 'macd', 'macd_signal',
    'bb_upper', 'bb_lower', 'log_return'
]


def compute_drift(symbol: str) -> dict:
    """
    Tính Data Drift cho một mã cổ phiếu bằng Evidently AI.
    So sánh dữ liệu "cũ" (reference) với dữ liệu "mới" (current).
    """
    try:
        provider = YahooData()
        
        # Lấy dữ liệu gộp (reference + current)
        total_days = REFERENCE_DAYS + CURRENT_DAYS + 30  # buffer
        df = provider.get_historical_data(symbol, days=total_days)
        
        if df is None or len(df) < REFERENCE_DAYS + CURRENT_DAYS:
            print(f"[Drift] Not enough data for {symbol}, skipping")
            return None
        
        # Chia thành Reference (dữ liệu cũ) và Current (dữ liệu mới)
        reference_data = df.iloc[:REFERENCE_DAYS][FEATURES].reset_index(drop=True)
        current_data = df.iloc[-CURRENT_DAYS:][FEATURES].reset_index(drop=True)
        
        # Chạy Evidently Report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)
        
        result = report.as_dict()
        
        # Trích xuất kết quả từ Evidently
        drift_metrics = result["metrics"][0]["result"]
        
        drift_info = {
            "symbol": symbol,
            "drift_score": drift_metrics.get("share_of_drifted_columns", 0),
            "drift_detected": drift_metrics.get("dataset_drift", False),
            "number_of_drifted_columns": drift_metrics.get("number_of_drifted_columns", 0),
            "total_columns": drift_metrics.get("number_of_columns", len(FEATURES)),
        }
        
        print(f"[Drift] {symbol}: score={drift_info['drift_score']:.2f}, "
              f"detected={drift_info['drift_detected']}, "
              f"drifted_cols={drift_info['number_of_drifted_columns']}/{drift_info['total_columns']}")
        
        return drift_info
        
    except Exception as e:
        print(f"[Drift] Error computing drift for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def update_metrics():
    """Cập nhật toàn bộ Prometheus metrics cho tất cả các mã."""
    print(f"[Monitor] Starting drift check for {MONITOR_SYMBOLS}...")
    
    for symbol in MONITOR_SYMBOLS:
        symbol = symbol.strip().upper()
        drift_info = compute_drift(symbol)
        
        if drift_info:
            drift_score_gauge.labels(ticker=symbol).set(drift_info["drift_score"])
            drift_detected_gauge.labels(ticker=symbol).set(1 if drift_info["drift_detected"] else 0)
            drifted_features_gauge.labels(ticker=symbol).set(drift_info["number_of_drifted_columns"])
    
    print(f"[Monitor] Drift check complete.")


def background_worker():
    """Worker chạy ngầm, cứ mỗi CHECK_INTERVAL giây lại tính toán drift."""
    while True:
        try:
            update_metrics()
        except Exception as e:
            print(f"[Monitor] Background worker error: {e}")
        time.sleep(CHECK_INTERVAL)


@app.on_event("startup")
def startup():
    """Khởi động background thread để tính drift định kỳ."""
    # Chạy lần đầu tiên ngay khi khởi động
    thread = threading.Thread(target=background_worker, daemon=True)
    thread.start()
    print(f"[Monitor] Background drift checker started (interval={CHECK_INTERVAL}s)")


@app.get("/metrics")
def metrics():
    """Endpoint cho Prometheus scrape."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok", "service": "model-monitor"}


@app.get("/drift/{ticker}")
def get_drift(ticker: str):
    """API để xem drift thủ công cho một mã cụ thể."""
    drift_info = compute_drift(ticker.upper())
    if drift_info is None:
        return {"error": f"Could not compute drift for {ticker}"}
    return drift_info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8084)
