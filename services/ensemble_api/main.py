import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import httpx
import asyncio
from fastapi import FastAPI, HTTPException
import numpy as np
import joblib
from src.models_logic.decision_policy import build_decision, DecisionContext
from src.models_logic.model_loader import download_model_artifacts

app = FastAPI(title="Ensemble API Gateway")

# URLs của các Microservices
DATA_URL = os.getenv("DATA_SERVICE_URL", "http://localhost:8001/fetch/{}")
TFT_URL = os.getenv("TFT_SERVICE_URL", "http://localhost:8002/predict/tft")
LGBM_URL = os.getenv("LGBM_SERVICE_URL", "http://localhost:8003/predict/lgbm")

async def fetch_async(client, url, payload=None):
    if payload:
        resp = await client.post(url, json=payload, timeout=10.0)
    else:
        resp = await client.get(url, timeout=10.0)
    resp.raise_for_status()
    return resp.json()

@app.get("/predict/{ticker}")
async def ensemble_predict(ticker: str):
    try:
        sym = ticker.upper()
        async with httpx.AsyncClient() as client:
            # 1. Gọi lấy dữ liệu
            data_res = await fetch_async(client, DATA_URL.format(ticker))
            
            # 2. Bắn request song song cho 2 AI con (TFT và LGBM)
            payload = {"ticker": ticker, "features": data_res["features"]}
            
            tft_task = fetch_async(client, TFT_URL, payload)
            lgbm_task = fetch_async(client, LGBM_URL, payload)
            
            # Dừng 1 điểm để đợi cả 2 trả về (tiết kiệm thời gian rảnh rỗi)
            tft_res, lgbm_res = await asyncio.gather(tft_task, lgbm_task)
            
        tft_price = tft_res.get("predicted_t3")
        lgbm_price = lgbm_res.get("predicted_t3")

        if tft_price is None or lgbm_price is None:
            raise ValueError(f"Model errors: TFT={tft_res.get('error')}, LGBM={lgbm_res.get('error')}")

        # 3. Phân giải bằng Meta-Learner để tìm giá trị cân bằng nhất
        try:
            MODELS_DIR = download_model_artifacts(sym)
            meta_path = os.path.join(MODELS_DIR, f"{sym}_meta_learner.pkl")
        except FileNotFoundError:
            meta_path = None
        
        if meta_path is None or not os.path.exists(meta_path):
             meta_prediction = (tft_price + lgbm_price) / 2
        else:
            meta_learner = joblib.load(meta_path)
            meta_input = np.array([[tft_price, lgbm_price]])
            meta_prediction = meta_learner.predict(meta_input)[0]

        # 4. Áp dụng Kế toán Giao dịch (Decision Policy)
        features = data_res["features"]
        current_price = features["close"][-1]
        
        uncertainty = abs(tft_price - lgbm_price) / current_price * 100
        
        ctx = DecisionContext(
            current_price=float(current_price),
            predicted_price=float(meta_prediction),
            uncertainty_pct=float(uncertainty)
        )
        
        result = build_decision(ctx)
        
        # Trả về kết quả cuối
        return {
            "ticker": sym,
            "current_price": float(current_price),
            "predicted_t3": float(meta_prediction),
            "predicted_t3_tft": float(tft_price),
            "predicted_t3_lgbm": float(lgbm_price),
            "expected_return_pct": result.expected_return_pct,
            "decision": result.action,
            "metrics": {
                "confidence": result.confidence,
                "reason": result.reason,
                "uncertainty_pct": uncertainty
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
