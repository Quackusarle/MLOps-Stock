# MLOps Stock Hybrid Ensemble (Microservices Edition)

Hệ thống dự báo giá chứng khoán VN30 (T+3) sử dụng kiến trúc Hybrid Ensemble (TFT + LightGBM) được triển khai theo mô hình Microservices.

## 🚀 Cấu trúc Dự án
- **`src/`**: Logic lõi (Data Pipeline, Model Architecture, Training logic).
- **`services/`**: Các trạm dịch vụ API (Data, TFT, LGBM, Ensemble Gateway, Dashboard UI).
- **`models/`**: Lưu trữ trọng số mô hình đã huấn luyện.

## 📦 Yêu cầu hệ thống
- Python 3.9+
- Docker & Docker Compose (Khuyên dùng)

## 🛠 Khởi động nhanh (Quick Start)

### 1. Huấn luyện mô hình (Nếu cần)
```powershell
python src/training/final_ensemble_train.py
```

### 2. Chạy hệ thống với Docker
```powershell
docker-compose up --build
```
Hệ thống sẽ chạy tại: [http://localhost:8081](http://localhost:8081)

## 📡 Danh sách Microservices
| Dịch vụ | Port | Chức năng |
| :--- | :--- | :--- |
| Data API | 8001 | Tải và xử lý dữ liệu Yahoo Finance |
| TFT API | 8002 | Suy luận mô hình Deep Learning TFT |
| LGBM API | 8003 | Suy luận mô hình Machine Learning LightGBM |
| Ensemble Gateway | 8080 | Gateway điều phối & Meta-learner |
| Dashboard UI | 8081 | Giao diện Web hiển thị kết quả |

## 🧪 Kiểm tra API
Bạn có thể kiểm tra trực tiếp qua curl:
```powershell
curl http://localhost:8080/predict/FPT
```

---
Dự án được thiết kế để dễ dàng mở rộng (Scalability) và sẵn sàng cho các quy trình MLOps tự động.
