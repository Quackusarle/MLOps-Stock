# MLOps Stock Hybrid Ensemble (Simplified Release)

**Hệ thống Dự báo Giá Chứng khoán dựa trên Mô hình Hybrid Ensemble (TFT + LightGBM)**

Dự án này là phiên bản tinh gọn (Simplified Edition) của hệ thống MLOps Stock Ensemble, tập trung thuần túy vào thuật toán **AI Cốt lõi** và **Giao diện Web / API**, lược bỏ hoàn toàn sự phức tạp của hạ tầng MLOps (MLflow) và Bảo mật (JWT) để dễ dàng triển khai, học tập và báo cáo.

---

## 🚀 Tính năng nổi bật
- **Hybrid Ensemble Model**: Kết hợp **Temporal Fusion Transformer (TFT)** chuyên dự tính xu hướng dài hạn và **LightGBM** chuyên phản ứng sự cố ngắn hạn để cho ra dự đoán chính xác nhất.
- **Mục tiêu T+3 Thực tế**: Dự đoán giá chứng khoán tại mốc **T+3**, phù hợp hoàn toàn với quy định chu kỳ thanh toán T+2.5 của thị trường chứng khoán Việt Nam.
- **Tự động hóa Feature Engineering**: Tải dữ liệu tự động hằng ngày từ Yahoo Finance và tính toán 8 chỉ báo kỹ thuật (RSI, MACD, Bollinger Bands...).
- **Kế toán Giao dịch (Decision Policy)**: Thuật toán không chỉ đoán giá, mà tính cả thuế phí, độ trượt giá, và rủi ro mâu thuẫn giữa 2 AI để chốt lệnh **MUA / BÁN / GIỮ**.
- **Web Dashboard**: Giao diện người dùng Glassmorphism cực kỳ hiện đại để theo dõi kết quả.
- **REST API**: API trả về dữ liệu chuẩn JSON siêu tốc bằng FastAPI.

## 📦 Cấu trúc Thư mục Hệ thống
Hệ thống gồm 9 module Python chính và thư mục giao diện:

- `models/`: Chứa các "não bộ" đã huấn luyện (File weights, pickle, scalers).
- `tft_model.py`: Mạng Deep Learning PyTorch (VSN, LSTM, Attention).
- `lgbm_model.py`: Mạng Cây quyết định Gradient Boosting.
- `ensemble_trainer.py`: Kịch bản huấn luyện trộn (Stacking Meta-Learner).
- `final_ensemble_train.py`: Khởi động huấn luyện hàng loạt cho 4 mã (VNM, VCB, HPG, FPT).
- `yahoo_data.py`: Giao tiếp tải dữ liệu từ Yahoo Finance.
- `indicators.py`: Tính toán các chỉ báo Phân tích kỹ thuật.
- `decision_policy.py`: Mạch ra quyết định MUA/BÁN Kế toán.
- `app.py`: Máy chủ phân phối REST API.
- `dashboard.py`: Máy chủ Server-Side Rendering cho Giao diện Trình duyệt.
- `templates/`: File `index.html` của trang Web.

## 🛠 Hướng dẫn Cài đặt & Chạy

### 1. Cài đặt môi trường
Bắt buộc cần Python 3.9+ 
```bash
pip install -r requirements.txt
```

### 2. Huấn luyện Mô hình (Tùy chọn)
Hệ thống đã có sẵn mô hình trong thư mục `./models/`. Nhưng nếu bạn muốn tự train lại với dữ liệu mới nhất:
```bash
python final_ensemble_train.py
```
*(Quá trình này tốn khoảng 10-20 phút cho 4 mã chứng khoán).*

### 3. Khởi động Web Dashboard (Giao diện đồ họa)
Mở terminal và chạy lệnh:
```bash
python dashboard.py
```
👉 Truy cập trình duyệt: `http://localhost:8081`

### 4. Khởi động Máy chủ API (Cho lập trình viên)
Mở một cửa sổ terminal khác và chạy lệnh:
```bash
python app.py
```
👉 Truy cập API: `http://localhost:8080/predict/FPT`
(Hệ thống sẽ trả về dữ liệu dạng JSON)

---
**Tác giả**: Lê Đình Hiếu, Trần Việt Hoàng (ATTT2023.1)  
**Giáo viên Hướng dẫn**: ThS. Lê Anh Tuấn
