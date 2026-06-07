# MLOps-Stock - Hệ thống Dự đoán Giá Cổ phiếu

Kho lưu trữ này chứa mã nguồn và luồng tích hợp liên tục (Continuous Integration - CI) cho hệ thống MLOps-Stock. Dự án tập trung vào việc áp dụng các mô hình học máy để phân tích và dự báo chuỗi thời gian dữ liệu chứng khoán.

## Kiến trúc Hệ thống & Công nghệ (Microservices)

Hệ thống được thiết kế theo kiến trúc vi dịch vụ (Microservices), bao gồm các thành phần cốt lõi sau:

- **Data API**: Đảm nhiệm việc thu thập, tiền xử lý và cung cấp dữ liệu chứng khoán theo thời gian thực.
- **Model APIs**:
  - **LightGBM API**: Phục vụ dự báo bằng mô hình Machine Learning truyền thống (LightGBM).
  - **TFT API**: Phục vụ dự báo bằng mô hình Deep Learning chuyên dụng cho chuỗi thời gian (Temporal Fusion Transformer).
- **Ensemble API**: Hoạt động như một Aggregation Layer, nhận kết quả từ các Model APIs và áp dụng logic kết hợp (Ensemble) để tối ưu hóa độ chính xác cuối cùng. Sử dụng Redis để caching.
- **Dashboard UI**: Giao diện người dùng trực quan để theo dõi các chỉ số và kết quả dự đoán.

## Luồng Tích hợp Liên tục (CI Pipeline)

Quy trình phát triển được tự động hóa hoàn toàn thông qua **GitHub Actions** với các Pipeline độc lập:
1. **Source Control & Trigger**: Kích hoạt khi có thay đổi trên mã nguồn, sử dụng cơ chế phát hiện thay đổi (path filtering) để chỉ build những dịch vụ bị ảnh hưởng (Monorepo strategy).
2. **Build & Package**: Đóng gói các dịch vụ thành các Docker Image bằng Docker Buildx.
3. **Security Scanning**: Tích hợp **Trivy** để quét các lỗ hổng bảo mật (CVEs) trên ảnh Docker trước khi phát hành.
4. **Image Signing**: Tích hợp **Cosign** để ký xác thực tính toàn vẹn của ảnh Docker.
5. **Registry Push**: Đẩy ảnh đã xác thực lên Docker Hub.
6. **Manifest Update**: Tự động cập nhật mã định danh ảnh (Image Hash) sang kho lưu trữ GitOps để kích hoạt quy trình triển khai (CD).

**Contributors:**
- Trần Việt Hoàng
- Lê Đình Hiếu
