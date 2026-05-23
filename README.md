# Khóa luận: Impact of Macroeconomic Shocks on the NASDAQ-100 Index

Đề tài phân tích chuỗi thời gian chuyên sâu (Time Series Analysis) nhằm đánh giá tác động của các cú sốc vĩ mô (Lạm phát - CPI, và Lãi suất mục tiêu của FED - FFR) lên tỷ suất sinh lợi của chỉ số công nghệ cốt lõi NASDAQ-100.

##  Mục tiêu Nghiên cứu (Objectives)
- **Đánh giá cấu trúc**: Sử dụng mô hình **SVAR** (Structural Vector Autoregression) để xác định sự truyền dẫn của các cú sốc vĩnh viễn từ lạm phát và chính sách tiền tệ đến thị trường chứng khoán (thông qua Phân rã Cholesky: *CPI → FFR → NDX*).
- **Kiểm định dự báo tính ngoại sinh**: Thiết lập một mô hình điểm chuẩn nội sinh **SARIMA** (Box-Jenkins) và so sánh hiệu suất dự báo Out-of-Sample với mô hình có yếu tố vĩ mô **SARIMAX**. Qua đó chứng minh liệu vĩ mô có đem lại giá trị thặng dư so với Thuyết Thị trường Hiệu quả (EMH).

## 🗂 Cấu trúc Dự án (Project Structure)

Dự án được tổ chức theo chuẩn Modular Data Science, tách biệt quy trình tính toán (`src/`) và quy trình trình bày/thực thi (`notebooks/`). Các folder kết quả đều tự động dồn về một mối `outputs/`.

```text
nasdaq-macro-shocks-analysis/
├── data/
│   ├── raw/                 # Dữ liệu gốc tải từ cổng thông tin FRED
│   └── processed/           # Dữ liệu đã làm sạch và chia tách (train, val, test)
│       └── splits/          # Dữ liệu phục vụ huấn luyện và đánh giá ngoài mẫu
├── docs/                    # Các tài liệu tham khảo chuyên ngành
├── notebooks/               # Pipeline phân tích
│   ├── 00_data_understanding.ipynb
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_cleaning_and_preparation.ipynb
│   ├── 03_feature_engineering_and_transformation.ipynb
│   ├── 04_svar_modeling_and_irf.ipynb
│   ├── 05_sarima_box_jenkins.ipynb
│   └── 06_sarimax_integration_and_evaluation.ipynb
├── outputs/                 # Kết xuất tự động (tất cả các script đều trỏ ra đây)
│   ├── figures/             # ACF/PACF, IRF, FEVD, Diagnostics, v.v.
│   └── tables/              # Tổng hợp tệp dữ liệu trung gian và xuất chuẩn Latex
├── src/                     # Source code tự định nghĩa 
│   ├── data_preprocessing.py   # Xử lý thời gian, resample dữ liệu
│   ├── evaluation_metrics.py   # Hệ đo lường sai số (RMSE, MAE, MAPE)
│   ├── feature_engineering.py  # Xử lý log return, sai phân, tạo lags
│   ├── fetch_data.py           # Gọi giao thức API
│   ├── model_utils.py          # Viết báo cáo cho các mô hình
│   ├── sarimax.py              # Auto-GridSearch và Chuẩn đoán Box-Jenkins
│   ├── statistical_tests.py    # Kiểm định trạm (ADF, KPSS) và độ tự tương quan
│   └── svar.py                 # Ước lượng, chuẩn đoán căn của ma trận và phân rã IRF
├── README.md                # Tài liệu hướng dẫn hiện tại
└── requirements.txt         # Khai báo môi trường Python
```

##  Phương pháp luận (Methodology)

1. **Tiền xử lý & EDA**: Xử lý dữ liệu khuyết thiếu ngày lễ, đồng bộ thời gian về cơ sở dữ liệu hàng tháng (Month End). Kiểm định tính dừng, tính chuẩn bằng ADF, KPSS, Ljung-Box.
2. **SVAR & Cú sốc cấu trúc**: 
   - Kiểm định ổn định của hệ thống VAR (Eigenvalue Modulus $>$ 1).
   - Hàm phản ứng xung kích trực giao (Orthogonalized IRF), và Hàm phân rã phương sai sai số độ trễ (FEVD).
3. **Box-Jenkins SARIMAX**:
   - Nhận diện ACF/PACF.
   - Ước lượng AIC bằng Grid Search tự động. Khuyến nghị và chốt Benchmark `SARIMA(0,0,1)`.
   - Kết luận chuẩn đoán White Noise Residuals.
4. **Out-of-Sample Forecasting**: So sánh cấu trúc $MA(1)$ đơn giản (nền tảng của Random Walk/Thị trường hiệu quả) với mô hình $MA(1) + X_t$ chứa biến Vĩ mô trong giai đoạn khó khăn (2020 - COVID_19 & Thắt chặt Chính sách Tiền tệ 2022+).

##  Hướng dẫn Cài đặt & Chạy (Quick Start)

Đây là tài liệu sử dụng Python >= 3.9.

1. **Khởi tạo và kích hoạt môi trường ảo (Virtual Env)**:
   Môi trường khuyên nghị là `.venv`.

2. **Cài đặt thư viện phụ thuộc**:
   ```bash
   pip install -r requirements.txt
   pip install pmdarima  # Cho thành phần SARIMA Grid Search
   ```

3. **Thực thi phân tích**:
   Hãy làm mới (restart kernel) và chạy (Run All) toàn bộ quy trình Notebooks từ số định danh `00` cho tiến đến `06`. Mọi bảng biểu và đồ họa sẽ tự động ghi đè và biên dịch tại thư viện `/outputs/figures` và `/outputs/tables`.

