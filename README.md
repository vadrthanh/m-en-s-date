# Phân Tích Cân Bằng Đa Môi Trường S-DATE-SDN

## Tổng Quan
Dự án này triển khai các mô hình học máy tiên tiến để phát hiện xâm nhập mạng trên nhiều môi trường khác nhau (UNSW, IoT, SDN). Nó bao gồm các chức năng toàn diện để tải dữ liệu, lựa chọn đặc trưng, giảm chiều dữ liệu, cân bằng dữ liệu, và huấn luyện mô hình với tăng tốc GPU.

## Tính Năng
- Phân tích dữ liệu đa môi trường (UNSW, IoT, SDN)
- Tính toán và lựa chọn đặc trưng quan trọng
- Giảm chiều dữ liệu sử dụng PCA và t-SNE
- Cân bằng dữ liệu với kỹ thuật tạo dữ liệu tổng hợp
- Nhiều mô hình học máy:
  - ML truyền thống: Random Forest, AdaBoost, ExtraTrees
  - Học sâu: LSTM, CNN, GRU
- Tăng tốc GPU cho cả mô hình học máy truyền thống và học sâu
- Tối ưu hóa ensemble sử dụng thuật toán meta-heuristic (PSO, GA, MFO, DE)
- Nhiều kỹ thuật cân bằng dữ liệu (SMOTE, ADASYN, CTGAN)

## Yêu Cầu Hệ Thống
Dự án yêu cầu các thư viện sau:
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- PyTorch (cho học sâu và tăng tốc GPU)
- imblearn (cho SMOTE và ADASYN)
- mealpy (cho các thuật toán tối ưu hóa)
- sdv (cho việc tạo dữ liệu tổng hợp CTGAN)

Bạn có thể cài đặt các thư viện cần thiết bằng:
```
pip install -r requirements.txt
```

## Cấu Trúc Dữ Liệu
Dự án sử dụng các bộ dữ liệu sau:
- Bộ dữ liệu UNSW: `UNSW_NB15_testing-set.csv`, `UNSW_NB15_training-set.csv`
- Bộ dữ liệu IoT: `IoT Network Intrusion Dataset.csv`
- Bộ dữ liệu SDN: `Normal_data.csv`, `OVS.csv`

## Dataset
Bạn có thể tải dữ liệu từ [link này](https://husteduvn-my.sharepoint.com/:f:/g/personal/thanh_nxc235623_sis_hust_edu_vn/EhUEEyE-g_ZLp5Ev928M3B8B3NQJIFBoN70tBG4Lt5qoAA?e=pnbVq1).

## Cách Sử Dụng
Để chạy script chính:
```
python m-en-s-date-gpu.py
```

Script sẽ thực hiện:
1. Kiểm tra khả năng sử dụng GPU
2. Tải và chuẩn bị dữ liệu từ tất cả các nguồn
3. Lựa chọn các đặc trưng quan trọng
4. Tạo các biểu đồ trực quan hóa
5. Cân bằng các bộ dữ liệu
6. Huấn luyện và đánh giá nhiều mô hình học máy
7. Lưu kết quả vào thư mục đầu ra

## Kết Quả Đầu Ra
Kết quả được lưu trong các thư mục có nhãn thời gian trong thư mục `output/`, bao gồm:
- Biểu đồ độ quan trọng của đặc trưng
- Trực quan hóa t-SNE
- Các chỉ số hiệu suất mô hình
- Đồ thị huấn luyện/kiểm định
- Ma trận nhầm lẫn

## Tăng Tốc GPU
Mã tự động phát hiện và sử dụng tăng tốc GPU khi có sẵn, với PyTorch là nền tảng hỗ trợ. Hỗ trợ GPU được triển khai cho cả mô hình học sâu và các thuật toán học máy truyền thống.

## Thành Phần Chính
- `setup_gpu()`: Cấu hình PyTorch để sử dụng GPU nếu có
- `load_data()`: Tải dữ liệu từ nhiều nguồn khác nhau
- `select_features()`: Chọn các đặc trưng quan trọng bằng ExtraTrees
- `data_balancing()`: Cân bằng dữ liệu sử dụng giảm chiều và tạo dữ liệu tổng hợp
- `train_evaluate_model()`: Huấn luyện và đánh giá mô hình với tùy chọn tăng tốc GPU

## Các Mô Hình
- ML Truyền thống: RandomForest, AdaBoost, ExtraTrees
- Học Sâu:
  - `LSTMModel`: Mạng dựa trên LSTM cho phân loại chuỗi
  - `CNNModel`: Mạng dựa trên CNN cho phân loại
  - `GRUModel`: Mạng dựa trên GRU cho phân loại chuỗi

## Tối Ưu Hóa
Dự án bao gồm các kỹ thuật tối ưu hóa cho mô hình ensemble sử dụng:
- PSO: Tối ưu hóa Bầy Đàn
- GA: Thuật toán Di truyền
- MFO: Tối ưu hóa Ngọn lửa Bướm đêm
- DE: Tiến hóa Vi phân