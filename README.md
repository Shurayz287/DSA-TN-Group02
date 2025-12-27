# DSA-TN-Group02
Repo chính về học phần mở rộng của nhóm 2, chương trình Tài năng ngành Khoa học Máy tính khóa K24  

## 📋 Giới thiệu dự án
Dự án này tập trung vào việc xây dựng một hệ thống tự động phát hiện và loại bỏ ảnh trùng lặp sử dụng các kỹ thuật xử lý ảnh và học máy. Công cụ giúp tối ưu hóa không gian lưu trữ, tăng tốc độ xử lý dữ liệu, và cải thiện chất lượng bộ dữ liệu ảnh cho các ứng dụng máy tính thị giác.

## 🧠 Phương pháp tiếp cận
Hệ thống sử dụng kết hợp đa tầng để phát hiện ảnh trùng lặp:
1. **MD5 Hashing**: Xác định và loại bỏ các ảnh trùng lặp hoàn toàn (bitwise identical)
2. **FAISS với Deep Features**: Phát hiện ảnh trùng lặp ngữ nghĩa sử dụng mô hình ResNet50 và tìm kiếm láng giềng gần

## 🚀 Các tính năng chính
- **Xử lý trực tiếp từ Google Drive**: Đọc và ghi dữ liệu trực tiếp từ Google Drive
- **Pipeline tự động hóa**: Từ nhận diện đến xóa bỏ ảnh trùng lặp
- **Đánh giá hiệu suất**: Tính toán Precision, Recall, và F1-Score
- **Trực quan hóa kết quả**: Hiển thị các cụm ảnh trùng lặp và ảnh đại diện
- **Tối ưu hóa bộ nhớ**: Theo dõi và quản lý sử dụng tài nguyên

## 🛠️ Công nghệ sử dụng
- **Python 3.8+** với Jupyter Notebook
- **PyTorch & timm**: Trích xuất đặc trưng ảnh
- **FAISS**: Tìm kiếm láng giềng gần với hiệu suất cao
- **OpenCV & PIL**: Xử lý và đọc ảnh
- **Google Colab**: Môi trường thực thi

## 📂 Cấu trúc dự án
```
DSA-TN-Group02/
├── research             # Folder chứa document
    ├── Image_deduplication.pdf                # doc
    ├── code.ipynb       # code
├── tool             # Folder chứa tool
    ├── img                                    # Folder chứa ảnh
    ├── image-deduplication-detection.ipynb    # Code
    ├── image-deduplication-detection.md       # Hướng dẫn
├── README.md                                  # Giới thiệu
└── index.html                                 # Pages
```

## 📖 Hướng dẫn sử dụng
1. **Chuẩn bị dữ liệu**: Đặt ảnh vào thư mục cá nhân trên Google Drive
2. **Chạy trên Google Colab**: 
   - Mở notebook `Image_deduplication_detection.ipynb`
   - Mount Google Drive và chỉ định đường dẫn thư mục ảnh
   - Chạy tất cả các cell để xử lý
3. **Kết quả**: 
   - Folder `Cleaned_faiss` chứa ảnh đại diện duy nhất
   - Báo cáo hiệu suất và trực quan hóa kết quả

## 📊 Đánh giá hiệu suất
Hệ thống cung cấp các chỉ số đánh giá:
- **Precision**: Độ chính xác trong việc phát hiện ảnh trùng lặp
- **Recall**: Khả năng tìm thấy tất cả ảnh trùng lặp
- **F1-Score**: Cân bằng giữa Precision và Recall
- **Thời gian xử lý & Sử dụng bộ nhớ**

## 🌐 Demo trực tuyến
Truy cập [https://shurayz287.github.io/DSA-TN-Group02/](https://shurayz287.github.io/DSA-TN-Group02/) để xem:
- Giới thiệu chi tiết về dự án
- Demo trực quan về quy trình xử lý
- Kết quả thử nghiệm và đánh giá
- Hướng dẫn sử dụng chi tiết

## 👥 Thành viên nhóm
- **Nguyễn Hoàng Gia Huy** 
- **Vũ Hoàng Hải** 
- **Nguyễn Ngọc Thạch** 

## 📝 Kết quả đạt được
- Xử lý thành công bộ dữ liệu ảnh với độ chính xác cao (>85%)
- Giảm đáng kể kích thước bộ dữ liệu bằng cách loại bỏ ảnh trùng lặp
- Cung cấp giao diện trực quan cho người dùng cuối

## 🔮 Hướng phát triển tương lai
- Tích hợp thêm các mô hình deep learning hiện đại
- Phát triển giao diện web độc lập
- Hỗ trợ xử lý video và ảnh độ phân giải cao
- Tối ưu hóa tốc độ xử lý cho bộ dữ liệu lớn

## 📄 Giấy phép
Dự án được phát triển cho mục đích học thuật và nghiên cứu.

---

*Dự án được phát triển bởi Nhóm 2 - Chương trình Tài năng Khoa học Máy tính K24*
