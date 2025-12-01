
# 📦 Cài đặt Môi trường và Thư viện cho Image Duplicate Detection Pipeline

## 📋 Yêu cầu hệ thống

### Hệ điều hành hỗ trợ
- Windows 10/11 (64-bit)
- macOS 10.14+
- Linux (Ubuntu 18.04+, CentOS 7+)

### Yêu cầu phần cứng
- RAM: Tối thiểu 8GB (khuyến nghị 16GB+)
- GPU: Không bắt buộc, nhưng khuyến nghị NVIDIA GPU với CUDA cho tốc độ xử lý nhanh hơn
- Dung lượng ổ cứng: Tối thiểu 2GB trống

### Phiên bản Python
- Python 3.8 - 3.11 (khuyến nghị Python 3.9)

## 🛠️ Cài đặt chi tiết theo từng hệ điều hành

### 1. **Cài đặt Python**

#### Windows
```bash
# Tải Python từ python.org
# Chọn "Add Python to PATH" khi cài đặt
# Kiểm tra cài đặt:
python --version
pip --version
```

#### macOS
```bash
# Cài qua Homebrew
brew install python@3.9

# Hoặc tải từ python.org
# Kiểm tra cài đặt:
python3 --version
pip3 --version
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.9 python3.9-venv python3-pip
python3.9 --version
pip3 --version
```

### 2. **Tạo và kích hoạt môi trường ảo (.venv)**

#### Windows
```powershell
# Tạo thư mục project
mkdir image-deduplication
cd image-deduplication

# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo
.venv\Scripts\activate

# Kiểm tra đã kích hoạt thành công (sẽ thấy (.venv) ở đầu dòng lệnh)
(.venv) PS C:\path\to\project>
```

#### macOS/Linux
```bash
# Tạo thư mục project
mkdir image-deduplication
cd image-deduplication

# Tạo môi trường ảo
python3 -m venv .venv

# Kích hoạt môi trường ảo
source .venv/bin/activate

# Kiểm tra đã kích hoạt thành công (sẽ thấy (.venv) ở đầu dòng lệnh)
(.venv) user@computer:~/image-deduplication$
```

### 3. **Cài đặt các thư viện cần thiết**

Tạo file `requirements.txt` với nội dung sau:

```txt
torch>=2.0.0
torchvision>=0.15.0
faiss-cpu>=1.7.4  # hoặc faiss-gpu nếu có CUDA
timm>=0.9.0
opencv-python>=4.8.0
numpy>=1.24.0
tqdm>=4.65.0
colorama>=0.4.6
psutil>=5.9.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
Pillow>=10.0.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

#### Cài đặt tất cả thư viện cùng lúc:

```bash
# Đảm bảo đã kích hoạt môi trường ảo
pip install -r requirements.txt
```

#### Hoặc cài đặt từng thư viện (nếu không có file requirements.txt):

```bash
# Đảm bảo đã kích hoạt môi trường ảo

# Cài đặt PyTorch (chọn phiên bản phù hợp với hệ thống)

# CPU version (cho mọi hệ thống)
pip install torch torchvision

# GPU version với CUDA 11.8 (cho NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# GPU version với CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Các thư viện khác
pip install faiss-cpu  # hoặc faiss-gpu nếu có CUDA
pip install timm opencv-python numpy tqdm colorama psutil matplotlib scikit-learn Pillow jupyter ipykernel
```

### 4. **Cài đặt đặc biệt cho từng hệ điều hành**

#### Windows - Cài đặt Visual C++ Redistributable (cần thiết cho một số package)
- Tải và cài đặt từ [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

#### macOS - Cài đặt thêm qua Homebrew (nếu cần)
```bash
# Cài đặt libomp cho FAISS trên macOS
brew install libomp
```

#### Linux - Cài đặt dependencies hệ thống
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev build-essential libopenblas-dev

# CentOS/RHEL/Fedora
sudo yum install python3-devel gcc-c++ openblas-devel
```

### 5. **Cấu hình Jupyter Kernel**

```bash
# Thêm kernel từ môi trường ảo vào Jupyter
python -m ipykernel install --user --name=.venv --display-name="Python (Image Deduplication)"

# Khởi động Jupyter Notebook
jupyter notebook
```

## 🔧 Kiểm tra cài đặt

Tạo file `test_installation.py` để kiểm tra:

```python
import sys
import torch
import cv2
import faiss
import timm
import numpy as np
from PIL import Image

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("OpenCV version:", cv2.__version__)
print("FAISS version:", faiss.__version__)
print("TIMM version:", timm.__version__)

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

print("✅ Tất cả thư viện đã được cài đặt thành công!")
```

Chạy kiểm tra:
```bash
python test_installation.py
```

## 🚀 Chạy notebook

1. **Tải file `SETUP.ipynb`** vào thư mục project
2. **Tạo thư mục chứa ảnh**:
   ```bash
   mkdir img
   # Thêm các file ảnh cần xử lý vào thư mục img/
   ```
3. **Khởi động Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
4. **Mở file `SETUP.ipynb`** trong Jupyter
5. **Chạy từng cell** theo thứ tự từ trên xuống

## ⚠️ Xử lý lỗi thường gặp

### Lỗi "No module named 'faiss'"
```bash
# Thử cài đặt phiên bản build sẵn
pip install faiss-cpu --no-cache-dir

# Hoặc build từ source
pip install faiss-cpu -v
```

### Lỗi liên quan đến CUDA
```bash
# Kiểm tra phiên bản CUDA tương thích
nvidia-smi

# Cài đặt phiên bản PyTorch phù hợp với CUDA
# Xem tại: https://pytorch.org/get-started/locally/
```

### Lỗi memory trên GPU
```bash
# Giảm batch_size trong code
batch_size = 16  # thay vì 32
```

## 📝 Ghi chú quan trọng

1. **Kích hoạt môi trường ảo** mỗi khi làm việc với project
2. **Kiểm tra phiên bản Python** trước khi cài đặt
3. **Lưu file requirements.txt** để dễ dàng khôi phục môi trường
4. Đối với **dataset lớn**, cần đảm bảo đủ RAM và dung lượng ổ cứng
5. **Sử dụng GPU** sẽ tăng tốc độ xử lý đáng kể

## 🔄 Khôi phục môi trường

```bash
# Lưu môi trường hiện tại
pip freeze > requirements.txt

# Khôi phục môi trường trên máy khác
pip install -r requirements.txt
```

## 📚 Tài liệu tham khảo

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [FAISS Installation](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
- [TIMM Documentation](https://github.com/huggingface/pytorch-image-models)
- [OpenCV Installation](https://docs.opencv.org/4.x/d5/de5/tutorial_py_setup_in_windows.html)

---
*Lưu ý: Quá trình cài đặt có thể mất 5-15 phút tùy vào tốc độ mạng và cấu hình hệ thống.*
