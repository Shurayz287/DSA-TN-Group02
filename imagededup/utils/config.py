from pathlib import Path
import torch

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_FOLDER = Path("dataset/random_fakes") # <-- Sửa lại đường dẫn tuyệt đối của bạn ở đây

# Biến để chạy trong main.py
METHODS_TO_RUN = ["faiss", "simhash", "minhash"]
MODELS_TO_RUN = ["resnet50", "efficientnet_b0", "mobilenetv3_large_100", "vit_base_patch16_224"]