import cv2
from albumentations import (
    Compose, RandomBrightnessContrast, ShiftScaleRotate, Blur, GaussNoise
)
from albumentations.core.composition import OneOf
import numpy as np
import os

# --- Cấu hình ---
input_folder = "OriginalSrc"        # 📂 Thư mục chứa ảnh gốc (ví dụ: 'data/original')
output_folder = "DesSrc"   # 📂 Thư mục chứa ảnh sau khi augment
num_aug_per_image = 20               # Số ảnh augment cho mỗi ảnh gốc

# --- Khởi tạo augmentations ---
augment = Compose([
    ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=5, p=0.6),
    RandomBrightnessContrast(p=0.5),
    OneOf([
        Blur(blur_limit=3),
        GaussNoise(var_limit=(10, 50))
    ], p=0.3)
])

# --- Tạo thư mục output ---
os.makedirs(output_folder, exist_ok=True)

# --- Lấy danh sách ảnh ---
valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_ext)]

if not image_files:
    raise ValueError(f"Không tìm thấy ảnh hợp lệ trong thư mục: {input_folder}")

# --- Xử lý từng ảnh ---
for filename in image_files:
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️  Bỏ qua: {filename} (không đọc được)")
        continue

    base_name, ext = os.path.splitext(filename)
    for i in range(num_aug_per_image):
        aug = augment(image=img)['image']
        save_name = f"{base_name}_{i}{ext}"
        save_path = os.path.join(output_folder, save_name)
        cv2.imwrite(save_path, aug)

    print(f"✅ {filename}: đã tạo {num_aug_per_image} ảnh mới.")

print(f"\n🎉 Hoàn tất! Tất cả ảnh đã lưu trong thư mục '{output_folder}'.")
