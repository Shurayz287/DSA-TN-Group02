import cv2
from albumentations import (
    Compose, RandomBrightnessContrast, ShiftScaleRotate, Blur, GaussNoise
)
from albumentations.core.composition import OneOf
import numpy as np
import os
from pathlib import Path

# Rename files
def rename_images_in_folder(folder_path, prefix="fl"):
    folder = Path(folder_path)
    image_files = sorted(
        [f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]],
        key=lambda x: x.name
    )

    print(f"Found {len(image_files)} image files.")
    for idx, file in enumerate(image_files, start=1):
        new_name = f"{prefix}{idx}{file.suffix.lower()}"
        new_path = folder / new_name

        # tránh trùng tên (nếu file đã tồn tại)
        if new_path.exists():
            print(f"Skipping {new_name} (already exists)")
            continue

        file.rename(new_path)
        print(f"Renamed: {file.name} → {new_name}")

    print("✅ Done renaming!")

rename_images_in_folder("D:\VSCODE\Python\DSA-TN-Group02\OriginalSrc")

# --- Cấu hình ---
input_folder = "OriginalSrc"        # 📂 Thư mục chứa ảnh gốc (ví dụ: 'data/original')
output_folder = "DesSrc"   # 📂 Thư mục chứa ảnh sau khi augment
num_aug_per_image = 10               # Số ảnh augment cho mỗi ảnh gốc

# --- Khởi tạo augmentations ---
augment = Compose([
    ShiftScaleRotate(shift_limit=0.01, scale_limit=0.02, rotate_limit=2, p=1.0),
    RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.1),
    # OneOf([
    #     Blur(blur_limit=1),
    #     GaussNoise(var_limit=(0, 1))
    # ], p=0.2)
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
