import cv2
from albumentations import (
    Compose, RandomBrightnessContrast, ShiftScaleRotate, Blur, GaussNoise
)
from albumentations.core.composition import OneOf
import numpy as np
import os
from pathlib import Path

# --- Rename images sequentially ---
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

        # Avoid overwriting if the target file already exists
        if new_path.exists():
            print(f"Skipping {new_name} (already exists)")
            continue

        file.rename(new_path)
        print(f"Renamed: {file.name} → {new_name}")

    print("✅ Done renaming all images!")

# Run renaming
rename_images_in_folder("D:\\VSCODE\\Python\\DSA-TN-Group02\\OriginalSrc")

# --- Configuration ---
input_folder = "OriginalSrc"         # 📂 Folder containing original images
output_folder = "DesSrc"             # 📂 Folder to save augmented images
num_aug_per_image = 10               # Number of augmented images per original

# --- Initialize augmentation pipeline ---
augment = Compose([
    ShiftScaleRotate(shift_limit=0.01, scale_limit=0.02, rotate_limit=2, p=1.0),
    RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.1),
])

# --- Create output directory if not exists ---
os.makedirs(output_folder, exist_ok=True)

# --- Get list of valid images ---
valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_ext)]

if not image_files:
    raise ValueError(f"No valid images found in folder: {input_folder}")

# --- Process each image ---
for filename in image_files:
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Skipping: {filename} (could not read file)")
        continue

    base_name, ext = os.path.splitext(filename)
    for i in range(num_aug_per_image):
        aug = augment(image=img)['image']
        save_name = f"{base_name}_{i}{ext}"
        save_path = os.path.join(output_folder, save_name)
        cv2.imwrite(save_path, aug)

    print(f"✅ {filename}: generated {num_aug_per_image} new augmented images.")

print(f"\n🎉 All done! Augmented images are saved in '{output_folder}'.")
