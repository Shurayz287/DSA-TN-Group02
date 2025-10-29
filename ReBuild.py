# import libs
from pathlib import Path
from PIL import Image
import os
import cv2
import numpy as np
import torch
import faiss
from torchvision import transforms
from tqdm import tqdm
import timm
import hashlib

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_FOLDER = Path("D:/VSCODE/Python/DSA-TN-Group02/IMAGE")

# Workflow
def get_model():
    model = timm.create_model("resnet50", pretrained=True, num_classes = 0)
    model.to(DEVICE).eval() # move model to CPU/GPU, eval(): turn off dropout, batchnorm and use running stats
    return model

def get_transform():
    transform = transforms.Compose([
        # Normalize
        transforms.Resize((224,224)), # form of ResNet
        transforms.ToTensor(), # transform from PIL image (0-255) (H x W x C) ->torch.FloatTensor (0-1) (C x H x W)
        transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225]) # (x - mean) / std
    ])
    return transform

def read_image_cv2(file_path):
    img = cv2.imread(str(file_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img

# Filer absolutely duplicate
def compute_md5(file_path):
    """Tính MD5 hash cho một ảnh (bitwise)."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def remove_exact_duplicates(image_folder):
    image_paths = [f for f in image_folder.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    hashes = {}
    duplicates = []

    image_files = list(Path(image_folder).glob("*.*"))
    total = len(image_files)

    for img_path in image_files:
        h = compute_md5(img_path)
        if h in hashes:
            duplicates.append(img_path)
        else:
            hashes[h] = img_path

    # Xoá ảnh trùng
    for dup in duplicates:
        os.remove(dup)

    print(f"Total inital images: {len(image_paths)}")
    print(f"Number of duplicate images: {len(duplicates)}")
    return list(hashes.values())

def extract_embeddings_cv2(image_paths, batch_size = 32):
    # batch_size: number of image solving each time
    model = get_model()
    transforms = get_transform()
    embeddings = []
    valid_files = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc = "Extract embeddings"):
        batch_files = image_paths[i:i + batch_size]
        batch_imgs = []

        for f in batch_files:
            img = read_image_cv2(f)
            if img is None:
                continue
            img_tensor = transforms(img)
            batch_imgs.append(img_tensor)
            valid_files.append(f)

        #batch_imgs empty -> skip
        if not batch_imgs:
            continue

        batch_tensor = torch.stack(batch_imgs).to(DEVICE)
        with torch.no_grad():
            batch_emb = model(batch_tensor)
            batch_emb = batch_emb.cpu().numpy()
            embeddings.append(batch_emb)
    
    embeddings = np.vstack(embeddings)
    return embeddings, valid_files

# FAISS - FILTER
def faiss_group_duplicate(embeddings, filenames, distance_threshold = 0.5, k_neighbors = 5):
    # - embeddings: numpy array (n, d)
    # - filenames: list Path
    # - distance_threshold: cosine similarity threshold
    # - k_neighbors: số neighbors kiểm tra

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    D, neighbors = index.search(embeddings, k = k_neighbors)

    visited = set() # lưu các index ảnh đã được nhóm để không xử lý trùng.
    groups = []
    representatives = []

    for i, nbrs in enumerate(neighbors):
        if i in visited:
            continue
        group = [filenames[i]]
        visited.add(i)
        for j_idx, j in enumerate(nbrs[1:]): # skip itself
            if j in visited:
                continue
            sim = D[i, j_idx + 1]
            if sim >= 1 - distance_threshold:
                group.append(filenames[j])
                visited.add(j)
        # Choose representatives
        best_file = None
        best_score = -1
        for f in group:
            img = read_image_cv2(f)
            if img is None:
                continue
            w, h = img.size
            score = w * h
            if score > best_score:
                best_score = score
                best_file = f
        groups.append(group)
        representatives.append(best_file)
    return groups, representatives

# Main Workflow
def main(image_folder):

    unique_images = remove_exact_duplicates(image_folder)

    print(f"After remove absolute duplicates, remaining {len(unique_images)} valid images.\n")

    embeddings, valid_files = extract_embeddings_cv2(unique_images, batch_size=32)
    print(f"Number of valid images: {len(valid_files)}")

    groups, representatives = faiss_group_duplicate(embeddings, valid_files, distance_threshold=0.5, k_neighbors=5)
    print(f"Numbers of nearly duplicate groups: {len(groups)}")
    print(f"Number of remaining images: {len(representatives)}")

    for idx, g in enumerate(groups[:5]):
        rep = representatives[idx]
        print(f"\nGroup {idx + 1}: representative = {rep.name}, number of image = {len(g)}")

    output_folder = image_folder / "cleaned"
    output_folder.mkdir(exist_ok=True)
    for f in representatives:
        img = cv2.imread(str(f))
        cv2.imwrite(str(output_folder / f.name), img)

    print(f"\nCleaned folder: {output_folder}")
    return groups, representatives
# Running pipline
if __name__ == "__main__":
    groups, representatives = main(IMAGE_FOLDER)