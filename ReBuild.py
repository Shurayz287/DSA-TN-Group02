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
from simhash import Simhash

import evaluate

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_FOLDER = Path("DesSrc")

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

    # delete duplicate
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

# Simhash - FILTER

# Simhash utilities
def compute_simhash(file_path):
    try:
        img = cv2.imread(str(file_path))
        if img is None:
            return None
        img = cv2.resize(img, (64, 64))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = []
        for i in range(0, 64, 8):
            for j in range(0, 64, 8):
                block = gray[i:i+8, j:j+8]
                avg = int(block.mean())
                features.append(f"block_{i}_{j}_{avg//16}")
        return Simhash(features)
    except Exception as e:
        print(f"[Error] {file_path}: {e}")
        return None

def hamming_distance(hash1, hash2):
    return hash1.distance(hash2)

def simhash_group_duplicate(image_paths, threshold = 22):
    hashes = []
    for f in tqdm(image_paths, desc="Compute SimHash"):
        h = compute_simhash(f)
        if h:
            hashes.append((f,h))
    
    visited = set()
    groups = []
    representatives = []

    for i in range(len(hashes)):
        if i in visited:
            continue
        f1, h1 = hashes[i]
        group = [f1]
        visited.add(i)

        for j in range(i + 1, len(hashes)):
            if j in visited:
                continue
            f2, h2 = hashes[j]
            dist = hamming_distance(h1, h2)
            if(dist <= threshold):
                group.append(f2)
                visited.add(j)

        # Choose representative
        best_file = None
        best_score = -1
        for f in group:
            img = cv2.imread(str(f))
            if img is None:
                continue
            h, w, _ = img.shape
            score = h * w
            if score > best_score:
                best_score = score
                best_file = f

        groups.append(group)
        representatives.append(best_file)

    return groups, representatives


# FAISS - FILTER
def faiss_group_duplicate(embeddings, filenames, distance_threshold = 0.6, k_neighbors = 10):
    # - embeddings: numpy array (n, d)
    # - filenames: list Path
    # - distance_threshold: cosine similarity threshold
    # - k_neighbors: neighbors checking

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    D, neighbors = index.search(embeddings, k = k_neighbors)

    visited = set() 
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
            if sim >= distance_threshold:
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

# Main Pipline
def main(image_folder, method = "faiss"):

    unique_images = remove_exact_duplicates(image_folder)

    print(f"After remove absolute duplicates, remaining {len(unique_images)} valid images.\n")

    groups = []
    representatives = []
    all_processed_files= [] 

    if method == "faiss":
        embeddings, valid_files = extract_embeddings_cv2(unique_images, batch_size=32)
        print(f"Number of valid images: {len(valid_files)}")
        groups, representatives = faiss_group_duplicate(embeddings, valid_files, distance_threshold=0.6, k_neighbors=10)
        all_processed_files = valid_files
    elif method == "simhash":
        groups, representatives = simhash_group_duplicate(unique_images, threshold=16) 
        all_processed_files = unique_images


    print(f"Numbers of nearly duplicate groups: {len(groups)}")
    print(f"Number of remaining images: {len(representatives)}")

    for idx, g in enumerate(groups[:5]):
        rep = representatives[idx]
        print(f"\nCluster {idx + 1}: representative = {rep.name}, number of image = {len(g)}")

    output_folder = image_folder / "cleaned"
    output_folder.mkdir(exist_ok=True)
    for f in representatives:
        img = cv2.imread(str(f))
        cv2.imwrite(str(output_folder / f.name), img)

    print(f"\nCleaned folder: {output_folder}")

    # Evaluation
    if all_processed_files:
        metrics = evaluate.calculate_metrics(groups, all_processed_files)
        print("\n--- Evaluation ---")
        print(f"Precision: {metrics['precision']:.2f}")
        print(f"Recall:    {metrics['recall']:.2f}")
        print(f"F1-Score:  {metrics['f1_score']:.2f}")
    else:
        print("\nKhông thể đánh giá vì không có file nào được xử lý.")

    return groups, representatives
# Running pipline
if __name__ == "__main__":
    groups, representatives = main(IMAGE_FOLDER, method ="simhash")