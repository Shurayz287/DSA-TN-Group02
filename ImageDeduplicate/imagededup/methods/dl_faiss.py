# imagededup/methods/dl_faiss.py

import torch
import timm
import cv2
import faiss
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Các import bắt buộc để code chạy được
from ..utils.config import DEVICE
from ..utils.image_helpers import read_image_cv2

# Workflow
def get_model(model_name="resnet50"):
    model = timm.create_model(model_name, pretrained=True, num_classes = 0)
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

def extract_embeddings_cv2(image_paths, model_name="resnet50", batch_size = 32):
    # batch_size: number of image solving each time
    model = get_model(model_name)
    transforms = get_transform()
    embeddings = []
    valid_files = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc = f"Extract embeddings ({model_name})"):
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
    
    if not embeddings:
        return np.array([]), []
        
    embeddings = np.vstack(embeddings)
    return embeddings, valid_files

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