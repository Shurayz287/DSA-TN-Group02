import cv2
import numpy as np
from simhash import Simhash
from tqdm import tqdm

# Simhash - FILTER
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