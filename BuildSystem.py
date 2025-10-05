import numpy as np
import math
from pathlib import Path 
from PIL import Image
from collections import defaultdict
import torch 
from torchvision import transforms
import timm
from simhash import Simhash
import bitarray 
import random
import mmh3
import faiss
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ================================
# 1. HashTable
# ================================

class HashTableIndex:
    # Collision sol: Linear probing
    def __init__(self, size = 1000):
        self.size = size
        self.table = [None] * size # create 1000 buckets, (key, [list_of_values])

    def _hash(self,key):
        return hash(key) % self.size
    
    def add(self, key, value):
        idx = self._hash(key)
        while self.table[idx] is not None:
            stored_key, store_values = self.table[idx]
            # check existence of key
            if stored_key == key:
                store_values.append(value)
                return
            
            # different key -> collision, linear probing
            idx = (idx + 1) % self.size

        self.table[idx] = (key, [value])

    def query(self, key):
        # Take all values corresponding to key
        idx = self._hash(key)
        start = idx
        while self.table[idx] is not None:
            stored_key, stored_values = self.table[idx]
            if stored_key == key:
                return stored_values
            idx = (idx + 1) % self.size
            if idx == start:  # break if full table
                break
        return None

# ================================
# 2. Bloom Filter
# ================================

class BloomFilterIndex:
    def __init__(self, size=1000, hash_count=3):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray.bitarray(size)
        self.bit_array.setall(0)  # initialize all bits to 0

    def _item_to_str(self, item):
        # convert item (vector, number, string) to string for hashing
        if isinstance(item, np.ndarray):
            return " ".join(map(str, item))
        else:
            return str(item)

    def add(self, item):
        item_str = self._item_to_str(item)
        for i in range(self.hash_count):
            idx = mmh3.hash(item_str, i) % self.size
            self.bit_array[idx] = 1

    def query(self, item):
        item_str = self._item_to_str(item)
        for i in range(self.hash_count):
            idx = mmh3.hash(item_str, i) % self.size
            if self.bit_array[idx] == 0:
                return False
        return True


# ================================
# 3. SimHash
# ================================

class SimhashIndex:
    def __init__(self, hash_size=64):
        self.hash_size = hash_size
        self.hashes = {}  # id -> hash
    
    def _simhash(self, features):
        bits = [0] * self.hash_size
        for i, val in enumerate(features):
            h = mmh3.hash64(f"{i}:{round(val,4)}")[0] & ((1 << self.hash_size) - 1)
            for b in range(self.hash_size):
                if (h >> b) & 1:
                    bits[b] += val
                else:
                    bits[b] -= val
        fingerprint = 0
        for b in range(self.hash_size):
            if bits[b] > 0:
                fingerprint |= 1 << b
        return fingerprint
    
    def add(self, features, item_id):
        h = self._simhash(features)
        self.hashes[item_id] = h

    def query(self, features, threshold=8):
        q_hash = self._simhash(features)
        results = []
        for item_id, h in self.hashes.items():
            dist = bin(q_hash ^ h).count("1")
            if dist <= threshold:
                results.append((item_id, dist))
        return results
    
# ================================
# 4. MinHash
# ================================   
class MinhashIndex:
    def __init__(self, num_hashes=100, max_val=2**32-1, top_k=50):
        self.num_hashes = num_hashes
        self.max_val = max_val
        self.top_k = top_k 
        # create many hash functions, especially pair (a,b)
        self.hash_funcs = [
            (random.randint(1, max_val), random.randint(0, max_val))
            for _ in range(num_hashes)
        ]
        self.signatures = {}  # Dict: id -> signature
    
    # take top k index max -> convert to int instead of float to work
    def vector_to_shingles(self, vec):
        topk_idx = np.argsort(vec)[-self.top_k:]  
        return set(topk_idx)

    def _signature(self, shingles):
        sig = []
        for a, b in self.hash_funcs:
            min_hash = min(((a * x + b) % self.max_val) for x in shingles)
            sig.append(min_hash)
        return sig

    def add(self, vec, item_id):
        shingles = self.vector_to_shingles(vec)
        sig = self._signature(shingles)
        self.signatures[item_id] = sig

    def query(self, vec, threshold=0.4):
        shingles = self.vector_to_shingles(vec)
        q_sig = self._signature(shingles)
        results = []
        for item_id, sig in self.signatures.items():
            matches = sum(1 for i in range(self.num_hashes) if sig[i] == q_sig[i]) # number of same match
            score = matches / self.num_hashes
            if score >= threshold:
                results.append((item_id, score))
        return results



# Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_FOLDER = Path("D:/VSCODE/Python/DSA-TN-Group02/images")
HASH_SIZE = 64

# Feature extracting

def get_model():
    model = timm.create_model("resnet18", pretrained=True, num_classes = 0)
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

def feature_extract(model, transform, img_path):
    img = Image.open(img_path).convert("RGB") # Guarantee the img has 3 channel
    x = transform(img).unsqueeze(0).to(DEVICE) # [B, C, H, W], add batch dimension
    with torch.no_grad(): # turn off gradient -> save time and memory(inference)
        feat = model(x) # return feature vector (512-d for ResNet18)
    return feat.cpu().numpy().flatten() # bring tensor to CPU, transform to numpy and flatten to 1D vector -> use for hash/FAISS


# Grouping
def group_by_hash_faiss(features, image_paths, hash_index, hash_type,
                        threshold_hash=12, threshold_faiss_sq=1.2):
    n = len(features)
    groups = []
    visited = set()

    # create mapping name -> index to avoid ValueError
    img_to_idx = {p: i for i, p in enumerate(image_paths)}

    for i in range(n):
        if i in visited:
            continue
        group = [image_paths[i]]
        visited.add(i)
        fi = features[i]
        img_id_i = image_paths[i]

        candidate_ids = []
        # Lấy candidate từ hash
        if hash_type == "simhash":
            results = hash_index.query(fi, threshold=31)
            candidate_ids = [r[0] for r in results]
        elif hash_type == "minhash":
            results = hash_index.query(fi, threshold=0.3)
            candidate_ids = [r[0] for r in results]
        elif hash_type == "hashtable":
            vals = hash_index.query(img_id_i)
            candidate_ids = [v for v in vals] if vals else []
        elif hash_type == "bloom":
            if hash_index.query(img_id_i):
                candidate_ids = [p.name for p in image_paths]  # all images are candidates
            else:
                candidate_ids = []

        # Use FAISS (or squared L2) to refine
        if candidate_ids:
            cand_idx = [img_to_idx[j] for j in candidate_ids if img_to_idx[j] not in visited]

            if cand_idx:  # when meet unvisited candidates
                cand_feats = np.array([features[j] for j in cand_idx], dtype="float32")

                # create FAISS index
                index_bucket = faiss.IndexFlatL2(cand_feats.shape[1])
                index_bucket.add(cand_feats)

                # find in bucket
                D, I = index_bucket.search(fi.reshape(1, -1).astype("float32"), len(cand_idx))

                for d, idx_local in zip(D[0], I[0]):
                    if d <= threshold_faiss_sq:
                        real_idx = cand_idx[idx_local]
                        group.append(image_paths[real_idx])
                        visited.add(real_idx)
        groups.append(group)
    return groups


# Pick the representative
def select_representatives(groups):
    # Sharpness
    def sharpness_score(img_path):
        import cv2
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        return cv2.Laplacian(img, cv2.CV_64F).var()
    
    representatives = []
    for group in groups:  # group is a list of image paths
        best_path = max(group, key=sharpness_score)
        representatives.append(best_path)
    return representatives

# Visualize

def visualize_groups(groups, fig_width=8, fig_height=6, max_cols=4):
    """
    Visualize image groups in a fixed figure size.
    
    Args:
        groups: list of list of Path objects
        fig_width: width of figure in inches
        fig_height: height of figure in inches
        max_cols: max number of columns per row
    """
    num_groups = len(groups)
    print(f"Total groups: {num_groups}")

    # Fix number of cols, calculate the number of rows
    cols = min(max_cols, num_groups)
    rows = math.ceil(num_groups / cols)

    # Create fix figure
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(fig_width, fig_height))
    
    # Normalized axes to list
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Create each group
    for ax, items in zip(axes, groups):
        rep_img = Image.open(items[0])
        ax.imshow(rep_img)
        ax.axis("off")
        ax.set_title(f"Group {groups.index(items)+1}: {len(items)} ảnh", fontsize=10)

    # Hidden sub axes 
    for ax in axes[len(groups):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# Main pipeline

def main():
    # 1. Load img
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_paths.extend(IMAGE_FOLDER.glob(ext))

    if not image_paths:
        print("Not found in folder images/")
        return
    print(f"Number of images: {len(image_paths)}")

    # 2. Load model & transform
    model = get_model()
    transform = get_transform()

    # 3. Extracting feature
    print("Extracting...")
    features = np.array([feature_extract(model, transform, p) for p in image_paths], dtype="float32")
    print("Shape features:", features.shape)

    # Normalize features
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    # 4. Hashing
    hash_type = "minhash"  # "hashtable", "bloom", "simhash", "minhash"

    if hash_type not in ["hashtable", "bloom", "simhash", "minhash"]:
        raise ValueError(f"hash_type {hash_type} không hợp lệ")

    if hash_type == "hashtable":
        index = HashTableIndex()
    elif hash_type == "bloom":
        index = BloomFilterIndex(size=5000, hash_count=5)
    elif hash_type == "simhash":
        index = SimhashIndex(hash_size=HASH_SIZE)
    elif hash_type == "minhash":
        index = MinhashIndex(num_hashes=100)


    for idx, feat in enumerate(features):
        img_id = image_paths[idx]  
        
        if hash_type in ["simhash", "minhash"]:
            index.add(feat, img_id)
        elif hash_type == "hashtable":
            key = tuple(feat)  
            index.add(key, img_id)
        elif hash_type == "bloom":
            key = tuple(feat) 
            # Bloom Filter only, need item
            index.add(feat)


    # 5. grouping
    #groups = group_by_faiss(features, image_paths)
    groups = group_by_hash_faiss(features, image_paths, index, hash_type)
    print(f"Total duplicate groups: {len(groups)}")

    # 6. Choose representative
    representatives = select_representatives(groups)
    print(f"Total representatives: {len(representatives)}")
    print("Representative:", [p.name for p in representatives])

    # 8. Visualize by matplotlib
    print("Visualizing...")
    visualize_groups(groups, max_cols=len(groups))

if __name__ == "__main__":
    main()