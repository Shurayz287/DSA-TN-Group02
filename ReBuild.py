# Import libs
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
import Evaluate
from colorama import Fore, Style, init
from datasketch import MinHash, MinHashLSH
from functools import lru_cache
import psutil, time
import Evaluate


# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_FOLDER = Path("DesSrc")

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

def read_image_cv2(file_path):
    img = cv2.imread(str(file_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img

# MD5 - FILTER
def compute_md5(file_path):
    """Calculate MD5 hash for a image (bitwise)."""
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


# Minhash - FILTER
# --- improved feature extraction: include block position + more levels
def extract_features(image_path, size=(64,64), block_size=8, levels=16):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return set()
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    h, w = size
    features = []
    for bi in range(0, h, block_size):
        for bj in range(0, w, block_size):
            block = img[bi:bi+block_size, bj:bj+block_size]
            if block.size == 0:
                continue
            avg = int(block.mean())
            level = int(avg * (levels - 1) / 255)  # 0 .. levels-1
            features.append(f"b_{bi}_{bj}_{level}")
    return set(features)

# cache resolution checks
@lru_cache(maxsize=None)
def get_resolution(path):
    try:
        img = cv2.imread(str(path))
        if img is None:
            return 0
        h, w = img.shape[:2]
        return w * h
    except Exception:
        return 0

def minhash_group_duplicate(image_paths, threshold=0.9, num_perm=128,
                            block_size=8, levels=16, size=(64,64)):
    # Create LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}

    for path in tqdm(image_paths, desc="MinHashing"):
        features = extract_features(path, size=size, block_size=block_size, levels=levels)
        if not features:
            # skip images with no features
            continue
        m = MinHash(num_perm=num_perm)
        for f in features:
            m.update(f.encode('utf8'))
        minhashes[path] = m
        lsh.insert(str(path), m)

    # Cluster via BFS (connected components in LSH graph)
    groups, representatives = [], []
    visited = set()
    for path in list(minhashes.keys()):
        if path in visited:
            continue
        queue = [path]
        group = []
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            group.append(current)
            similar = lsh.query(minhashes[current])
            for s in similar:
                s_path = Path(s)
                if s_path not in visited and s_path in minhashes:
                    queue.append(s_path)

        # choose highest resolution representative
        rep = max(group, key=get_resolution)
        groups.append(group)
        representatives.append(rep)

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

# Config output
init(autoreset=True)

def print_summary(groups, representatives, all_processed_files, image_folder, metrics=None, total_initial=None, after_md5=None, 
                  elapsed_time=None, mem_used=None):
    print("\n" + "="*70)
    print(Fore.CYAN + Style.BRIGHT + "📊 DUPLICATE DETECTION SUMMARY".center(70))
    print("="*70)

    if total_initial is not None:
        print(f"{Fore.YELLOW}Total Initial Images: {Fore.WHITE}{total_initial}")
    if after_md5 is not None:
        removed = total_initial - after_md5 if total_initial is not None else "?"
        print(f"{Fore.YELLOW}Remaining After MD5 Filter: {Fore.WHITE}{after_md5} "
              f"({removed} removed)")

    print(f"{Fore.YELLOW}Total Clusters Found: {Fore.WHITE}{len(groups)}")
    print(f"{Fore.YELLOW}Total Representatives: {Fore.WHITE}{len(representatives)}")
    print(f"{Fore.YELLOW}Total Processed Images: {Fore.WHITE}{len(all_processed_files)}")
    print(f"{Fore.YELLOW}Cleaned Folder: {Fore.WHITE}{Path(image_folder) / 'cleaned'}")

    print("\n" + "="*70)
    print(Fore.CYAN + Style.BRIGHT + "📊 PERFORMANCE EVALUATION".center(70))
    print("="*70)

    if elapsed_time is not None:
        print(f"{Fore.GREEN}Processing Time: {Fore.WHITE}{elapsed_time:.2f} seconds")
    if mem_used is not None:
        print(f"{Fore.GREEN}Memory Used: {Fore.WHITE}{mem_used:.2f} MB")

    print("\n" + "="*70)
    print(Fore.CYAN + Style.BRIGHT + "📊 SOME CLUSTERS".center(70))
    print("="*70)
    for idx, group in enumerate(groups):
        if idx >= 5:
            break
        rep = representatives[idx]
        rep_name = rep.name if hasattr(rep, "name") else Path(rep).name
        print(f"{Fore.CYAN}Cluster {idx + 1:>3} {Fore.WHITE}| {len(group):>3} images | "
              f"Representative: {Fore.GREEN}{rep_name}")
    print("-"*70)

    if metrics:
        print("\n" + "="*70)
        print(Fore.CYAN + Style.BRIGHT + "📊 EVALUATION METRICS".center(70))
        print("="*70)
        print(f"{Fore.CYAN}Precision: {Fore.WHITE}{metrics.get('precision', 0)*100:.2f}%")
        print(f"{Fore.CYAN}Recall:    {Fore.WHITE}{metrics.get('recall', 0)*100:.2f}%")
        print(f"{Fore.CYAN}F1-Score:  {Fore.WHITE}{metrics.get('f1_score', 0)*100:.2f}%")
    print("="*70 + "\n")


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 ** 2)

# Main pipline
def main(image_folder, method = "faiss", model_name="resnet50"):

    image_paths = [f for f in image_folder.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    total_initial = len(image_paths)

    unique_images = remove_exact_duplicates(image_folder)
    after_md5 = len(unique_images)

    #print(f"After remove absolute duplicates, remaining {len(unique_images)} valid images.\n")

    groups = []
    representatives = []
    all_processed_files= [] 

    # Timing
    start_time = time.time()
    mem_before = get_memory_usage()

    if method == "faiss":
        embeddings, valid_files = extract_embeddings_cv2(unique_images, batch_size=32)
        groups, representatives = faiss_group_duplicate(embeddings, valid_files, distance_threshold=0.6, k_neighbors=10)
        all_processed_files = valid_files

    elif method == "simhash":
        # Binary signature
        groups, representatives = simhash_group_duplicate(unique_images, threshold=16) 
        all_processed_files = unique_images
    
    elif method == "minhash":
        # MinHash + LSH
        groups, representatives = minhash_group_duplicate(unique_images, threshold=0.6)
        all_processed_files = unique_images

    else:
        raise ValueError(f"Unknown method: {method}. Choose from ['faiss', 'simhash', 'minhash'].")
    
    # Result of timing

    elapsed_time = time.time() - start_time
    mem_after = get_memory_usage()
    mem_used = mem_after - mem_before

    output_folder = image_folder / "Cleaned"
    output_folder.mkdir(exist_ok=True)
    for f in representatives:
        img = cv2.imread(str(f))
        cv2.imwrite(str(output_folder / f.name), img)

    print(f"\nCleaned folder: {output_folder}")

    # Evaluation
    metrics = None
    if all_processed_files:
        metrics = Evaluate.calculate_metrics(groups, all_processed_files)
    else:
        print(Fore.RED + "\n⚠️ Cannot evaluate because no images were processed.")

    print_summary(groups, representatives, all_processed_files, image_folder, metrics, total_initial=total_initial, after_md5=after_md5,
                  elapsed_time=elapsed_time, mem_used=mem_used)
    return groups, representatives

# Running pipline
if __name__ == "__main__":
    methods = ["faiss", "simhash", "minhash"]
    models = ["resnet50", "efficientnet_b0", "mobilenetv3_large_100", "vit_base_patch16_224"]
    groups, representatives = main(IMAGE_FOLDER, method=methods[0], model_name=models[0])