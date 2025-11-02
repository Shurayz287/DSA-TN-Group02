import cv2
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
from pathlib import Path
from ..utils.image_helpers import get_resolution 

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