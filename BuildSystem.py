import numpy as np
from pathlib import Path 
from PIL import Image
from collections import defaultdict
import torch 
from torchvision import transforms
import timm
from simhash import Simhash
import faiss
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_FOLDER = Path("images")
HASH_SIZE = 64
K_NEAREST = 3 # 3 nearest by FAISS

# Feature extracting

def get_model():
    model = timm.create_model("resnet18", pretrained=True, num_classes = 0)
    model.to(DEVICE).eval()
    return model

def get_transform():
    transform = transforms.Compose([
        transforms.Resize((224,224)), # form of ResNet
        transforms.ToTensor(), # transform from PIL image(0-255, H x W x C) ->torch.FloatTensor (0-1, C x H x W)
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # (x - mean) / std
    ])
    return transform

def feature_extract(model, transform, img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE) # [B, C, H, W]
    with torch.no_grad():
        feat = model(x)
    return feat.cpu().numpy().flatten()

# Hashing (SimHash)
def simhash_vector(vector, hash_size = HASH_SIZE):
    vector_str = " ".join([str(v) for v in vector])
    return Simhash(vector_str, f = hash_size).value # value of binary sequence

# FAISS
def build_faiss_index(features):
    d = features.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(features)
    return index

# Grouping
def group_by_faiss(features, image_paths, threshold=50):
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    groups = [] 
    visited = set()
    for i, feat in enumerate(features):
        if i in visited:
            continue
        D, I = index.search(feat.reshape(1, -1), len(features))
        group = []
        for dist, idx in zip(D[0], I[0]):
            if dist <= threshold:
                group.append(image_paths[idx])
                visited.add(idx)
        groups.append(group)
    return groups


# Pick the representative
def select_representatives(groups):
    def get_size(img_path):
        img = Image.open(img_path)
        return img.width * img.height
    representatives = []
    for group in groups:  # group is a list of image paths
        best_path = max(group, key=get_size)
        representatives.append(best_path)
    return representatives

# Visualize

def visualize_groups(groups, max_cols=3):
    import math
    num_groups = len(groups)
    print(f"Tổng số nhóm: {num_groups}")

    cols = min(max_cols, num_groups)
    rows = math.ceil(num_groups / cols)

    fig, axes = plt.subplots(
        nrows=rows, ncols=cols,
        figsize=(cols * 4, rows * 4)
    )

    # Chuẩn hóa axes thành list
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, items in zip(axes, groups):
        rep_img = Image.open(items[0])
        ax.imshow(rep_img)
        ax.axis("off")
        ax.set_title(f"Nhóm {groups.index(items)+1} - {len(items)} ảnh", fontsize=10)

    # Ẩn các axes dư thừa
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
        print("Không tìm thấy ảnh trong folder images/")
        return
    print(f"Tổng số ảnh: {len(image_paths)}")

    # 2. Load model & transform
    model = get_model()
    transform = get_transform()

    # 3. Extracting feature
    print("Đang trích xuất đặc trưng...")
    features = np.array([feature_extract(model, transform, p) for p in image_paths], dtype="float32")
    print("Shape features:", features.shape)

    # 4. Hash by SimHash
    print("Đang tính hash...")
    hashes = [simhash_vector(f) for f in features]

    # 5. find k_nearest bằng FAISS (exp for the first img)
    index = build_faiss_index(features)
    query = features[0].reshape(1, -1)
    D, I = index.search(query, K_NEAREST)
    print("FAISS - ảnh gần nhất cho ảnh đầu tiên:")
    for rank, idx in enumerate(I[0]):
        print(f"{rank+1}: {image_paths[idx].name} (distance={D[0][rank]:.4f})")

    # 6. grouping
    groups = group_by_faiss(features, image_paths)
    print(f"Tổng số nhóm trùng lặp: {len(groups)}")

    # 7. Choose representative
    representatives = select_representatives(groups)
    print(f"Tổng số ảnh đại diện: {len(representatives)}")
    print("Ảnh đại diện:", [p.name for p in representatives])

    # 8. Visualize by matplotlib
    print("Đang trực quan hóa các nhóm ảnh...")
    visualize_groups(groups)

if __name__ == "__main__":
    main()