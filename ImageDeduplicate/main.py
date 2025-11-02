import time
import os
import cv2
from pathlib import Path
from colorama import Fore

from imagededup import (
    IMAGE_FOLDER, METHODS_TO_RUN, MODELS_TO_RUN,
    remove_exact_duplicates,
    simhash_group_duplicate,
    minhash_group_duplicate,
    extract_embeddings_cv2,
    faiss_group_duplicate,
    calculate_metrics,
    print_summary,
    get_memory_usage
)

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
        groups, representatives = simhash_group_duplicate(unique_images, threshold=18) 
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
        metrics = calculate_metrics(groups, all_processed_files)
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