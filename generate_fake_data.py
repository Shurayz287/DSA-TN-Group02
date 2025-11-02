import requests
import os
from PIL import Image, ImageFilter
import io 
import random
from tqdm import tqdm 

API_KEY = "sRiKJFO70QGjz_mKSUZee2YtwmX34Y4ZpXrgOGAnHYI" 

SEARCH_QUERY = "flowers" 

TOTAL_ORIGINAL_IMAGES = 50

MIN_FAKES_PER_IMAGE = 5   
MAX_FAKES_PER_IMAGE = 10   

OUTPUT_DIR = "dataset/random_fakes"

def random_resize(img):
    width, height = img.size
    new_ratio = random.uniform(0.8, 0.95)
    new_w = int(width * new_ratio)
    new_h = int(height * new_ratio)
    return img.resize((new_w, new_h), Image.LANCZOS)

def random_rotate(img):
    angle = random.uniform(-15, 15)
    return img.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor='black')

def random_crop(img):
    width, height = img.size
    crop_ratio_w = random.uniform(0.8, 0.9)
    crop_ratio_h = random.uniform(0.8, 0.9)
    crop_w = int(width * crop_ratio_w)
    crop_h = int(height * crop_ratio_h)
    x1 = random.randint(0, width - crop_w)
    y1 = random.randint(0, height - crop_h)
    box = (x1, y1, x1 + crop_w, y1 + crop_h)
    return img.crop(box)

def create_random_fake(original_img):
    operations = [random_resize, random_rotate, random_crop]
    num_to_apply = random.randint(1, 1) 
    random.shuffle(operations)
    fakes_to_apply = operations[:num_to_apply]
    
    img_copy = original_img.copy()
    for op in fakes_to_apply:
        img_copy = op(img_copy)
        
    return img_copy.resize(original_img.size, Image.LANCZOS)


def download_and_augment_images():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Bắt đầu tải {TOTAL_ORIGINAL_IMAGES} ảnh gốc, từ khóa '{SEARCH_QUERY}'...")
    
    print(f"Sẽ tạo từ {MIN_FAKES_PER_IMAGE} đến {MAX_FAKES_PER_IMAGE} fakes/ảnh gốc.")
    print(f"Sẽ lưu vào thư mục: '{OUTPUT_DIR}'")

    PER_PAGE = 30
    total_pages = (TOTAL_ORIGINAL_IMAGES // PER_PAGE) + 1
    count = 0 
    total_fakes_created = 0 

    for page_num in range(1, total_pages + 1):
        if count >= TOTAL_ORIGINAL_IMAGES:
            break

        api_url = "https://api.unsplash.com/search/photos"
        params = {"query": SEARCH_QUERY, "page": page_num, "per_page": PER_PAGE, "client_id": API_KEY}

        try:
            response = requests.get(api_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            for item in data.get("results", []):
                if count >= TOTAL_ORIGINAL_IMAGES:
                    break
                
                count += 1
                image_id = item["id"]
                image_url = item["urls"]["regular"]
                base_name = f"fl{count}"
                
                print(f"\n--- Đang xử lý nhóm {count}/{TOTAL_ORIGINAL_IMAGES} (ID: {image_id}) ---")

                try:
                    image_data = requests.get(image_url, timeout=10).content
                    original_img = Image.open(io.BytesIO(image_data))
                    original_img = original_img.convert("RGB") 

                    original_file_name = f"{base_name}_0.jpg"
                    original_path = os.path.join(OUTPUT_DIR, original_file_name)
                    original_img.save(original_path, "JPEG", quality=95)
                    print(f"Đã lưu: {original_file_name} (ảnh gốc)")

                    num_fakes_for_this_image = random.randint(MIN_FAKES_PER_IMAGE, MAX_FAKES_PER_IMAGE)
                    print(f"Đang tạo {num_fakes_for_this_image} ảnh fake ngẫu nhiên...")

                    for i in range(1, num_fakes_for_this_image + 1):
                        try:
                            fake_img = create_random_fake(original_img)
                            fake_file_name = f"{base_name}_{i}.jpg"
                            fake_path = os.path.join(OUTPUT_DIR, fake_file_name)
                            fake_img.convert('RGB').save(fake_path, "JPEG", quality=95)
                            total_fakes_created += 1
                            
                        except Exception as e_aug:
                            print(f"    LỖI khi tạo biến thể fake_{i}: {e_aug}")

                except Exception as e_img:
                    print(f"LỖI khi tải hoặc lưu ảnh gốc {image_id}: {e_img}")

        except requests.exceptions.RequestException as e_api:
            print(f"LỖI API: {e_api}")
            break
            
    print("\n" + "="*50)
    print("--- HOÀN TẤT ---")
    print(f"Đã tải và xử lý: {count} ảnh gốc.")
    print(f"Đã tạo: {total_fakes_created} ảnh biến thể.")
    print(f"Tổng số file trong thư mục '{OUTPUT_DIR}': {count + total_fakes_created}")
    print("="*50)

if __name__ == "__main__":
    if "YOUR_ACCESS_KEY" in API_KEY:
        print("LỖI: Bạn chưa thay thế 'YOUR_ACCESS_KEY' ở đầu file!")
    else:
        download_and_augment_images()