import cv2
from PIL import Image
from functools import lru_cache

def read_image_cv2(file_path):
    img = cv2.imread(str(file_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img

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