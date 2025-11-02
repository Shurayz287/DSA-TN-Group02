# imagededup/__init__.py

# Export các hàm từ utils
from .utils.config import DEVICE, IMAGE_FOLDER, METHODS_TO_RUN, MODELS_TO_RUN
from .utils.system_helpers import print_summary, get_memory_usage
from .utils.image_helpers import read_image_cv2, get_resolution

# Export các hàm từ evaluation
from .evaluation.evaluate import calculate_metrics

# Export các hàm từ methods
from .methods.hashing_md5 import remove_exact_duplicates
from .methods.hashing_simhash import simhash_group_duplicate
from .methods.hashing_minhash import minhash_group_duplicate
from .methods.dl_faiss import extract_embeddings_cv2, faiss_group_duplicate