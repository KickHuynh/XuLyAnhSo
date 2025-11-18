import os

#cau hinh
TARGET_W, TARGET_H = 1200, 627
DEFAULT_INPUT_DIR = "./resources/input_images" # Sửa đường dẫn cho cấu trúc mới
OUTPUT_DIR = "./resources/output_images"  # Sửa đường dẫn cho cấu trúc mới
EXTS = ("*.jpg","*.png", "*.JPG", "*.PNG")

os.makedirs(DEFAULT_INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)