import os

#cau hinh
TARGET_W, TARGET_H = 1200, 627
DEFAULT_INPUT_DIR = "./resources/input_images" 
OUTPUT_DIR = "./resources/output_images"  
EXTS = ("*.jpg","*.png", "*.JPG", "*.PNG")

os.makedirs(DEFAULT_INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)