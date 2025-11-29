import cv2 as cv
import numpy as np
import math
from PIL import Image

DISPLAY_SIZE = 250

def _visualize_kernel(kernel):
    kernel_min = kernel.min()
    kernel_max = kernel.max()
    if kernel_max == kernel_min:
        kernel_norm = np.full(kernel.shape, 127, dtype=np.uint8) 
    else:
        kernel_norm = (kernel - kernel_min) / (kernel_max - kernel_min) * 255
        kernel_norm = kernel_norm.astype(np.uint8)
    
    img_pil = Image.fromarray(kernel_norm)
    img_pil = img_pil.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.NEAREST)
    img_cv = np.array(img_pil)
    
    img_cv = cv.cvtColor(img_cv, cv.COLOR_GRAY2BGR)
    
    h, w, _ = img_cv.shape
    grid_size = w // kernel.shape[0] 
    for i in range(1, kernel.shape[0]):
        cv.line(img_cv, (0, i * grid_size), (w, i * grid_size), (0, 0, 255), 1)
        cv.line(img_cv, (i * grid_size, 0), (i * grid_size, h), (0, 0, 255), 1)

    return img_cv

def get_morphology_elements(params):
    ksize = int(float(params['se_size'].get()))
    iterations = int(float(params['iterations'].get()))
    se_type_str = params['se_type'].get()
    
    ksize = ksize if ksize % 2 != 0 else ksize + 1 
    if ksize < 3: ksize = 3
    
    if 'Rect' in se_type_str:
        se_type = cv.MORPH_RECT
    elif 'Cross' in se_type_str:
        se_type = cv.MORPH_CROSS
    elif 'Ellipse' in se_type_str:
        se_type = cv.MORPH_ELLIPSE
    else:
        se_type = cv.MORPH_RECT

    kernel = cv.getStructuringElement(se_type, (ksize, ksize))
    return kernel, iterations, ksize

def execute_morphology(img_original_cv, alg, params):    
    thres_val = int(float(params['thres_morph'].get()))
    _, img_binary = cv.threshold(img_original_cv, thres_val, 255, cv.THRESH_BINARY)
    
    kernel, iterations, ksize = get_morphology_elements(params)
    
    result_img = None
    op_name = alg.split(': ')[1]
    
    if op_name == "Erosion":
        result_img = cv.erode(img_binary, kernel, iterations=iterations)
    elif op_name == "Dilation":
        result_img = cv.dilate(img_binary, kernel, iterations=iterations)
    elif op_name == "Opening":
        result_img = cv.morphologyEx(img_binary, cv.MORPH_OPEN, kernel, iterations=iterations)
    elif op_name == "Closing":
        result_img = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel, iterations=iterations)

    results = [
        (f'Binary Input (T={thres_val})', img_binary),
        (f'{op_name} (SE={ksize}, I={iterations})', result_img),
    ]
    
    if op_name == "Opening":
        img_eroded = cv.erode(img_binary, kernel, iterations=iterations)
        results.insert(1, ('Step 1: Erosion', img_eroded))
    elif op_name == "Closing":
        img_dilated = cv.dilate(img_binary, kernel, iterations=iterations)
        results.insert(1, ('Step 1: Dilation', img_dilated))
    return results

def execute_homework(img_original_cv, params):    
    # Lấy tham số ngưỡng và kích thước SE cho Boundary Extraction
    thres_val = int(float(params['thres_hw'].get()))
    se_size_boundary = int(float(params['se_size_hw'].get()))
    # 1. Nhị phân hóa ảnh đầu vào (A)
    _, img_A = cv.threshold(img_original_cv, thres_val, 255, cv.THRESH_BINARY)
    results = [('Binary Input (A)', img_A)]
    # --- HW4-1: Erosion & Dilation với Kernel 3x3 Tùy chỉnh ---
    kernel_hw4_1 = np.array([[1, 0, 0], 
                             [0, 1, 0], 
                             [0, 0, 1]], dtype=np.uint8)
    # Erosion: A ⊖ B
    img_erode_custom = cv.erode(img_A, kernel_hw4_1, iterations=1)
    # Dilation: A ⊕ B
    img_dilate_custom = cv.dilate(img_A, kernel_hw4_1, iterations=1)
    results.append(('HW4-1: Erosion (Custom SE)', img_erode_custom))
    results.append(('HW4-1: Dilation (Custom SE)', img_dilate_custom))
    # --- HW4-2: Boundary Extraction (Trích Biên) ---
    kernel_boundary = cv.getStructuringElement(cv.MORPH_RECT, (se_size_boundary, se_size_boundary))
    # 1. Thực hiện Erosion: A ⊖ B
    img_A_eroded_B = cv.erode(img_A, kernel_boundary, iterations=1)
    # 2. Thực hiện Phép Hiệu (Set Difference): A - (A ⊖ B)
    img_boundary = cv.subtract(img_A, img_A_eroded_B)
    results.append((f'HW4-2: A eroded B (SE={se_size_boundary}x{se_size_boundary})', img_A_eroded_B))
    results.append((f'HW4-2: Boundary (A - A o B)', img_boundary))

    return results