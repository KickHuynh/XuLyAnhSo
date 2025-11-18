import numpy as np
import cv2
import time # <<< THÊM VÀO

# Frequency 
def create_D_matrix(rows, cols):
    center_row, center_col = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt((U - center_row)**2 + (V - center_col)**2)
    return D

# === HÀM ĐÃ ĐƯỢC CẬP NHẬT ĐỂ TRẢ VỀ TIMINGS ===
def apply_frequency_filter(img_gray, H_filter_func, D0, n=None):
    rows, cols = img_gray.shape
    
    if D0 == 0: D0 = 1e-6 
        
    timings = {} # Dictionary để lưu thời gian
    
    # === CÔNG ĐOẠN 1: Forward DFT ===
    start_time = time.perf_counter()
    dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    timings['1_Forward_DFT_ms'] = (time.perf_counter() - start_time) * 1000
    
    # === CÔNG ĐOẠN 2: Shift (dịch tâm) ===
    start_time = time.perf_counter()
    dft_shift = np.fft.fftshift(dft)
    timings['2_FFT_Shift_ms'] = (time.perf_counter() - start_time) * 1000
    
    # Tạo bộ lọc H
    if n is None:
        H = H_filter_func(rows, cols, D0)
    else:
        H = H_filter_func(rows, cols, D0, n)
    
    H_complex = np.zeros_like(dft_shift)
    H_complex[:,:,0] = H
    H_complex[:,:,1] = H
    
    # === CÔNG ĐOẠN 3: Nhân với bộ lọc (H * F) ===
    start_time = time.perf_counter()
    G_shift = dft_shift * H_complex
    timings['3_Multiply_Filter_H_ms'] = (time.perf_counter() - start_time) * 1000
    
    # === CÔNG ĐOẠN 4: Inverse Shift (dịch về) ===
    start_time = time.perf_counter()
    G_ishift = np.fft.ifftshift(G_shift)
    timings['4_IFFT_Shift_ms'] = (time.perf_counter() - start_time) * 1000
    
    # === CÔNG ĐOẠN 5: Inverse DFT (về miền không gian) ===
    start_time = time.perf_counter()
    img_back = cv2.idft(G_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    img_out = np.uint8(img_back)
    timings['5_Inverse_DFT_ms'] = (time.perf_counter() - start_time) * 1000
    
    # Trả về cả ảnh VÀ dictionary thời gian
    return img_out, timings

# (Các hàm IHPF, ILPF, BLPF... giữ nguyên)
def IHPF(rows, cols, D0):
    D = create_D_matrix(rows, cols)
    H = np.ones((rows, cols))
    H[D <= D0] = 0
    return H

def ILPF(rows, cols, D0):
    D = create_D_matrix(rows, cols)
    H = np.zeros((rows, cols))
    H[D <= D0] = 1 
    return H

def BLPF(rows, cols, D0, n=2):
    D = create_D_matrix(rows, cols)
    H = 1 / (1 + (D / D0)**(2 * n))
    return H

def BHPF(rows, cols, D0, n=2):
    D = create_D_matrix(rows, cols)
    D[D == 0] = 1e-6 # Tránh chia cho 0
    H = 1 / (1 + (D0 / D)**(2 * n))
    return H

def GLPF(rows, cols, D0):
    D = create_D_matrix(rows, cols)
    H = np.exp(-(D**2) / (2 * (D0**2)))
    return H

def GHPF(rows, cols, D0):
    D = create_D_matrix(rows, cols)
    H = 1 - np.exp(-(D**2) / (2 * (D0**2)))
    return H