import numpy as np
import cv2
import time 

# Frequency 
def create_D_matrix(rows, cols):
    center_row, center_col = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt((U - center_row)**2 + (V - center_col)**2)
    return D

# === HÀM ĐÃ ĐƯỢC CẬP NHẬT ĐỂ XỬ LÝ ẢNH MÀU (YUV) ===
def apply_frequency_filter(img_bgr, H_filter_func, D0, n=None):
    
    timings = {} # Dictionary để lưu thời gian
    
    # === CÔNG ĐOẠN A: Chuyển sang YUV ===
    start_time = time.perf_counter()
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)
    timings['A_Convert_YUV_ms'] = (time.perf_counter() - start_time) * 1000
    
    rows, cols = y.shape # Lấy kích thước từ kênh Y
    if D0 == 0: D0 = 1e-6 
    
    # === CÔNG ĐOẠN 1: Forward DFT (trên kênh Y) ===
    start_time = time.perf_counter()
    dft = cv2.dft(np.float32(y), flags=cv2.DFT_COMPLEX_OUTPUT)
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
    y_filtered = np.uint8(img_back) # Đây là kênh Y đã được lọc
    timings['5_Inverse_DFT_ms'] = (time.perf_counter() - start_time) * 1000
    
    # === CÔNG ĐOẠN B: Ghép YUV và chuyển về BGR ===
    start_time = time.perf_counter()
    img_yuv_filtered = cv2.merge([y_filtered, u, v])
    img_bgr_filtered = cv2.cvtColor(img_yuv_filtered, cv2.COLOR_YUV2BGR)
    timings['B_Merge_BGR_ms'] = (time.perf_counter() - start_time) * 1000
    
    # Trả về ảnh MÀU đã lọc và dictionary thời gian
    return img_bgr_filtered, timings

# Cac ham
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

def process_hw3_1_sequential(img_bgr, D0=25):
    start_time = time.perf_counter() 
    img_lowpass, timings_lp = apply_frequency_filter(img_bgr, GLPF, D0)
    img_final, timings_hp = apply_frequency_filter(img_lowpass, GHPF, D0)
    end_time = time.perf_counter()
    total_timings = {
        "Total_time_ms": (end_time - start_time) * 1000,
        "LP_Filter_Time_ms": sum(timings_lp.values()),
        "HP_Filter_Time_ms": sum(timings_hp.values())
    }
    return img_final, total_timings

def process_hw3_2_iterative_ghpf(img_bgr, D0=30, passes=[1, 10, 100]):
    results = {}
    for num_passes in passes:
        start_time_total = time.perf_counter()
        img_processed = img_bgr.copy() 
        for i in range(num_passes):
            img_processed, _ = apply_frequency_filter(img_processed, GHPF, D0)
        end_time_total = time.perf_counter()
        results[num_passes] = {
            'image': img_processed,
            'time_ms': (end_time_total - start_time_total) * 1000
        }
    return results