import numpy as np
import cv2 # Sử dụng cv2

# Frequency 
def create_D_matrix(rows, cols):
    center_row, center_col = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt((U - center_row)**2 + (V - center_col)**2)
    return D

def apply_frequency_filter(img_gray, H_filter_func, D0, n=None):
    rows, cols = img_gray.shape
    
    # Đảm bảo D0 không phải là 0 để tránh lỗi chia
    if D0 == 0: D0 = 1e-6 
        
    dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    if n is None:
        H = H_filter_func(rows, cols, D0)
    else:
        H = H_filter_func(rows, cols, D0, n)
    
    H_complex = np.zeros_like(dft_shift)
    H_complex[:,:,0] = H
    H_complex[:,:,1] = H
    
    G_shift = dft_shift * H_complex
    G_ishift = np.fft.ifftshift(G_shift)
    img_back = cv2.idft(G_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    img_out = np.uint8(img_back)
    
    return img_out

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