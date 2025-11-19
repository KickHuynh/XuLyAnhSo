import numpy as np
from PIL import Image, ImageOps, ImageFilter

# =========================================
# 1. Hàm tích chập (Conv)
# =========================================
def conv(A, k, b=0):
    kh, kw = k.shape
    if b > 0:
        h, w = A.shape
        B = np.zeros((h + kh - 1, w + kw - 1)) 
        th = int(kh / 2)
        tw = int(kw / 2)
        B[th:h + th, tw:w + tw] = A
        A = B
    
    h, w = A.shape
    C = np.zeros((h, w)) 
    
    for i in range(0, h - kh + 1):
        for j in range(0, w - kw + 1):
            sA = A[i:i + kh, j:j + kw]
            C[i, j] = np.sum(k * sA)
            
    C = C[0:h - kh + 1, 0:w - kw + 1]
    return C

# =========================================
# 2. Biến đổi cường độ (Transform)
# =========================================
def negative(image):
    """Âm bản. Mong đợi ảnh PIL, trả về ảnh PIL."""
    if image.mode == 'L':
        return ImageOps.invert(image)
    
    np_img = np.array(image)
    np_negative = 255 - np_img
    return Image.fromarray(np_negative)

def log_transform_pil(image, c):
    """Biến đổi Log. Mong đợi ảnh PIL, trả về ảnh PIL."""
    img_rgb = image.convert('RGB')
    img_array = np.array(img_rgb, dtype=np.float64)
    # Thêm 1e-6 để tránh log(0)
    s = c * np.log(1.0 + img_array + 1e-6)
    s = np.clip(s, 0, 255)
    return Image.fromarray(s.astype(np.uint8), 'RGB')

def gamma_transform_pil(image, c, gamma):
    """Biến đổi Gamma. Mong đợi ảnh PIL, trả về ảnh PIL."""
    img = np.array(image, dtype=np.float64) / 255.0
    s = c * np.power(img, gamma)
    s = np.clip(s * 255.0, 0, 255)
    return Image.fromarray(s.astype(np.uint8), 'RGB')

def piecewise_linear_pil(image, low, high):
    """Biến đổi tuyến tính (dùng interp như GUI cũ)."""
    # Chuyển low, high (0-1) về 0-255
    low_val = low * 255.0
    high_val = high * 255.0
    
    # Tạo bảng tra (LUT)
    lut = np.interp(np.arange(256),
                    [0, 127, 255], # Điểm vào
                    [0, low_val, high_val] # Điểm ra
                   ).astype(np.uint8)
    
    if image.mode == 'L':
        return image.point(lut)
    
    # Áp dụng cho từng kênh nếu là ảnh màu
    channels = image.split()
    processed_channels = [ch.point(lut) for ch in channels]
    return Image.merge(image.mode, processed_channels)

def equalize_histogram_pil(image):
    """Cân bằng histogram (dùng PIL)."""
    if image.mode == 'L':
        return ImageOps.equalize(image)
    
    # Xử lý ảnh màu
    img_yuv = image.convert('YUV')
    channels = img_yuv.split()
    y_equalized = ImageOps.equalize(channels[0])
    img_equalized = Image.merge('YUV', (y_equalized, channels[1], channels[2]))
    return img_equalized.convert('RGB')

def threshold_filter_pil(image, threshold_val):
    """Lọc ngưỡng (dùng PIL)."""
    img_gray = image.convert('L') # Chuyển sang ảnh xám
    # 1 (Trắng) nếu > threshold, 0 (Đen) nếu <=
    img_thresh = img_gray.point(lambda p: 255 if p > threshold_val else 0)
    return img_thresh.convert(image.mode) # Chuyển về mode cũ (đen/trắng)

# =========================================
# 3. Lọc không gian (Filter)
# =========================================
def average_filter(image, n):
    """Lọc trung bình. Mong đợi ảnh PIL, trả về ảnh PIL."""
    k = np.ones((n, n)) / (n ** 2)
    r, g, b = image.split()
    r, g, b = np.array(r), np.array(g), np.array(b)
    
    R = conv(r, k, 1)
    G = conv(g, k, 1)
    B = conv(b, k, 1)
    
    return Image.merge('RGB', (Image.fromarray(R.astype('uint8')),
                                 Image.fromarray(G.astype('uint8')),
                                 Image.fromarray(B.astype('uint8'))))

def gaussian_filter_pil(image, n, sigma=1.0):
    """Lọc Gaussian. Mong đợi ảnh PIL, trả về ảnh PIL."""
    k = np.zeros((n, n))
    mid = n // 2
    for i in range(n):
        for j in range(n):
            x = i - mid
            y = j - mid
            k[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    k /= np.sum(k)
    
    r, g, b = image.split()
    r, g, b = np.array(r), np.array(g), np.array(b)
    
    R = conv(r, k, 1)
    G = conv(g, k, 1)
    B = conv(b, k, 1)
    
    return Image.merge('RGB', (Image.fromarray(R.astype('uint8')),
                                 Image.fromarray(G.astype('uint8')),
                                 Image.fromarray(B.astype('uint8'))))

def median_filter(image, n):
    """Lọc trung vị. Mong đợi ảnh PIL, trả về ảnh PIL."""
    # Dùng hàm `median_filter` của bạn (hỗ trợ cả ảnh PIL)
    return image.filter(ImageFilter.MedianFilter(size=n))

def max_min_filter(image, n, filter_type='min'):
    """Lọc Min/Max. Mong đợi ảnh PIL, trả về ảnh PIL."""
    # Dùng hàm `max_min_filter` của bạn (hỗ trợ cả ảnh PIL)
    if filter_type == 'min':
        return image.filter(ImageFilter.MinFilter(size=n))
    else:
        return image.filter(ImageFilter.MaxFilter(size=n))

def midpoint_filter(image, n):
    """Lọc Midpoint. Mong đợi ảnh PIL, trả về ảnh PIL."""
    # Dùng hàm `midpoint_filter` của bạn
    img = np.array(image, dtype=np.uint8)
    h, w, c = img.shape
    s = n // 2
    padded_img = np.pad(img, ((s, s), (s, s), (0, 0)), mode='edge')
    Imid = np.zeros((h, w, c), np.uint8)
    
    for ch in range(c):
        for i in range(h):
            for j in range(w):
                region = padded_img[i:i + n, j:j + n, ch]
                Imin = np.min(region)
                Imax = np.max(region)
                Imid[i, j, ch] = (Imin + Imax) / 2
                
    return Image.fromarray(Imid, mode='RGB')

def sobel_filter_pil(image):
    """Lọc Sobel. Mong đợi ảnh PIL, trả về ảnh PIL."""
    # Dùng hàm `sobel_filter` của bạn
    img_l = image.convert('L') # Chuyển sang ảnh xám
    img = np.array(img_l, dtype=np.float32)
    is_pil = True
    
    ky = np.array([[-1.0, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    kx = np.transpose(ky)
    
    Gx = conv(img, kx, 1)
    Gy = conv(img, ky, 1)
    
    Gm = np.sqrt(Gx**2 + Gy**2)
    
    if Gm.max() > 0:
        Gm = (Gm * 255.0 / Gm.max())
    
    Gm = np.clip(Gm, 0, 255).astype(np.uint8)
    
    return Image.fromarray(Gm, 'L').convert(image.mode)

# =========================================
# 4. Alias (bí danh) cho GUI
# =========================================
negative_image = negative
log_transform = log_transform_pil
gamma_transform = gamma_transform_pil
piecewise_linear = piecewise_linear_pil
equalize_histogram = equalize_histogram_pil
threshold_filter_basic = threshold_filter_pil

mean_filter_basic = average_filter
gaussian_filter_basic = lambda img, k: gaussian_filter_pil(img, k, sigma=1.0) # Thêm sigma
median_filter_basic = median_filter
min_filter_basic = lambda img, k: max_min_filter(img, k, 'min')
max_filter_basic = lambda img, k: max_min_filter(img, k, 'max')
midpoint_filter_basic = midpoint_filter
sobel_filter_basic = lambda img, k: sobel_filter_pil(img) # Bỏ qua k_size