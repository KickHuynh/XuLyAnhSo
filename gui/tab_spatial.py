import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import time 

from processing.hw2_ops_spatial_pil import (
    negative_image, log_transform, gamma_transform, piecewise_linear, equalize_histogram,
    mean_filter_basic, gaussian_filter_basic,
    median_filter_basic, min_filter_basic, max_filter_basic, midpoint_filter_basic,
    sobel_filter_basic, threshold_filter_basic
)

class TabSpatial(ttk.Frame):
    def __init__(self, parent, main_app_ref=None):
        super().__init__(parent)
        self.main_app = main_app_ref
        
        self.img_pil = None         
        self.img_edited_pil = None  
        self.history = []           
        self.slider_timer = None

        # ===== LAYOUT =====
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        label_frame = ttk.Frame(left_frame)
        label_frame.pack(fill=tk.X, pady=5)
        ttk.Label(label_frame, text="Ảnh gốc", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT, padx=10)
        ttk.Label(label_frame, text="Ảnh sau xử lý", font=("Segoe UI", 11, "bold")).pack(side=tk.RIGHT, padx=10)

        self.image_frame = ttk.Frame(left_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        self.original_canvas = tk.Label(self.image_frame, bg="#ddd", relief="sunken")
        self.original_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.edited_canvas = tk.Label(self.image_frame, bg="#ddd", relief="sunken")
        self.edited_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)

        canvas = tk.Canvas(right_frame, bg="#f5f6fa", highlightthickness=0, width=300)
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=canvas.yview)
        self.scrollable = ttk.Frame(canvas) 
        
        self.scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0,0), window=self.scrollable, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # ===== CÁC NÚT BẤM VÀ SLIDER =====
        ttk.Label(self.scrollable, text="📂 Ảnh nguồn", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=5)
        ttk.Button(self.scrollable, text="Mở ảnh", command=self.open_image).pack(fill=tk.X, pady=3)
        ttk.Button(self.scrollable, text="Lưu ảnh", command=self.save_image).pack(fill=tk.X, pady=3)
        ttk.Button(self.scrollable, text="Hoàn tác (Undo)", command=self.undo_image).pack(fill=tk.X, pady=3)
        ttk.Button(self.scrollable, text="Khôi phục ảnh gốc", command=self.reset_image).pack(fill=tk.X, pady=3)
        ttk.Separator(self.scrollable).pack(fill=tk.X, pady=10)

        ttk.Label(self.scrollable, text="⚙️ Biến đổi cường độ", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=5)
        self.transform_choice = tk.StringVar(value="Negative") # Đặt mặc định là Negative
        transform_list = ["Negative", "Log", "Gamma", "Piecewise Linear", "Equalize Histogram", "Threshold"]
        
        ttk.Label(self.scrollable, text="Chọn kiểu biến đổi:").pack(anchor="w")
        transform_cb = ttk.Combobox(self.scrollable, textvariable=self.transform_choice,
                                         values=transform_list, state="readonly")
        transform_cb.pack(fill=tk.X, pady=5)
        
        # Gọi hàm cập nhật tham số và xem trước
        transform_cb.bind("<<ComboboxSelected>>", self.on_transform_change)

        # === 1. FRAME CHO THAM SỐ C/GAMMA (Log, Gamma) ===
        self.frame_c_gamma = ttk.Frame(self.scrollable)
        self.param_c = tk.DoubleVar(value=30)
        ttk.Label(self.frame_c_gamma, text="Tham số c:").pack(anchor="w")
        tk.Scale(self.frame_c_gamma, from_=1, to=100, length=280, orient="horizontal", variable=self.param_c,
                command=lambda e: self.delayed_apply(self.apply_transform)).pack(fill=tk.X)

        self.param_gamma = tk.DoubleVar(value=1.0)
        ttk.Label(self.frame_c_gamma, text="Tham số γ:").pack(anchor="w")
        tk.Scale(self.frame_c_gamma, from_=0.1, to=3.0, resolution=0.1, length=280, orient="horizontal",
                variable=self.param_gamma, command=lambda e: self.delayed_apply(self.apply_transform)).pack(fill=tk.X)

        # === 2. FRAME CHO THAM SỐ LOW/HIGH (Piecewise Linear) ===
        self.frame_low_high = ttk.Frame(self.scrollable)
        self.param_low = tk.DoubleVar(value=0.3)
        self.param_high = tk.DoubleVar(value=0.7)
        ttk.Label(self.frame_low_high, text="Low (0-1):").pack(anchor="w")
        tk.Scale(self.frame_low_high, from_=0, to=1, resolution=0.05, length=280, orient="horizontal",
                variable=self.param_low, command=lambda e: self.delayed_apply(self.apply_transform)).pack(fill=tk.X)
        ttk.Label(self.frame_low_high, text="High (0-1):").pack(anchor="w")
        tk.Scale(self.frame_low_high, from_=0, to=1, resolution=0.05, length=280, orient="horizontal",
                variable=self.param_high, command=lambda e: self.delayed_apply(self.apply_transform)).pack(fill=tk.X)

        # === 3. FRAME CHO THAM SỐ THRESHOLD (Threshold) ===
        self.frame_threshold = ttk.Frame(self.scrollable)
        self.param_thresh = tk.DoubleVar(value=130)
        ttk.Label(self.frame_threshold, text="Ngưỡng (Threshold 0-255):").pack(anchor="w")
        tk.Scale(self.frame_threshold, from_=0, to=255, resolution=1, length=280, orient="horizontal",
                variable=self.param_thresh, command=lambda e: self.delayed_apply(self.apply_transform)).pack(fill=tk.X)
        
        # Khởi tạo: Ẩn/hiện tham số lần đầu
        self.update_transform_controls()
        
        ttk.Button(self.scrollable, text="Áp dụng biến đổi", command=self.apply_transform).pack(fill=tk.X, pady=5)
        ttk.Separator(self.scrollable).pack(fill=tk.X, pady=10)

        # ===== LỌC KHÔNG GIAN =====
        ttk.Label(self.scrollable, text="🧩 Lọc không gian", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=5)
        self.filter_choice = tk.StringVar(value="Mean")
        ttk.Label(self.scrollable, text="Chọn loại lọc:").pack(anchor="w")
        
        filters = ["Mean", "Gaussian", "Median", "Min", "Max", "Midpoint", "Sobel"]
        
        filter_cb = ttk.Combobox(self.scrollable, textvariable=self.filter_choice, values=filters, state="readonly")
        filter_cb.pack(fill=tk.X, pady=5)
        filter_cb.bind("<<ComboboxSelected>>", lambda e: self.apply_filter(live=True))

        self.kernel_size = tk.IntVar(value=3)
        ttk.Label(self.scrollable, text="Kích thước kernel:").pack(anchor="w")
        tk.Scale(self.scrollable, from_=3, to=11, resolution=2, length=280, orient="horizontal",
                variable=self.kernel_size, command=lambda e: self.delayed_apply(self.apply_filter)).pack(fill=tk.X)
        ttk.Button(self.scrollable, text="Áp dụng lọc", command=self.apply_filter).pack(fill=tk.X, pady=5)
        
    # === CẬP NHẬT GIAO DIỆN THAM SỐ ===
    def on_transform_change(self, event):
        """Xử lý sự kiện khi kiểu biến đổi thay đổi."""
        self.update_transform_controls()
        self.apply_transform(live=True) 

    def update_transform_controls(self):
        """Ẩn/hiện các Frame tham số dựa trên lựa chọn ComboBox."""
        
        # 1. Ẩn tất cả các Frame
        self.frame_c_gamma.pack_forget()
        self.frame_low_high.pack_forget()
        self.frame_threshold.pack_forget()

        mode = self.transform_choice.get()
        
        # 2. Hiển thị Frame tương ứng
        if mode in ["Log", "Gamma"]:
            self.frame_c_gamma.pack(fill=tk.X)
        elif mode == "Piecewise Linear":
            self.frame_low_high.pack(fill=tk.X)
        elif mode == "Threshold":
            self.frame_threshold.pack(fill=tk.X)
        # Negative và Equalize Histogram không cần tham số


    # ======= HÀM CHUYỂN ĐỔI CV2 <-> PIL =======
    def cv2_to_pil(self, img_cv):
        """Chuyển ảnh CV2 (BGR) sang PIL (RGB)."""
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    def pil_to_cv2(self, img_pil):
        """Chuyển ảnh PIL (RGB) sang CV2 (BGR)."""
        img_np = np.array(img_pil)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # ======= HÀM LOGIC CHUNG =======
    def set_new_image(self, img_cv):
        """Hàm này được MainApp gọi. Nhận ảnh CV2, chuyển sang PIL."""
        self.img_pil = self.cv2_to_pil(img_cv)
        self.img_edited_pil = self.img_pil.copy()
        self.history.clear()
        self.display_images()

    def delayed_apply(self, func):
        if self.slider_timer:
            self.after_cancel(self.slider_timer)
        self.slider_timer = self.after(150, lambda: func(live=True))

    def check_image_loaded(self):
        if self.img_pil is None: 
            messagebox.showwarning("⚠️ Cảnh báo", "Vui lòng mở ảnh trước!")
            return False
        return True

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path: return
        try:
            img_cv = cv2.imread(path) 
            if img_cv is None:
                raise Exception(f"Không thể đọc file: {path}")
            self.set_new_image(img_cv)
        except Exception as e:
            messagebox.showerror("Lỗi mở ảnh", str(e))

    def save_image(self):
        if self.img_edited_pil is None: 
            messagebox.showwarning("⚠️ Cảnh báo", "Không có ảnh để lưu!")
            return
        path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                             filetypes=[("JPEG", "*.jpg"), ("PNG", ".png")])
        if path:
            img_to_save_cv = self.pil_to_cv2(self.img_edited_pil)
            cv2.imwrite(path, img_to_save_cv)
            messagebox.showinfo("✅ Thành công", f"Đã lưu ảnh tại:\n{path}")

    def display_images(self):
        def render(pil_img, canvas): 
            try:
                canvas_w = canvas.winfo_width() - 10
                canvas_h = canvas.winfo_height() - 10
                if canvas_w <= 1 or canvas_h <= 1:
                    canvas_w, canvas_h = 650, 650

                img_pil_copy = pil_img.copy() 
                img_pil_copy.thumbnail((canvas_w, canvas_h))
                
                img_tk = ImageTk.PhotoImage(img_pil_copy)
                canvas.configure(image=img_tk)
                canvas.image = img_tk
            except Exception as e:
                print(f"Lỗi hiển thị ảnh: {e}")

        if self.img_pil is not None:
            render(self.img_pil, self.original_canvas)
        if self.img_edited_pil is not None:
            render(self.img_edited_pil, self.edited_canvas)

    def undo_image(self):
        if not self.history:
            messagebox.showinfo("Thông báo", "Không có thao tác để hoàn tác.")
            return
        self.img_edited_pil = self.history.pop() 
        self.display_images()

    def reset_image(self):
        if not self.check_image_loaded(): return
        self.img_edited_pil = self.img_pil.copy() 
        self.history.clear()
        self.display_images()

    def display_live_preview(self, preview_pil_img):
        """Hiển thị ảnh xem trước (PIL) trên canvas 'edited'"""
        try:
            canvas_w = self.edited_canvas.winfo_width() - 10
            canvas_h = self.edited_canvas.winfo_height() - 10
            if canvas_w <= 1 or canvas_h <= 1:
                canvas_w, canvas_h = 650, 650

            img_pil_copy = preview_pil_img.copy()
            img_pil_copy.thumbnail((canvas_w, canvas_h))
            
            img_tk = ImageTk.PhotoImage(img_pil_copy)
            self.edited_canvas.configure(image=img_tk)
            self.edited_canvas.image = img_tk
        except Exception as e:
            print(f"Lỗi hiển thị live preview (PIL): {e}")


    # ===== BIẾN ĐỔI CƯỜNG ĐỘ (ĐÃ SỬA LIVE PREVIEW) =====
    def apply_transform(self, live=False):
        if not self.check_image_loaded(): return
        
        # 1. XÁC ĐỊNH ẢNH ĐẦU VÀO VÀ LƯU LỊCH SỬ
        if live:
            # Ảnh đầu vào là ảnh đã chỉnh sửa hiện tại
            img_input = self.img_edited_pil.copy() 
        else:
            # Áp dụng: Lưu lịch sử và dùng ảnh đã chỉnh sửa hiện tại làm đầu vào
            self.history.append(self.img_edited_pil.copy()) 
            img_input = self.img_edited_pil.copy() 

        mode = self.transform_choice.get()
        try:
            start_time = time.perf_counter()
            
            # --- Áp dụng biến đổi ---
            if mode == "Negative":
                result = negative_image(img_input)
            elif mode == "Log":
                result = log_transform(img_input, self.param_c.get())
            elif mode == "Gamma":
                result = gamma_transform(img_input, self.param_c.get(), self.param_gamma.get())
            elif mode == "Piecewise Linear":
                result = piecewise_linear(img_input, self.param_low.get(), self.param_high.get())
            elif mode == "Equalize Histogram":
                result = equalize_histogram(img_input)
            elif mode == "Threshold":
                result = threshold_filter_basic(img_input, self.param_thresh.get())
            else:
                return
            
            total_time_ms = (time.perf_counter() - start_time) * 1000

            if live:
                self.display_live_preview(result)
            else:
                self.img_edited_pil = result 
                self.display_images()
                messagebox.showinfo(
                    "Đo thời gian (Miền Không gian)",
                    f"Thao tác: {mode}\n"
                    f"Tổng thời gian: {total_time_ms:.2f} ms"
                )
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi biến đổi (PIL): {e}")

    # ===== LỌC KHÔNG GIAN (ĐÃ SỬA LIVE PREVIEW) =====
    def apply_filter(self, live=False):
        if not self.check_image_loaded(): return
        
        # 1. XÁC ĐỊNH ẢNH ĐẦU VÀO VÀ LƯU LỊCH SỬ
        if live:
            # Ảnh đầu vào là ảnh đã chỉnh sửa hiện tại
            img_input = self.img_edited_pil.copy()
        else:
            # Áp dụng: Lưu lịch sử và dùng ảnh đã chỉnh sửa hiện tại làm đầu vào
            self.history.append(self.img_edited_pil.copy())
            img_input = self.img_edited_pil.copy()

        mode = self.filter_choice.get()
        k = self.kernel_size.get()
        if k % 2 == 0: k += 1 

        try:
            start_time = time.perf_counter()
            
            # --- Áp dụng lọc ---
            if mode == "Mean":
                result = mean_filter_basic(img_input, k)
            elif mode == "Gaussian":
                result = gaussian_filter_basic(img_input, k)
            elif mode == "Median":
                result = median_filter_basic(img_input, k)
            elif mode == "Min":
                result = min_filter_basic(img_input, k)
            elif mode == "Max":
                result = max_filter_basic(img_input, k)
            elif mode == "Midpoint":
                result = midpoint_filter_basic(img_input, k)
            elif mode == "Sobel":
                result = sobel_filter_basic(img_input, k)
            else:
                return

            total_time_ms = (time.perf_counter() - start_time) * 1000

            if live:
                self.display_live_preview(result)
            else:
                self.img_edited_pil = result 
                self.display_images()
                messagebox.showinfo(
                    "Đo thời gian (Miền Không gian)",
                    f"Thao tác: {mode} (kích thước {k}x{k})\n"
                    f"Tổng thời gian: {total_time_ms:.2f} ms"
                )
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi lọc (PIL): {e}")