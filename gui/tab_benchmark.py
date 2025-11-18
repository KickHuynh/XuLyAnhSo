import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
import io # Dùng để lưu biểu đồ vào bộ nhớ

# Thư viện mới để vẽ biểu đồ
import matplotlib.pyplot as plt

# Import cả hai bộ não logic để so sánh
import processing.hw2_ops_spatial_pil as spatial_ops
import processing.hw3_ops_frequency as freq_ops

class TabBenchmark(ttk.Frame):
    def __init__(self, parent, main_app_ref=None):
        super().__init__(parent)
        self.main_app = main_app_ref
        
        self.image_path = None
        self.img_pil = None
        self.img_gray_cv = None
        self.chart_tk = None # Biến giữ tham chiếu đến ảnh biểu đồ

        # ===== Bố cục Giao diện =====
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Cột Cài đặt (Bên trái) ---
        settings_frame = ttk.Frame(main_frame, width=350)
        settings_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

        ttk.Label(settings_frame, text="📊 Bộ So sánh Hiệu năng", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(settings_frame, text="So sánh tốc độ Lọc Không gian (PIL) và Lọc Tần số (FFT).",
                  wraplength=350).pack(anchor="w", pady=10)

        ttk.Separator(settings_frame).pack(fill=tk.X, pady=15)

        # 1. Chọn ảnh
        ttk.Button(settings_frame, text="1. Mở ảnh để kiểm tra", command=self.select_image).pack(fill=tk.X, pady=5)
        self.lbl_image_name = ttk.Label(settings_frame, text="Chưa chọn ảnh", style="TLabel")
        self.lbl_image_name.pack(anchor="w", pady=5)
        
        # 2. Xem trước ảnh
        self.preview_label = tk.Label(settings_frame, bg="#ddd", relief="sunken", text="Xem trước")
        # === DÒNG SỬA LỖI: Đã xóa 'minheight=200' ===
        self.preview_label.pack(fill=tk.BOTH, expand=False, pady=10)

        # 3. Chạy Benchmark
        ttk.Separator(settings_frame).pack(fill=tk.X, pady=15)
        self.run_button = ttk.Button(settings_frame, text="2. Bắt đầu So sánh (Lọc Gaussian)",
                                     command=self.run_benchmark, state=tk.DISABLED)
        self.run_button.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(settings_frame, text="", style="TLabel")
        self.status_label.pack(anchor="w", pady=10)


        # --- Cột Kết quả (Bên phải) ---
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 1. Bảng Thống kê
        ttk.Label(results_frame, text="Bảng Kết quả", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        
        table_frame = ttk.Frame(results_frame)
        table_frame.pack(fill=tk.X, expand=False, pady=10)

        cols = ('kernel', 'spatial', 'frequency')
        self.tree = ttk.Treeview(table_frame, columns=cols, show='headings', height=8)
        
        self.tree.heading('kernel', text='Kích thước Kernel')
        self.tree.heading('spatial', text='Miền Không gian (ms)')
        self.tree.heading('frequency', text='Miền Tần số (ms)')
        
        self.tree.column('kernel', width=150, anchor='center')
        self.tree.column('spatial', width=200, anchor='e') # anchor 'e' = right-align
        self.tree.column('frequency', width=200, anchor='e')

        tree_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 2. Biểu đồ
        ttk.Label(results_frame, text="Biểu đồ So sánh", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(20, 10))
        self.chart_label = tk.Label(results_frame, bg="#f0f0f0", relief="sunken")
        self.chart_label.pack(fill=tk.BOTH, expand=True)

    # ===== HÀM LOGIC =====

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
            
        try:
            self.image_path = path
            
            # Load cả 2 phiên bản
            self.img_pil = Image.open(path).convert('RGB')
            img_cv_bgr = cv2.imread(path)
            self.img_gray_cv = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2GRAY)
            
            # Hiển thị preview
            preview_pil = self.img_pil.copy()
            preview_pil.thumbnail((300, 200)) # Thu nhỏ
            img_tk = ImageTk.PhotoImage(preview_pil)
            
            self.preview_label.config(image=img_tk, text="")
            self.preview_label.image = img_tk
            
            self.lbl_image_name.config(text=f".../{self.image_path.split('/')[-1]}")
            self.run_button.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở ảnh: {e}")
            self.image_path = None
            self.img_pil = None
            self.img_gray_cv = None
            self.run_button.config(state=tk.DISABLED)

    def run_benchmark(self):
        if not self.img_pil or not self.img_gray_cv:
            messagebox.showwarning("Thiếu ảnh", "Vui lòng chọn một ảnh để kiểm tra trước.")
            return

        # Xóa kết quả cũ
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.chart_label.config(image=None)
        self.chart_label.image = None
        self.status_label.config(text="Đang chạy... Vui lòng đợi...")
        self.update_idletasks() # Ép Tkinter cập nhật giao diện

        try:
            kernel_sizes = [3, 5, 7, 11, 15, 21, 31]
            spatial_times = []
            freq_times = []
            
            # === 1. Đo lường Miền Tần số (chỉ 1 lần) ===
            # Thời gian FFT không phụ thuộc vào bộ lọc, chỉ phụ thuộc kích thước ảnh
            start_freq = time.perf_counter()
            # Dùng GLPF (Gaussian Low Pass) làm đại diện, D0=30 là tùy chọn
            freq_ops.apply_frequency_filter(self.img_gray_cv, freq_ops.GLPF, 30)
            end_freq = time.perf_counter()
            freq_time_ms = (end_freq - start_freq) * 1000
            
            # === 2. Đo lường Miền Không gian (lặp) ===
            for k in kernel_sizes:
                self.status_label.config(text=f"Đang kiểm tra kernel {k}x{k}...")
                self.update_idletasks()
                
                # Đo Spatial
                start_spatial = time.perf_counter()
                # Dùng logic gaussian_filter_pil (vốn dùng conv tự viết)
                spatial_ops.gaussian_filter_basic(self.img_pil, k)
                end_spatial = time.perf_counter()
                spatial_time_ms = (end_spatial - start_spatial) * 1000
                
                # Lưu kết quả
                spatial_times.append(spatial_time_ms)
                freq_times.append(freq_time_ms) # Thời gian freq là hằng số
                
                # Thêm vào bảng
                self.tree.insert('', 'end', values=(f"{k}x{k}", 
                                                   f"{spatial_time_ms:.2f} ms", 
                                                   f"{freq_time_ms:.2f} ms"))
                self.tree.yview_moveto(1.0) # Cuộn xuống cuối
            
            # === 3. Vẽ Biểu đồ ===
            self.draw_chart(kernel_sizes, spatial_times, freq_time_ms)
            self.status_label.config(text="Hoàn tất!")

        except Exception as e:
            self.status_label.config(text="Đã xảy ra lỗi.")
            messagebox.showerror("Lỗi Benchmark", f"Lỗi: {e}")

    def draw_chart(self, k_sizes, spatial_times, freq_time):
        try:
            # Dùng plt để vẽ
            plt.figure(figsize=(7, 5), dpi=100) # Tạo 1 figure mới
            
            # 1. Vẽ đường Lọc Không gian
            plt.plot(k_sizes, spatial_times, marker='o', label='Miền Không gian (Gaussian PIL)')
            
            # 2. Vẽ đường Lọc Tần số (là 1 đường ngang)
            plt.axhline(y=freq_time, color='r', linestyle='--', label='Miền Tần số (FFT + GLPF)')
            
            plt.title('So sánh Hiệu năng Lọc Không gian vs. Tần số')
            plt.xlabel('Kích thước Kernel (n x n)')
            plt.ylabel('Thời gian thực thi (mili giây)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # Lưu biểu đồ vào bộ nhớ (buffer)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Mở ảnh từ buffer và hiển thị lên Label
            chart_img_pil = Image.open(buf)
            
            # Thay đổi kích thước để vừa với Label
            label_w = self.chart_label.winfo_width() - 10
            label_h = self.chart_label.winfo_height() - 10
            if label_w <= 1 or label_h <= 1:
                label_w, label_h = 700, 500 # Kích thước mặc định
            
            chart_img_pil.thumbnail((label_w, label_h))
            
            self.chart_tk = ImageTk.PhotoImage(chart_img_pil)
            self.chart_label.config(image=self.chart_tk)
            
            buf.close()
            plt.close() # Rất quan trọng: Đóng figure để giải phóng bộ nhớ

        except Exception as e:
            messagebox.showerror("Lỗi vẽ biểu đồ", f"Lỗi: {e}")