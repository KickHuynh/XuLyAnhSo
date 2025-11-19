import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
import io 
import threading 

import matplotlib.pyplot as plt

import processing.hw2_ops_spatial_pil as spatial_ops
import processing.hw3_ops_frequency as freq_ops

class TabBenchmark(ttk.Frame):
    def __init__(self, parent, main_app_ref=None):
        super().__init__(parent)
        self.main_app = main_app_ref
        
        self.image_path = None
        self.img_pil = None
        self.img_gray_cv = None
        self.chart_tk = None 

        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        settings_frame = ttk.Frame(main_frame, width=350)
        settings_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        ttk.Label(settings_frame, text="📊 Bộ So sánh Hiệu năng", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(settings_frame, text="So sánh tốc độ Lọc Gaussian (Không gian - conv) và Lọc GLPF (Tần số - FFT).",
                  wraplength=350).pack(anchor="w", pady=10)
        ttk.Separator(settings_frame).pack(fill=tk.X, pady=15)
        ttk.Button(settings_frame, text="1. Mở ảnh để kiểm tra", command=self.select_image).pack(fill=tk.X, pady=5)
        self.lbl_image_name = ttk.Label(settings_frame, text="Chưa chọn ảnh", style="TLabel")
        self.lbl_image_name.pack(anchor="w", pady=5)
        self.preview_label = tk.Label(settings_frame, bg="#ddd", relief="sunken", text="Xem trước")
        self.preview_label.pack(fill=tk.BOTH, expand=False, pady=10)
        ttk.Separator(settings_frame).pack(fill=tk.X, pady=15)
        self.run_button = ttk.Button(settings_frame, text="2. Bắt đầu So sánh (Lọc Gaussian vs. GLPF)",
                                     command=self.run_benchmark, state=tk.DISABLED)
        self.run_button.pack(fill=tk.X, pady=5)
        self.status_label = ttk.Label(settings_frame, text="", style="TLabel")
        self.status_label.pack(anchor="w", pady=10)
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        ttk.Label(results_frame, text="Bảng Kết quả", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        table_frame = ttk.Frame(results_frame)
        table_frame.pack(fill=tk.X, expand=False, pady=10)
        cols = ('kernel', 'spatial', 'frequency')
        self.tree = ttk.Treeview(table_frame, columns=cols, show='headings', height=8)
        self.tree.heading('kernel', text='Kích thước Kernel')
        self.tree.heading('spatial', text='Miền Không gian (ms) - (Gaussian)')
        self.tree.heading('frequency', text='Miền Tần số (ms) - (GLPF)')
        self.tree.column('kernel', width=150, anchor='center')
        self.tree.column('spatial', width=200, anchor='e')
        self.tree.column('frequency', width=200, anchor='e')
        tree_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
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
            self.img_pil = Image.open(path).convert('RGB')
            img_cv_bgr = cv2.imread(path)
            self.img_gray_cv = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2GRAY)
            preview_pil = self.img_pil.copy()
            preview_pil.thumbnail((300, 200))
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

    # === HÀM 1: BẮT ĐẦU CHẠY ===
    def run_benchmark(self):
        if not self.img_pil or not self.img_gray_cv:
            messagebox.showwarning("Thiếu ảnh", "Vui lòng chọn một ảnh để kiểm tra trước.")
            return

        # 1. Chuẩn bị giao diện
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.chart_label.config(image=None)
        self.chart_label.image = None
        self.status_label.config(text="Đang chạy... (CÓ THỂ RẤT LÂU)... Vui lòng đợi...")
        self.run_button.config(state=tk.DISABLED) 
        self.update_idletasks() 

        # 2. Tạo và khởi động LUỒNG NỀN
        thread = threading.Thread(target=self._benchmark_worker_thread, daemon=True)
        thread.start()

    # === HÀM 2: CÔNG VIỆC NẶNG ===
    def _benchmark_worker_thread(self):
        """Hàm này chạy trong luồng nền, không được đụng vào GUI"""
        try:
            kernel_sizes = [3, 5] 
            spatial_times = []
            freq_times = []
            
            h, w = self.img_gray_cv.shape
            
            for k in kernel_sizes:
                # === GỬI THÔNG BÁO VỀ LUỒNG GUI ===
                # Yêu cầu luồng GUI cập nhật status
                self.after(0, self._update_status, f"Đang kiểm tra kernel {k}x{k}... (Chậm...)")

                # --- Đo lường Miền Không gian (Gaussian Filter - conv chậm) ---
                start_spatial = time.perf_counter()
                spatial_ops.gaussian_filter_basic(self.img_pil, k)
                end_spatial = time.perf_counter()
                spatial_time_ms = (end_spatial - start_spatial) * 1000
                
                # --- Đo lường Miền Tần số (GLPF) ---
                sigma_equiv = k / 6.0 
                d0_equiv = sigma_equiv
                start_freq = time.perf_counter()
                freq_ops.apply_frequency_filter(self.img_gray_cv, freq_ops.GLPF, d0_equiv)
                end_freq = time.perf_counter()
                freq_time_ms = (end_freq - start_freq) * 1000
                
                # --- Lưu kết quả ---
                spatial_times.append(spatial_time_ms)
                freq_times.append(freq_time_ms)

                # === GỬI KẾT QUẢ TẠM THỜI VỀ LUỒNG GUI ===
                # Dùng self.after(0, ...) để yêu cầu luồng GUI chạy hàm này
                self.after(0, self._update_benchmark_table, k, spatial_time_ms, freq_time_ms)
            
            # === BÁO CÁO HOÀN THÀNH VỀ LUỒNG GUI ===
            results = {
                "k_sizes": kernel_sizes,
                "spatial": spatial_times,
                "freq": freq_times
            }
            self.after(0, self._on_benchmark_complete, results)

        except Exception as e:
            # === BÁO LỖI VỀ LUỒNG GUI ===
            self.after(0, self._on_benchmark_error, e)

    # === HÀM 3: CẬP NHẬT GIAO DIỆN (Chạy trên luồng GUI) ===
    def _update_status(self, message):
        """Hàm nhỏ để cập nhật thanh trạng thái"""
        self.status_label.config(text=message)

    def _update_benchmark_table(self, k, spatial_ms, freq_ms):
        """Hàm này được luồng nền gọi để cập nhật từng dòng của bảng"""
        self.tree.insert('', 'end', values=(f"{k}x{k}", 
                                           f"{spatial_ms:.2f} ms", 
                                           f"{freq_ms:.2f} ms"))
        self.tree.yview_moveto(1.0) # Cuộn xuống cuối

    # === HÀM 4: HOÀN THÀNH (Chạy trên luồng GUI) ===
    def _on_benchmark_complete(self, results):
        """Hàm này được luồng nền gọi khi mọi thứ hoàn tất"""
        self.status_label.config(text="Hoàn tất! Đang vẽ biểu đồ...")
        
        # Vẽ biểu đồ
        self.draw_chart(results["k_sizes"], results["spatial"], results["freq"])
        
        self.status_label.config(text="Hoàn tất!")
        self.run_button.config(state=tk.NORMAL) # Bật lại nút bấm

    # === HÀM 5: BÁO LỖI (Chạy trên luồng GUI) ===
    def _on_benchmark_error(self, error):
        """Hàm này được luồng nền gọi nếu có lỗi"""
        self.status_label.config(text="Đã xảy ra lỗi.")
        self.run_button.config(state=tk.NORMAL)
        messagebox.showerror("Lỗi Benchmark", f"Lỗi trong luồng nền: {error}")

    # (Hàm draw_chart giữ nguyên)
    def draw_chart(self, k_sizes, spatial_times, freq_times):
        try:
            plt.figure(figsize=(7, 5), dpi=100)
            
            plt.plot(k_sizes, spatial_times, marker='o', label='Miền Không gian (Gaussian - conv)')
            plt.plot(k_sizes, freq_times, marker='s', color='r', label='Miền Tần số (GLPF - FFT)')
            
            plt.title('So sánh Hiệu năng Lọc Không gian vs. Tần số')
            plt.xlabel('Kích thước Kernel (n x n)')
            plt.ylabel('Thời gian thực thi (mili giây)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            chart_img_pil = Image.open(buf)
            
            label_w = self.chart_label.winfo_width() - 10
            label_h = self.chart_label.winfo_height() - 10
            if label_w <= 1 or label_h <= 1:
                label_w, label_h = 700, 500
            
            chart_img_pil.thumbnail((label_w, label_h))
            
            self.chart_tk = ImageTk.PhotoImage(chart_img_pil)
            self.chart_label.config(image=self.chart_tk)
            
            buf.close()
            plt.close()

        except Exception as e:
            messagebox.showerror("Lỗi vẽ biểu đồ", f"Lỗi: {e}")