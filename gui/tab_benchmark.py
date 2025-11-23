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
        # Biến lưu ảnh BGR 3 kênh cho lọc Tần số
        self.img_bgr_cv = None 
        # Biến lưu ảnh xám 1 kênh cho các mục đích khác 
        self.img_gray_cv = None 
        self.chart_tk = None 

        # ===== DANH SÁCH BỘ LỌC KHẢ DỤ =====
        # Lưu ý: Các hàm này phải có tham số (image, k)
        self.SPATIAL_FILTERS = {
            "Gaussian": spatial_ops.gaussian_filter_basic,
            "Mean": spatial_ops.mean_filter_basic,
            "Median": spatial_ops.median_filter_basic,
            "Min": spatial_ops.min_filter_basic,
            "Max": spatial_ops.max_filter_basic,
            "Sobel": spatial_ops.sobel_filter_basic # Sobel cần k=3, nhưng ta vẫn truyền k để đồng bộ vòng lặp
        }
        # Lưu ý: Các hàm này phải có tham số (rows, cols, D0, n=None)
        self.FREQUENCY_FILTERS = {
            "GLPF": freq_ops.GLPF,
            "BLPF": freq_ops.BLPF, # Cần bậc n
            "ILPF": freq_ops.ILPF,
            "GHPF": freq_ops.GHPF,
            "BHPF": freq_ops.BHPF, # Cần bậc n
            "IHPF": freq_ops.IHPF,
        }
        
        # ===== LAYOUT =====

        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        settings_frame = ttk.Frame(main_frame, width=350)
        settings_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        ttk.Label(settings_frame, text="📊 Bộ So sánh Hiệu năng", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(settings_frame, text="So sánh tốc độ các Bộ lọc Không gian (S) và Tần số (F).",
                  wraplength=350).pack(anchor="w", pady=10)
        
        ttk.Separator(settings_frame).pack(fill=tk.X, pady=15)
        
        # --- CÁC TÙY CHỌN BỔ SUNG ---
        
        ttk.Label(settings_frame, text="Cấu hình Bộ lọc:", font=("Segoe UI", 11, "bold")).pack(anchor="w")

        # Bộ lọc Không gian 1
        ttk.Label(settings_frame, text="Bộ lọc S1 (Không gian):").pack(anchor="w", pady=(5, 0))
        self.spatial1_name = tk.StringVar(value="Gaussian")
        self.spatial1_cb = ttk.Combobox(settings_frame, textvariable=self.spatial1_name, 
                                        values=list(self.SPATIAL_FILTERS.keys()), state="readonly")
        self.spatial1_cb.pack(fill=tk.X, pady=(0, 5))
        
        # Bộ lọc Không gian 2
        ttk.Label(settings_frame, text="Bộ lọc S2 (Không gian):").pack(anchor="w", pady=(5, 0))
        self.spatial2_name = tk.StringVar(value="Mean")
        self.spatial2_cb = ttk.Combobox(settings_frame, textvariable=self.spatial2_name, 
                                        values=list(self.SPATIAL_FILTERS.keys()), state="readonly")
        self.spatial2_cb.pack(fill=tk.X, pady=(0, 5))

        # Bộ lọc Tần số 1
        ttk.Label(settings_frame, text="Bộ lọc F1 (Tần số):").pack(anchor="w", pady=(5, 0))
        self.freq1_name = tk.StringVar(value="GLPF")
        self.freq1_cb = ttk.Combobox(settings_frame, textvariable=self.freq1_name, 
                                     values=list(self.FREQUENCY_FILTERS.keys()), state="readonly")
        self.freq1_cb.pack(fill=tk.X, pady=(0, 5))
        
        # Bộ lọc Tần số 2
        ttk.Label(settings_frame, text="Bộ lọc F2 (Tần số):").pack(anchor="w", pady=(5, 0))
        self.freq2_name = tk.StringVar(value="BLPF")
        self.freq2_cb = ttk.Combobox(settings_frame, textvariable=self.freq2_name, 
                                     values=list(self.FREQUENCY_FILTERS.keys()), state="readonly")
        self.freq2_cb.pack(fill=tk.X, pady=(0, 15))
        
        # --- Mở ảnh ---
        ttk.Button(settings_frame, text="1. Mở ảnh để kiểm tra", command=self.select_image).pack(fill=tk.X, pady=5)
        self.lbl_image_name = ttk.Label(settings_frame, text="Chưa chọn ảnh", style="TLabel")
        self.lbl_image_name.pack(anchor="w", pady=5)
        self.preview_label = tk.Label(settings_frame, bg="#ddd", relief="sunken", text="Xem trước")
        self.preview_label.pack(fill=tk.BOTH, expand=False, pady=10)
        
        ttk.Separator(settings_frame).pack(fill=tk.X, pady=15)
        
        self.run_button = ttk.Button(settings_frame, text="2. Bắt đầu So sánh (4 Bộ lọc)",
                                     command=self.run_benchmark, state=tk.DISABLED)
        self.run_button.pack(fill=tk.X, pady=5)
        self.status_label = ttk.Label(settings_frame, text="", style="TLabel")
        self.status_label.pack(anchor="w", pady=10)
        
        # --- BẢNG KẾT QUẢ ---
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        ttk.Label(results_frame, text="Bảng Kết quả", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        table_frame = ttk.Frame(results_frame)
        table_frame.pack(fill=tk.X, expand=False, pady=10)
        
        # Cột mới: S1, S2, F1, F2 
        cols = ('kernel', 'spatial1', 'spatial2', 'frequency1', 'frequency2')
        self.tree = ttk.Treeview(table_frame, columns=cols, show='headings', height=8)
        self.tree.heading('kernel', text='Kích thước K')
        self.tree.heading('spatial1', text='S1 (Gaussian) ms')
        self.tree.heading('spatial2', text='S2 (Mean) ms')
        self.tree.heading('frequency1', text='F1 (GLPF) ms')
        self.tree.heading('frequency2', text='F2 (BLPF) ms')
        
        self.tree.column('kernel', width=100, anchor='center')
        self.tree.column('spatial1', width=120, anchor='e')
        self.tree.column('spatial2', width=120, anchor='e')
        self.tree.column('frequency1', width=120, anchor='e')
        self.tree.column('frequency2', width=120, anchor='e')
        
        tree_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # --- BIỂU ĐỒ ---
        ttk.Label(results_frame, text="Biểu đồ So sánh", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(20, 10))
        self.chart_label = tk.Label(results_frame, bg="#f0f0f0", relief="sunken")
        self.chart_label.pack(fill=tk.BOTH, expand=True)
        
        # Cập nhật tên cột ngay khi khởi tạo
        self.update_headings()
        self.spatial1_cb.bind("<<ComboboxSelected>>", lambda e: self.update_headings())
        self.spatial2_cb.bind("<<ComboboxSelected>>", lambda e: self.update_headings())
        self.freq1_cb.bind("<<ComboboxSelected>>", lambda e: self.update_headings())
        self.freq2_cb.bind("<<ComboboxSelected>>", lambda e: self.update_headings())


    # ===== HÀM LOGIC =====
    
    def update_headings(self):
        """Cập nhật tiêu đề bảng dựa trên lựa chọn Combobox."""
        self.tree.heading('spatial1', text=f"S1 ({self.spatial1_name.get()}) ms")
        self.tree.heading('spatial2', text=f"S2 ({self.spatial2_name.get()}) ms")
        self.tree.heading('frequency1', text=f"F1 ({self.freq1_name.get()}) ms")
        self.tree.heading('frequency2', text=f"F2 ({self.freq2_name.get()}) ms")
        
    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        try:
            self.image_path = path
            self.img_pil = Image.open(path).convert('RGB')
            img_cv_bgr_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            
            if img_cv_bgr_raw is None:
                raise ValueError("OpenCV không thể đọc file ảnh")

            # CHUẨN HÓA ẢNH MÀU CHO LỌC TẦN SỐ (BẮT BUỘC 3 KÊNH BGR)
            if img_cv_bgr_raw.ndim == 2:
                # Nếu ảnh ban đầu là xám, ta chuyển nó thành BGR
                self.img_bgr_cv = cv2.cvtColor(img_cv_bgr_raw, cv2.COLOR_GRAY2BGR)
            elif img_cv_bgr_raw.ndim == 4:
                # Nếu ảnh có alpha channel (4), chuyển BGRA -> BGR
                self.img_bgr_cv = cv2.cvtColor(img_cv_bgr_raw, cv2.COLOR_BGRA2BGR)
            else:
                self.img_bgr_cv = img_cv_bgr_raw
                
            # Tạo ảnh Xám cho Lọc Không gian (nếu cần, hoặc giữ nguyên logic cũ)
            self.img_gray_cv = cv2.cvtColor(self.img_bgr_cv, cv2.COLOR_BGR2GRAY)


            # Xem trước
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
            self.img_bgr_cv = None # Đảm bảo reset biến mới
            self.img_gray_cv = None
            self.run_button.config(state=tk.DISABLED)

    # === HÀM 1: BẮT ĐẦU CHẠY ===
    def run_benchmark(self):
        # Kiểm tra biến BGR mới
        if self.img_pil is None or self.img_bgr_cv is None:
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
        self.update_headings() # Cập nhật tiêu đề lần cuối trước khi chạy

        # 2. Tạo và khởi động LUỒNG NỀN
        thread = threading.Thread(target=self._benchmark_worker_thread, daemon=True)
        thread.start()

    # === HÀM 2: CÔNG VIỆC NẶNG (ĐÃ CẢI TIẾN) ===
    def _benchmark_worker_thread(self):
        """Hàm này chạy trong luồng nền, không được đụng vào GUI"""
        try:
            # Tăng kích thước kernel để thấy rõ sự khác biệt hiệu năng
            kernel_sizes = [3, 5, 9, 15] 
            
            # Khởi tạo kết quả
            results = {
                "k_sizes": kernel_sizes,
                "S1": [], "S2": [], "F1": [], "F2": []
            }
            
            # Lấy hàm lọc dựa trên tên chọn từ Combobox
            s1_name = self.spatial1_name.get()
            s2_name = self.spatial2_name.get()
            f1_name = self.freq1_name.get()
            f2_name = self.freq2_name.get()

            s1_func = self.SPATIAL_FILTERS[s1_name]
            s2_func = self.SPATIAL_FILTERS[s2_name]
            f1_func = self.FREQUENCY_FILTERS[f1_name]
            f2_func = self.FREQUENCY_FILTERS[f2_name]
            
            # Lưu tên ngắn gọn để đặt nhãn biểu đồ
            results["S1_label"] = s1_name
            results["S2_label"] = s2_name
            results["F1_label"] = f1_name
            results["F2_label"] = f2_name
            
            for k in kernel_sizes:
                self.after(0, self._update_status, f"Đang kiểm tra kernel {k}x{k}... (Chậm...)")
                
                # 1. ĐO LƯỜNG MIỀN KHÔNG GIAN (S1, S2)
                
                # S1
                start_s1 = time.perf_counter()
                s1_func(self.img_pil, k)
                end_s1 = time.perf_counter()
                s1_time = (end_s1 - start_s1) * 1000
                
                # S2
                start_s2 = time.perf_counter()
                s2_func(self.img_pil, k)
                end_s2 = time.perf_counter()
                s2_time = (end_s2 - start_s2) * 1000
                
                # 2. ĐO LƯỜNG MIỀN TẦN SỐ (F1, F2)
                
                d0_equiv = k / 6.0
                n_butterworth = 2 # Giả định bậc n cố định cho BLPF/BHPF
                
                # F1
                start_f1 = time.perf_counter()
                # BLPF/BHPF cần bậc n
                if f1_name in ["BLPF", "BHPF"]: 
                    freq_ops.apply_frequency_filter(self.img_bgr_cv, f1_func, d0_equiv, n=n_butterworth)
                else:
                    freq_ops.apply_frequency_filter(self.img_bgr_cv, f1_func, d0_equiv)
                end_f1 = time.perf_counter()
                f1_time = (end_f1 - start_f1) * 1000

                # F2
                start_f2 = time.perf_counter()
                # BLPF/BHPF cần bậc n
                if f2_name in ["BLPF", "BHPF"]: 
                    freq_ops.apply_frequency_filter(self.img_bgr_cv, f2_func, d0_equiv, n=n_butterworth)
                else:
                    freq_ops.apply_frequency_filter(self.img_bgr_cv, f2_func, d0_equiv)
                end_f2 = time.perf_counter()
                f2_time = (end_f2 - start_f2) * 1000

                # --- Lưu kết quả vào dictionary chính ---
                results["S1"].append(s1_time)
                results["S2"].append(s2_time)
                results["F1"].append(f1_time)
                results["F2"].append(f2_time)

                # === GỬI KẾT QUẢ TẠM THỜI VỀ LUỒNG GUI ===
                self.after(0, self._update_benchmark_table, k, s1_time, s2_time, f1_time, f2_time)
            
            # === BÁO CÁO HOÀN THÀNH VỀ LUỒNG GUI ===
            self.after(0, self._on_benchmark_complete, results)

        except Exception as e:
            # === BÁO LỖI VỀ LUỒNG GUI ===
            self.after(0, self._on_benchmark_error, e)

    # === HÀM 3: CẬP NHẬT GIAO DIỆN (Chạy trên luồng GUI) ===
    def _update_status(self, message):
        """Hàm nhỏ để cập nhật thanh trạng thái"""
        self.status_label.config(text=message)

    def _update_benchmark_table(self, k, s1_ms, s2_ms, f1_ms, f2_ms):
        """Hàm này được luồng nền gọi để cập nhật từng dòng của bảng"""
        self.tree.insert('', 'end', values=(f"{k}x{k}", 
                                             f"{s1_ms:.2f}", 
                                             f"{s2_ms:.2f}", 
                                             f"{f1_ms:.2f}",
                                             f"{f2_ms:.2f}"))
        self.tree.yview_moveto(1.0) # Cuộn xuống cuối

    # === HÀM 4: HOÀN THÀNH (Chạy trên luồng GUI) ===
    def _on_benchmark_complete(self, results):
        """Hàm này được luồng nền gọi khi mọi thứ hoàn tất"""
        self.status_label.config(text="Hoàn tất! Đang vẽ biểu đồ...")
        
        # Vẽ biểu đồ
        self.draw_chart(results)
        
        self.status_label.config(text="Hoàn tất!")
        self.run_button.config(state=tk.NORMAL) # Bật lại nút bấm

    # === HÀM 5: BÁO LỖI (Chạy trên luồng GUI) ===
    def _on_benchmark_error(self, error):
        """Hàm này được luồng nền gọi nếu có lỗi"""
        self.status_label.config(text="Đã xảy ra lỗi.")
        self.run_button.config(state=tk.NORMAL)
        messagebox.showerror("Lỗi Benchmark", f"Lỗi trong luồng nền: {error}")

    # (Hàm draw_chart đã được cập nhật để vẽ 4 đường với nhãn động)
    def draw_chart(self, results):
        try:
            k_sizes = results["k_sizes"]
            
            plt.figure(figsize=(7, 5), dpi=100)
            
            # S1
            plt.plot(k_sizes, results["S1"], marker='o', linestyle='-', color='blue', 
                     label=f'S1 ({results["S1_label"]} - conv)')
            # S2
            plt.plot(k_sizes, results["S2"], marker='o', linestyle='--', color='cyan', 
                     label=f'S2 ({results["S2_label"]} - conv)')
            
            # F1
            plt.plot(k_sizes, results["F1"], marker='s', linestyle='-', color='red', 
                     label=f'F1 ({results["F1_label"]} - FFT)')
            # F2
            plt.plot(k_sizes, results["F2"], marker='s', linestyle='--', color='orange', 
                     label=f'F2 ({results["F2_label"]} - FFT)')
            
            plt.title('So sánh Hiệu năng Lọc Không gian vs. Tần số')
            plt.xlabel('Kích thước Kernel (n x n)')
            plt.ylabel('Thời gian thực thi (mili giây)')
            
            plt.yscale('log') # Thường dùng thang log để thấy rõ sự khác biệt tốc độ
            
            plt.legend()
            plt.grid(True, which="both", ls="--", linewidth=0.5)
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