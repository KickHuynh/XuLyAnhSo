import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import time # <<< THÊM VÀO

# Import các hàm lọc tần số
from processing.hw3_ops_frequency import (
    apply_frequency_filter, IHPF, ILPF, BLPF, BHPF, GLPF, GHPF
)

class TabFrequency(ttk.Frame):
    def __init__(self, parent, main_app_ref=None):
        super().__init__(parent)
        self.main_app = main_app_ref

        self.img_original_cv = None # Ảnh màu gốc
        self.img_gray_cv = None     # Ảnh xám (đầu vào của bộ lọc)
        self.img_processed_cv = None # Ảnh đã xử lý (đầu ra của bộ lọc)
        self.slider_timer = None
        
        # === THÊM HISTORY ĐỂ "HOÀN TÁC" ===
        self.history = []

        # ===== LAYOUT (Tương tự Tab 2) =====
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- KHUNG ẢNH ---
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        label_frame = ttk.Frame(left_frame)
        label_frame.pack(fill=tk.X, pady=5)
        ttk.Label(label_frame, text="Ảnh gốc (hoặc ảnh xám)", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT, padx=10)
        ttk.Label(label_frame, text="Ảnh sau xử lý (Tần số)", font=("Segoe UI", 11, "bold")).pack(side=tk.RIGHT, padx=10)

        self.image_frame = ttk.Frame(left_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        self.original_canvas = tk.Label(self.image_frame, bg="#ddd", relief="sunken")
        self.original_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.edited_canvas = tk.Label(self.image_frame, bg="#ddd", relief="sunken")
        self.edited_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # --- KHUNG CÔNG CỤ ---
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)

        canvas = tk.Canvas(right_frame, bg="#f5f6fa", highlightthickness=0, width=300)
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=canvas.yview)
        scrollable = ttk.Frame(canvas)

        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0,0), window=scrollable, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # ===== NHÓM 1: Quản lý ảnh (ĐÃ BỔ SUNG) =====
        ttk.Label(scrollable, text="📂 Ảnh nguồn", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=5)
        ttk.Button(scrollable, text="Mở ảnh", command=self.open_image).pack(fill=tk.X, pady=3)
        
        # === THÊM NÚT "LƯU ẢNH" ===
        ttk.Button(scrollable, text="Lưu ảnh", command=self.save_image).pack(fill=tk.X, pady=3)
        # === THÊM NÚT "HOÀN TÁC" ===
        ttk.Button(scrollable, text="Hoàn tác (Undo)", command=self.undo_image).pack(fill=tk.X, pady=3)
        
        ttk.Button(scrollable, text="Khôi phục ảnh gốc", command=self.reset_image).pack(fill=tk.X, pady=3)
        ttk.Separator(scrollable).pack(fill=tk.X, pady=10)

        # ===== NHÓM 2: Lọc Miền Tần số =====
        ttk.Label(scrollable, text="📡 Lọc tần số", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=5)
        
        self.filter_choice = tk.StringVar(value="ILPF")
        filter_list = ["ILPF", "IHPF", "BLPF", "BHPF", "GLPF", "GHPF"]
        
        ttk.Label(scrollable, text="Chọn bộ lọc:").pack(anchor="w")
        filter_cb = ttk.Combobox(scrollable, textvariable=self.filter_choice,
                                        values=filter_list, state="readonly")
        filter_cb.pack(fill=tk.X, pady=5)
        filter_cb.bind("<<ComboboxSelected>>", self.on_filter_selected)

        # --- Tham số D0 ---
        self.param_d0 = tk.DoubleVar(value=50)
        ttk.Label(scrollable, text="Tần số cắt D0:").pack(anchor="w")
        tk.Scale(scrollable, from_=1, to=250, resolution=1, length=280, orient="horizontal",
                variable=self.param_d0, command=lambda e: self.delayed_apply(self.apply_filter_live)).pack(fill=tk.X)

        # --- Tham số n (cho Butterworth) ---
        self.param_n = tk.DoubleVar(value=2)
        self.label_n = ttk.Label(scrollable, text="Bậc n (cho Butterworth):")
        self.label_n.pack(anchor="w")
        
        self.scale_n = tk.Scale(scrollable, from_=1, to=10, resolution=1, length=280, orient="horizontal",
                variable=self.param_n, command=lambda e: self.delayed_apply(self.apply_filter_live),
                state=tk.DISABLED) # Bắt đầu với trạng thái DISABLED
        self.scale_n.pack(fill=tk.X)

        # === SỬA LẠI NÚT ÁP DỤNG ===
        ttk.Button(scrollable, text="Áp dụng lọc", command=self.apply_filter_final).pack(fill=tk.X, pady=5)
        self.on_filter_selected()


    # ======= HÀM MỚI: Nhận ảnh từ MainApp =======
    def set_new_image(self, img_cv):
        """Hàm này được MainApp gọi để tải ảnh mới vào tab này"""
        self.img_original_cv = img_cv.copy() # Lưu ảnh màu gốc
        
        if len(img_cv.shape) == 3:
            self.img_gray_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        else:
            self.img_gray_cv = img_cv.copy()
            
        self.img_processed_cv = self.img_gray_cv.copy() 
        self.history.clear() # Xóa history cũ
        self.display_images()

    # ======= HÀM TIỆN ÍCH (ĐÃ BỔ SUNG) =======
    def delayed_apply(self, func):
        if self.slider_timer:
            self.after_cancel(self.slider_timer)
        self.slider_timer = self.after(150, func) # Không cần lambda

    def check_image_loaded(self):
        if self.img_gray_cv is None:
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

    # === HÀM "LƯU ẢNH" MỚI ===
    def save_image(self):
        if self.img_processed_cv is None:
            messagebox.showwarning("⚠️ Cảnh báo", "Không có ảnh đã xử lý để lưu!")
            return
        path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                            filetypes=[("JPEG", "*.jpg"), ("PNG", ".png")])
        if path:
            # Lưu ảnh đã xử lý (ảnh xám)
            cv2.imwrite(path, self.img_processed_cv)
            messagebox.showinfo("✅ Thành công", f"Đã lưu ảnh (xám) tại:\n{path}")

    # === HÀM "HIỂN THỊ" (SỬA LẠI) ===
    def display_images(self):
        def render(cv_img, canvas):
            try:
                if len(cv_img.shape) == 2:
                    img_to_show = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
                else:
                    img_to_show = cv_img

                canvas_w = canvas.winfo_width() - 10
                canvas_h = canvas.winfo_height() - 10
                if canvas_w <= 1 or canvas_h <= 1:
                    canvas_w, canvas_h = 650, 650
                    
                img_rgb = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_pil.thumbnail((canvas_w, canvas_h))
                
                img_tk = ImageTk.PhotoImage(img_pil)
                canvas.configure(image=img_tk)
                canvas.image = img_tk
            except Exception as e:
                print(f"Lỗi hiển thị ảnh: {e}")

        if self.img_gray_cv is not None:
            # Khung gốc luôn hiển thị ảnh xám
            render(self.img_gray_cv, self.original_canvas)
        if self.img_processed_cv is not None:
            # Khung đã sửa hiển thị ảnh đã xử lý
            render(self.img_processed_cv, self.edited_canvas)

    # === HÀM "HOÀN TÁC" MỚI ===
    def undo_image(self):
        if not self.history:
            messagebox.showinfo("Thông báo", "Không có thao tác để hoàn tác.")
            return
        # Lấy ảnh (NumPy array) từ history
        self.img_processed_cv = self.history.pop()
        self.display_images()

    def reset_image(self):
        if not self.check_image_loaded(): return
        self.img_processed_cv = self.img_gray_cv.copy()
        self.history.clear()
        self.display_images()

    def on_filter_selected(self, event=None):
        selected = self.filter_choice.get()
        if selected in ["BLPF", "BHPF"]:
            self.label_n.config(state=tk.NORMAL)
            self.scale_n.config(state=tk.NORMAL)
        else:
            self.label_n.config(state=tk.DISABLED)
            self.scale_n.config(state=tk.DISABLED)

    # ===== HÀM XỬ LÝ (ĐÃ CẬP NHẬT) =====
    
    def _run_filter_logic(self):
        """Hàm logic chung. Trả về (ảnh kết quả, dictionary thời gian)"""
        if not self.check_image_loaded(): return None, None
        
        mode = self.filter_choice.get()
        d0 = self.param_d0.get()
        n = self.param_n.get()
        img_input = self.img_gray_cv.copy() # Luôn lọc từ ảnh xám gốc

        try:
            filter_func = None
            if mode == "ILPF": filter_func = ILPF
            elif mode == "IHPF": filter_func = IHPF
            elif mode == "BLPF": filter_func = BLPF
            elif mode == "BHPF": filter_func = BHPF
            elif mode == "GLPF": filter_func = GLPF
            elif mode == "GHPF": filter_func = GHPF
            else:
                return None, None

            # === HÀM NÀY BÂY GIỜ TRẢ VỀ (result_cv, timings) ===
            if filter_func:
                if mode in ["BLPF", "BHPF"]:
                    return apply_frequency_filter(img_input, filter_func, d0, n)
                else:
                    return apply_frequency_filter(img_input, filter_func, d0)
            return None, None
        
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi lọc tần số: {e}")
            return None, None

    def apply_filter_live(self):
        """Được gọi bởi SLIDER. Chỉ xem trước, không lưu history."""
        result_cv, _ = self._run_filter_logic() # Bỏ qua timings
        if result_cv is not None:
            self.display_live_preview(result_cv) # Hiển thị trên canvas 'edited'
            
    def apply_filter_final(self):
        """Được gọi bởi NÚT BẤM. Áp dụng, lưu history, và hiển thị thời gian."""
        # 1. Lưu trạng thái hiện tại vào history
        self.history.append(self.img_processed_cv.copy())
        
        # 2. Chạy bộ lọc và nhận thời gian
        result_cv, timings = self._run_filter_logic()
        
        # 3. Cập nhật ảnh chính
        if result_cv is not None:
            self.img_processed_cv = result_cv
            self.display_images() # Hiển thị (cập nhật vĩnh viễn)
            
            # === HIỂN THỊ POPUP THÔNG BÁO THỜI GIAN ===
            if timings:
                total_time = sum(timings.values())
                # Format chuỗi thông báo
                details = "\n".join([f"  - {step}: {ms:.2f} ms" for step, ms in timings.items()])
                messagebox.showinfo(
                    "Đo thời gian (Miền Tần số)",
                    f"Thao tác: {self.filter_choice.get()}\n"
                    f"Tổng thời gian: {total_time:.2f} ms\n\n"
                    f"Chi tiết 5 công đoạn:\n"
                    f"{details}"
                )

    def display_live_preview(self, preview_img):
        """Hiển thị ảnh xem trước trên canvas 'edited' (Giống display_images)"""
        try:
            canvas_w = self.edited_canvas.winfo_width() - 10
            canvas_h = self.edited_canvas.winfo_height() - 10
            if canvas_w <= 1 or canvas_h <= 1:
                canvas_w, canvas_h = 650, 650

            img_to_show = cv2.cvtColor(preview_img, cv2.COLOR_GRAY2BGR)
            img_rgb = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil.thumbnail((canvas_w, canvas_h))
            
            img_tk = ImageTk.PhotoImage(img_pil)
            self.edited_canvas.configure(image=img_tk)
            self.edited_canvas.image = img_tk
        except Exception as e:
            print(f"Lỗi hiển thị live preview: {e}")