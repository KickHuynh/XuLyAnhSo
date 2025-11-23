import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import time 

from processing.hw3_ops_frequency import (
    apply_frequency_filter, IHPF, ILPF, BLPF, BHPF, GLPF, GHPF
)

class TabFrequency(ttk.Frame):
    def __init__(self, parent, main_app_ref=None):
        super().__init__(parent)
        self.main_app = main_app_ref

        self.img_original_cv = None # Ảnh màu gốc
        # self.img_gray_cv = None     
        self.img_processed_cv = None # Ảnh đã xử lý 
        self.slider_timer = None
        self.history = []

        # ===== LAYOUT  =====
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- KHUNG ẢNH ---
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        label_frame = ttk.Frame(left_frame)
        label_frame.pack(fill=tk.X, pady=5)
        ttk.Label(label_frame, text="Ảnh gốc", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT, padx=10)
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
        ttk.Label(scrollable, text="📂 Ảnh nguồn", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=5)
        ttk.Button(scrollable, text="Mở ảnh", command=self.open_image).pack(fill=tk.X, pady=3)
        ttk.Button(scrollable, text="Lưu ảnh", command=self.save_image).pack(fill=tk.X, pady=3)
        ttk.Button(scrollable, text="Hoàn tác (Undo)", command=self.undo_image).pack(fill=tk.X, pady=3)
        ttk.Button(scrollable, text="Khôi phục ảnh gốc", command=self.reset_image).pack(fill=tk.X, pady=3)
        ttk.Separator(scrollable).pack(fill=tk.X, pady=10)
        ttk.Label(scrollable, text="📡 Lọc tần số", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=5)
        self.filter_choice = tk.StringVar(value="ILPF")
        filter_list = ["ILPF", "IHPF", "BLPF", "BHPF", "GLPF", "GHPF"]
        ttk.Label(scrollable, text="Chọn bộ lọc:").pack(anchor="w")
        filter_cb = ttk.Combobox(scrollable, textvariable=self.filter_choice,
                                         values=filter_list, state="readonly")
        filter_cb.pack(fill=tk.X, pady=5)
        filter_cb.bind("<<ComboboxSelected>>", self.on_filter_selected)
        self.param_d0 = tk.DoubleVar(value=50)
        ttk.Label(scrollable, text="Tần số cắt D0:").pack(anchor="w")
        tk.Scale(scrollable, from_=1, to=250, resolution=1, length=280, orient="horizontal",
                variable=self.param_d0, command=lambda e: self.delayed_apply(self.apply_filter_live)).pack(fill=tk.X)
        self.param_n = tk.DoubleVar(value=2)
        self.label_n = ttk.Label(scrollable, text="Bậc n (cho Butterworth):")
        self.label_n.pack(anchor="w")
        self.scale_n = tk.Scale(scrollable, from_=1, to=10, resolution=1, length=280, orient="horizontal",
                variable=self.param_n, command=lambda e: self.delayed_apply(self.apply_filter_live),
                state=tk.DISABLED)
        self.scale_n.pack(fill=tk.X)
        ttk.Button(scrollable, text="Áp dụng lọc", command=self.apply_filter_final).pack(fill=tk.X, pady=5)
        self.on_filter_selected()

    # ======= Nhận ảnh từ MainApp =======
    def set_new_image(self, img_cv):
        """Hàm này được MainApp gọi để tải ảnh mới vào tab này"""
        self.img_original_cv = img_cv.copy() # Lưu ảnh màu gốc
        self.img_processed_cv = self.img_original_cv.copy() # Ảnh xử lý cũng là ảnh màu
        self.history.clear() 
        self.display_images()

    # ======= HÀM TIỆN ÍCH =======
    def delayed_apply(self, func):
        if self.slider_timer:
            self.after_cancel(self.slider_timer)
        self.slider_timer = self.after(150, func) 

    def check_image_loaded(self):
        if self.img_original_cv is None:
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
        if self.img_processed_cv is None:
            messagebox.showwarning("⚠️ Cảnh báo", "Không có ảnh đã xử lý để lưu!")
            return
        path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                             filetypes=[("JPEG", "*.jpg"), ("PNG", ".png")])
        if path:
            # === Lưu ảnh màu ===
            cv2.imwrite(path, self.img_processed_cv)
            messagebox.showinfo("✅ Thành công", f"Đã lưu ảnh (màu) tại:\n{path}")

    # === HÀM "HIỂN THỊ" ===
    def display_images(self):
        def render(cv_img, canvas):
            try:
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
        
        # ===  Hiển thị ảnh màu gốc ===
        if self.img_original_cv is not None:
            render(self.img_original_cv, self.original_canvas)
        if self.img_processed_cv is not None:
            render(self.img_processed_cv, self.edited_canvas)

    def undo_image(self):
        if not self.history:
            messagebox.showinfo("Thông báo", "Không có thao tác để hoàn tác.")
            return
        self.img_processed_cv = self.history.pop()
        self.display_images()

    # === HÀM RESET_IMAGE  ===
    def reset_image(self):
        if not self.check_image_loaded(): return
        
        self.img_processed_cv = self.img_original_cv.copy() # Reset về ảnh màu gốc
        self.history.clear()
        self.display_images()
        
        self.param_d0.set(50)
        self.param_n.set(2)
        self.filter_choice.set("ILPF")
        
        messagebox.showinfo("Đã khôi phục", "Đã khôi phục ảnh gốc và reset tất cả thanh trượt.")

    def on_filter_selected(self, event=None):
        selected = self.filter_choice.get()
        if selected in ["BLPF", "BHPF"]:
            self.label_n.config(state=tk.NORMAL)
            self.scale_n.config(state=tk.NORMAL)
        else:
            self.label_n.config(state=tk.DISABLED)
            self.scale_n.config(state=tk.DISABLED)

    # ===== HÀM XỬ LÝ (ĐÃ SỬA LỖI LIVE PREVIEW) =====
    
    def _run_filter_logic(self, img_base):
        """Hàm logic chung. Nhận ảnh cơ sở (img_base). Trả về (ảnh kết quả, dictionary thời gian)"""
        if not self.check_image_loaded(): return None, None
        
        mode = self.filter_choice.get()
        d0 = self.param_d0.get()
        n = self.param_n.get()
        
        img_input = img_base.copy() # Dùng ảnh được truyền vào (có thể là original hoặc processed)

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
        if not self.check_image_loaded(): return
        
        # Dùng self.img_processed_cv để xem trước trên ảnh đã chỉnh sửa hiện tại.
        img_base_for_live = self.img_processed_cv 
        
        mode = self.filter_choice.get()
        d0 = self.param_d0.get()
        n = self.param_n.get()
        
        try:
            filter_func = None
            if mode == "ILPF": filter_func = ILPF
            elif mode == "IHPF": filter_func = IHPF
            elif mode == "BLPF": filter_func = BLPF
            elif mode == "BHPF": filter_func = BHPF
            elif mode == "GLPF": filter_func = GLPF
            elif mode == "GHPF": filter_func = GHPF
            else: return

            result_cv, _ = (None, None) # Khởi tạo
            if filter_func:
                if mode in ["BLPF", "BHPF"]:
                    # Truyền ảnh đã xử lý hiện tại vào logic lọc
                    result_cv, _ = apply_frequency_filter(img_base_for_live, filter_func, d0, n)
                else:
                    # Truyền ảnh đã xử lý hiện tại vào logic lọc
                    result_cv, _ = apply_frequency_filter(img_base_for_live, filter_func, d0)

            if result_cv is not None:
                self.display_live_preview(result_cv) # Hiển thị trên canvas 'edited'
        except Exception as e:
            print(f"Lỗi live preview: {e}")
            
    def apply_filter_final(self):
        """Được gọi bởi NÚT BẤM. Áp dụng, lưu history, và hiển thị thời gian."""
        
        # Lưu ảnh đã xử lý hiện tại vào lịch sử TRƯỚC khi áp dụng bộ lọc mới
        self.history.append(self.img_processed_cv.copy())
        
        # Truyền self.img_processed_cv làm ảnh cơ sở cho lần lọc này
        result_cv, timings = self._run_filter_logic(self.img_processed_cv) 
        
        if result_cv is not None:
            self.img_processed_cv = result_cv
            self.display_images()
            
            if timings:
                total_time = sum(timings.values())
                # Sắp xếp lại timings để dễ đọc
                time_order = ['A_Convert_YUV_ms', '1_Forward_DFT_ms', '2_FFT_Shift_ms', 
                              '3_Multiply_Filter_H_ms', '4_IFFT_Shift_ms', '5_Inverse_DFT_ms', 'B_Merge_BGR_ms']
                details = "\n".join([f"  - {step}: {timings[step]:.2f} ms" for step in time_order if step in timings])

                messagebox.showinfo(
                    "Đo thời gian (Miền Tần số - Ảnh màu)",
                    f"Thao tác: {self.filter_choice.get()}\n"
                    f"Tổng thời gian: {total_time:.2f} ms\n\n"
                    f"Chi tiết công đoạn:\n"
                    f"{details}"
                )

    def display_live_preview(self, preview_img):
        """Hiển thị ảnh xem trước trên canvas 'edited'"""
        try:
            canvas_w = self.edited_canvas.winfo_width() - 10
            canvas_h = self.edited_canvas.winfo_height() - 10
            if canvas_w <= 1 or canvas_h <= 1:
                canvas_w, canvas_h = 650, 650

            img_to_show = preview_img
            img_rgb = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil.thumbnail((canvas_w, canvas_h))
            
            img_tk = ImageTk.PhotoImage(img_pil)
            self.edited_canvas.configure(image=img_tk)
            self.edited_canvas.image = img_tk
        except Exception as e:
            print(f"Lỗi hiển thị live preview: {e}")