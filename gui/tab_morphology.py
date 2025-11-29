import tkinter as tk
from tkinter import ttk, filedialog, messagebox, OptionMenu, StringVar
from tkinter.ttk import Scale
from PIL import Image, ImageTk
import cv2
import numpy as np

POPUP_IMAGE_SIZE = 250 

try:
    from processing.hw4_ops_morphology import (
        execute_morphology, execute_homework 
    )
except ImportError:
    def execute_morphology(*args): return [(f"Error: {args[1]} not found", np.zeros((POPUP_IMAGE_SIZE,POPUP_IMAGE_SIZE), dtype=np.uint8))]
    def execute_homework(*args): 
        img_sample = np.zeros((POPUP_IMAGE_SIZE,POPUP_IMAGE_SIZE), dtype=np.uint8) + 127
        return [
            ("Binary Input (A)", img_sample),
            ("HW4-1: Erosion (Custom SE)", img_sample),
            ("HW4-1: Dilation (Custom SE)", img_sample),
            ("HW4-2: A eroded B", img_sample),
            ("HW4-2: Boundary (A - A o B)", img_sample)
        ]
    
class MorphologyTab(ttk.Frame):
    def __init__(self, parent, main_app_ref=None):
        super().__init__(parent)
        self.main_app = main_app_ref

        self.img_original_cv = None 
        self.img_processed_cv = None
        self.history = []
        self.slider_timer = None 

        # Biến điều khiển
        self.op_choice = tk.StringVar(value="Morphological: Erosion")
        self.se_type = tk.StringVar(value="Rect (cv.MORPH_RECT)")
        self.se_size = tk.DoubleVar(value=5) 
        self.iterations = tk.DoubleVar(value=1)
        self.thres_morph = tk.DoubleVar(value=127) 
        self.thres_hw = tk.DoubleVar(value=127)
        self.se_size_hw = tk.DoubleVar(value=3)


        # ===== LAYOUT =====
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- KHUNG ẢNH CHÍNH (original và edited_canvas) ---
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        label_frame = ttk.Frame(left_frame)
        label_frame.pack(fill=tk.X, pady=5)
        ttk.Label(label_frame, text="Ảnh gốc", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT, padx=10)
        ttk.Label(label_frame, text="Ảnh sau xử lý (Hình thái học)", font=("Segoe UI", 11, "bold")).pack(side=tk.RIGHT, padx=10)

        self.image_frame = ttk.Frame(left_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        self.original_canvas = tk.Label(self.image_frame, bg="#ddd", relief="sunken")
        self.original_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.edited_frame = ttk.Frame(self.image_frame) 
        self.edited_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.edited_canvas = tk.Label(self.edited_frame, bg="#ddd", relief="sunken") 
        self.edited_canvas.pack(fill=tk.BOTH, expand=True) 

        # --- KHUNG CÔNG CỤ (Scrollable) ---
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
        
        # --- CÁC NÚT ĐIỀU KHIỂN CHUNG ---
        ttk.Label(scrollable, text="📂 Ảnh nguồn", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=5)
        ttk.Button(scrollable, text="Mở ảnh", command=self.open_image).pack(fill=tk.X, pady=3)
        ttk.Button(scrollable, text="Lưu ảnh", command=self.save_image).pack(fill=tk.X, pady=3)
        ttk.Button(scrollable, text="Hoàn tác (Undo)", command=self.undo_image).pack(fill=tk.X, pady=3)
        ttk.Button(scrollable, text="Khôi phục ảnh gốc", command=self.reset_image).pack(fill=tk.X, pady=3)
        ttk.Separator(scrollable).pack(fill=tk.X, pady=10)

        # --- LỌC HÌNH THÁI HỌC (Điều khiển) ---
        ttk.Label(scrollable, text="🧬 Hình thái học (Morphology)", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=5)
        
        op_list = ["Morphological: Erosion", "Morphological: Dilation", 
                   "Morphological: Opening", "Morphological: Closing",
                   "Morphological: Homework/Exercises"]
        
        ttk.Label(scrollable, text="Chọn Phép toán:").pack(anchor="w")
        op_cb = ttk.Combobox(scrollable, textvariable=self.op_choice, values=op_list, state="readonly")
        op_cb.pack(fill=tk.X, pady=5)
        op_cb.bind("<<ComboboxSelected>>", self.on_operation_selected)

        # --- Tham số chung ---
        self.label_thres = ttk.Label(scrollable, text="Ngưỡng nhị phân (T):")
        self.label_thres.pack(anchor="w")
        tk.Scale(scrollable, from_=0, to=255, resolution=1, length=280, orient="horizontal",
                 variable=self.thres_morph, command=lambda e: self.delayed_apply(self.apply_filter_live)).pack(fill=tk.X)
        
        # --- Tham số SE Type ---
        self.label_se_type = ttk.Label(scrollable, text="Loại Kernel (SE):")
        self.label_se_type.pack(anchor="w")
        se_types = ['Rect (cv.MORPH_RECT)', 'Cross (cv.MORPH_CROSS)', 'Ellipse (cv.MORPH_ELLIPSE)']
        self.se_type_cb = ttk.Combobox(scrollable, textvariable=self.se_type, values=se_types, state="readonly")
        self.se_type_cb.pack(fill=tk.X, pady=5)
        self.se_type_cb.bind("<<ComboboxSelected>>", lambda e: self.delayed_apply(self.apply_filter_live))

        # --- Tham số Kích thước SE ---
        self.label_se_size = ttk.Label(scrollable, text="Kích thước Kernel (k x k, lẻ):")
        self.label_se_size.pack(anchor="w")
        self.scale_se_size = tk.Scale(scrollable, from_=3, to=21, resolution=2, length=280, orient="horizontal",
                 variable=self.se_size, command=lambda e: self.delayed_apply(self.apply_filter_live))
        self.scale_se_size.pack(fill=tk.X)
        
        # --- Tham số Iterations ---
        self.label_iterations = ttk.Label(scrollable, text="Số lần lặp (Iterations):")
        self.label_iterations.pack(anchor="w")
        self.scale_iterations = tk.Scale(scrollable, from_=1, to=10, resolution=1, length=280, orient="horizontal",
                 variable=self.iterations, command=lambda e: self.delayed_apply(self.apply_filter_live))
        self.scale_iterations.pack(fill=tk.X)
        
        ttk.Button(scrollable, text="Áp dụng Phép toán", command=self.apply_filter_final).pack(fill=tk.X, pady=10)

        # --- KHUNG BÀI TẬP HW4 ---
        ttk.Separator(scrollable).pack(fill=tk.X, pady=10)
        ttk.Label(scrollable, text="✅ Giải Bài Tập HW4", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=5)
        
        self.hw_param_frame = ttk.Frame(scrollable)
        self.hw_param_frame.pack(fill=tk.X)
        
        # Điều khiển cho Boundary SE Size (HW4-2)
        ttk.Label(self.hw_param_frame, text="SE size cho Trích Biên:").pack(anchor="w")
        self.scale_se_size_hw = tk.Scale(self.hw_param_frame, from_=3, to=11, resolution=2, length=280, orient="horizontal",
                 variable=self.se_size_hw)
        self.scale_se_size_hw.pack(fill=tk.X)
        
        self.hw_button = ttk.Button(scrollable, text="Chạy HW4-1 & HW4-2", command=self.run_homework)
        self.hw_button.pack(fill=tk.X, pady=3)
        
        self.on_operation_selected()


    # ======= HÀM NHẬN ẢNH TỪ MAINAPP =======
    def set_new_image(self, img_cv):
        self.img_original_cv = img_cv.copy()
        self.img_processed_cv = self.img_original_cv.copy() 
        self.history.clear() 
        self.display_images()

    # ======= HÀM TIỆN ÍCH GUI =======
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
            
            if self.main_app and hasattr(self.main_app, 'load_image_to_editors'):
                self.main_app.load_image_to_editors(path)
            else:
                self.set_new_image(img_cv)
        except Exception as e:
            messagebox.showerror("Lỗi mở ảnh", str(e))

    def save_image(self):
        if self.img_processed_cv is None:
            messagebox.showwarning("⚠️ Cảnh báo", "Không có ảnh đã xử lý để lưu!")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", 
                                             filetypes=[("PNG", ".png"), ("JPEG", "*.jpg")])
        if path:
            cv2.imwrite(path, self.img_processed_cv)
            messagebox.showinfo("✅ Thành công", f"Đã lưu ảnh (màu) tại:\n{path}")

    def undo_image(self):
        if not self.history:
            messagebox.showinfo("Thông báo", "Không có thao tác để hoàn tác.")
            return
        self.img_processed_cv = self.history.pop()
        self.display_images()

    def reset_image(self):
        if not self.check_image_loaded(): return
        
        self.img_processed_cv = self.img_original_cv.copy() 
        self.history.clear()
        self.display_images()
        
        self.thres_morph.set(127)
        self.se_size.set(5)
        self.iterations.set(1)
        self.op_choice.set("Morphological: Erosion")
        self.on_operation_selected()
        
        messagebox.showinfo("Đã khôi phục", "Đã khôi phục ảnh gốc và reset tham số.")

    def on_operation_selected(self, event=None):
        op = self.op_choice.get()
        is_hw = op == "Morphological: Homework/Exercises"

        for widget in [self.label_se_type, self.se_type_cb, self.label_se_size, self.scale_se_size, self.label_iterations, self.scale_iterations]:
            widget.pack_forget()

        if not is_hw:
            self.label_thres.config(text="Ngưỡng nhị phân (T):")
            self.label_thres.pack(anchor="w")
            
            self.label_se_type.pack(anchor="w")
            self.se_type_cb.pack(fill=tk.X, pady=5)
            self.label_se_size.pack(anchor="w")
            self.scale_se_size.pack(fill=tk.X)
            self.label_iterations.pack(anchor="w")
            self.scale_iterations.pack(fill=tk.X)
            
            self.hw_param_frame.pack_forget()
            self.hw_button.pack_forget()
            
            # Xóa các label Pop-up trước khi hiển thị lại canvas đơn
            self.clear_edited_frame(keep_canvas=True)
            self.edited_canvas.pack(fill=tk.BOTH, expand=True) 
            
        else:
            self.label_thres.config(text="Ngưỡng Nhị phân (chung):")
            self.label_thres.pack(anchor="w")
            self.hw_param_frame.pack(fill=tk.X)
            self.hw_button.pack(fill=tk.X, pady=3)
            
            # Hiển thị thông báo khi chọn HW Mode
            self.edited_canvas.pack_forget()
            self.clear_edited_frame() 
            
            msg_label = ttk.Label(self.edited_frame, 
                                  text="Nhấn 'Chạy HW4-1 & HW4-2' để xem 5 kết quả trong cửa sổ mới.", 
                                  font=("Segoe UI", 10), wraplength=POPUP_IMAGE_SIZE*1.5, justify=tk.CENTER)
            msg_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=50)


    # --- HÀM "HIỂN THỊ" ---
    def render_image_on_canvas(self, cv_img, canvas, size=None):
        """Hàm con để hiển thị ảnh trên một canvas cụ thể (original/edited) với tùy chọn size."""
        try:
            if size is None:
                canvas_w = canvas.winfo_width() - 10
                canvas_h = canvas.winfo_height() - 10
            else:
                 canvas_w, canvas_h = size
            
            if canvas_w <= 1 or canvas_h <= 1:
                canvas_w, canvas_h = 650, 650

            if len(cv_img.shape) == 2 or cv_img.shape[2] == 1:
                img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
            else:
                 img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                 
            img_pil = Image.fromarray(img_rgb)
            img_pil.thumbnail((canvas_w, canvas_h))
            
            img_tk = ImageTk.PhotoImage(img_pil)
            canvas.configure(image=img_tk)
            canvas.image = img_tk
        except Exception as e:
            print(f"Lỗi hiển thị ảnh: {e}")
            
    def display_images(self):
        if self.img_original_cv is not None:
            self.render_image_on_canvas(self.img_original_cv, self.original_canvas)
        
        if self.op_choice.get() != "Morphological: Homework/Exercises" and self.img_processed_cv is not None:
             self.render_image_on_canvas(self.img_processed_cv, self.edited_canvas)

    def display_live_preview(self, preview_img):
        if self.op_choice.get() != "Morphological: Homework/Exercises":
             self.render_image_on_canvas(preview_img, self.edited_canvas)

    # --- Xử lý đa ảnh cho Homework (Dùng Pop-up) ---
    def clear_edited_frame(self, keep_canvas=False):
        """Xóa tất cả widget trong edited_frame."""
        for widget in self.edited_frame.winfo_children():
             if keep_canvas and widget == self.edited_canvas:
                continue
             widget.destroy()

    def display_homework_results_popup(self, results, cols=3):
        comp_window = tk.Toplevel(self.winfo_toplevel())
        comp_window.title("HW4: Erosion, Dilation & Boundary Extraction")
        comp_window.transient(self.winfo_toplevel())
        frame = ttk.Frame(comp_window, padding="10")
        frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        max_size = (POPUP_IMAGE_SIZE, POPUP_IMAGE_SIZE)

        for col in range(cols):
             frame.grid_columnconfigure(col, weight=1, uniform="group1")

        for i, (title, img_cv) in enumerate(results):
            col = i % cols
            row = i // cols
            sub_frame = ttk.Frame(frame, borderwidth=1, relief="solid")
            sub_frame.grid(row=row*2, column=col, padx=5, pady=5, sticky=tk.N+tk.S+tk.E+tk.W)
            ttk.Label(sub_frame, text=title, font=("Segoe UI", 9, "bold")).pack(side=tk.TOP, pady=2)
            panel = tk.Label(sub_frame, width=POPUP_IMAGE_SIZE, height=POPUP_IMAGE_SIZE)
            panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            if len(img_cv.shape) == 2:
                img_cv_show = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
            else:
                img_cv_show = img_cv
                
            self.render_image_on_canvas(img_cv_show, panel, size=max_size)


    # ===== HÀM XỬ LÝ LOGIC =====
    
    def _run_morphology_logic(self, img_base):
        if not self.check_image_loaded(): return None
        op = self.op_choice.get()
        params = {
            'thres_morph': self.thres_morph,
            'se_type': self.se_type,
            'se_size': self.se_size,
            'iterations': self.iterations
        }
        
        try:
             results = execute_morphology(cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY), op, params)
             if results and len(results) > 0:
                 return cv2.cvtColor(results[-1][1], cv2.COLOR_GRAY2BGR)
             return None
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi Morphology: {e}")
            return None

    def apply_filter_live(self):
        if self.op_choice.get() == "Morphological: Homework/Exercises":
             return
        result_cv = self._run_morphology_logic(self.img_processed_cv) 
        if result_cv is not None:
             self.display_live_preview(result_cv)

    def apply_filter_final(self):
        if self.op_choice.get() == "Morphological: Homework/Exercises":
             messagebox.showinfo("Thông báo", "Vui lòng sử dụng nút 'Chạy HW4-1 & HW4-2' cho bài tập này.")
             return

        self.history.append(self.img_processed_cv.copy())
        result_cv = self._run_morphology_logic(self.img_processed_cv)
        
        if result_cv is not None:
            self.img_processed_cv = result_cv
            self.display_images()
            messagebox.showinfo("✅ Thành công", f"Đã áp dụng phép toán {self.op_choice.get()}")

    def run_homework(self):
        if not self.check_image_loaded(): return
        self.history.append(self.img_processed_cv.copy())
        params = {
            'thres_hw': self.thres_hw,
            'se_size_hw': self.se_size_hw
        }
        
        try:
            img_gray_base = cv2.cvtColor(self.img_original_cv, cv2.COLOR_BGR2GRAY)
            results = execute_homework(img_gray_base, params)
        except Exception as e:
            messagebox.showerror("Lỗi HW4", f"Lỗi trong quá trình xử lý HW4: {e}")
            return

        if results:
            self.display_homework_results_popup(results, cols=3)
            self.img_processed_cv = self.img_original_cv.copy() 
            self.display_images()
            messagebox.showinfo("✅ Hoàn thành HW4", 
                                "Đã chạy HW4-1 & HW4-2. Kết quả hiển thị trong cửa sổ Pop-up mới.")