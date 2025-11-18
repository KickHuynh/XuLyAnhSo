import os, threading, cv2, numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from processing.hw1_utils import list_images, read_bgr, save_jpg_png, center_crop_quarter, rotate_animation
from config import TARGET_W, TARGET_H, DEFAULT_INPUT_DIR, OUTPUT_DIR

class TabGallery(ttk.Frame):
    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app # Lưu lại main_app để gọi "cầu nối"

        self.view_mode = "icon"
        self.folder = DEFAULT_INPUT_DIR
        self.image_paths = list_images(self.folder)
        self.thumbnails = []
        self.selected_index = None
        self.tk_preview = None
        self._resize_after_id = None
        self._resizing = False

        self._build_layout()
        self._refresh_view()
        self.bind("<Configure>", self._on_resize_safe)

    # ===== GIAO DIỆN =====
    def _build_layout(self):
        main = ttk.Frame(self, padding=6)
        main.pack(fill=tk.BOTH, expand=True)

        # Sidebar
        sidebar = ttk.Frame(main, width=230)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))

        ttk.Label(sidebar, text="Thư mục nguồn:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.lbl_folder = ttk.Label(sidebar, text=self.folder, foreground="#236", wraplength=200)
        self.lbl_folder.pack(anchor="w", pady=(0, 6))

        ttk.Button(sidebar, text="Chọn thư mục...", command=self.choose_folder).pack(fill=tk.X, pady=2)
        ttk.Button(sidebar, text="Hiển thị từng ảnh ", command=self.run_show_all).pack(fill=tk.X, pady=2)
        ttk.Button(sidebar, text="Tách RGB", command=self.run_split_rgb).pack(fill=tk.X, pady=2)
        ttk.Button(sidebar, text="Ảnh xám", command=self.run_gray).pack(fill=tk.X, pady=2)
        ttk.Button(sidebar, text="Xoay", command=self.run_rotate).pack(fill=tk.X, pady=2)
        ttk.Button(sidebar, text="Crop", command=self.run_crop).pack(fill=tk.X, pady=2)
        ttk.Button(sidebar, text="Xuất JPG/PNG", command=self.run_export_all).pack(fill=tk.X, pady=2)
        ttk.Button(sidebar, text="AUTO DEMO", command=self.run_auto_demo).pack(fill=tk.X, pady=2)

        # Trung tâm: danh sách / icon
        center = ttk.Frame(main)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(center, bg="#fafafa", highlightthickness=1, highlightbackground="#ccc")
        self.scrollbar = ttk.Scrollbar(center, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Panel phải: xem trước
        right = ttk.Frame(main, width=360)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        ttk.Label(right, text="Ảnh xem trước", font=("Segoe UI", 10, "bold")).pack(anchor="center", pady=4)
        self.preview_label = ttk.Label(right, text="(Chưa chọn ảnh)", anchor="center")
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    # ===== VIEW LOGIC =====
    def _on_resize_safe(self, event):
        if self._resizing:
            return
        self._resizing = True

        def delayed():
            self._resizing = False
            try:
                self._refresh_view()
            except Exception as e:
                print("Lỗi khi refresh:", e)

        if self._resize_after_id:
            self.after_cancel(self._resize_after_id) # Sửa lỗi: self.resize_after_id -> self._resize_after_id
        self._resize_after_id = self.after(300, delayed)

    def _set_view_mode(self, mode):
        self.view_mode = mode
        self._refresh_view()

    def _refresh_view(self):
        self.canvas.delete("all")
        self.image_paths = list_images(self.folder)
        self.thumbnails.clear()

        if self.view_mode == "icon":
            self._draw_icon_view()
        else:
            self._draw_list_view()

        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _draw_icon_view(self):
        self.canvas.update_idletasks()
        canvas_width = self.canvas.winfo_width() or 800
        thumb_w, thumb_h = 180, 120
        col_w, row_h = 220, 180
        max_cols = max(1, canvas_width // col_w)
        x0, y0 = 30, 20
        x, y = x0, y0
        col_count = 0

        for i, path in enumerate(self.image_paths):
            try:
                img = read_bgr(path, ensure_size=False)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(img)
                pil.thumbnail((thumb_w, thumb_h))
                imgtk = ImageTk.PhotoImage(pil)
                self.thumbnails.append(imgtk)

                tag = f"thumb_{i}"
                self.canvas.create_image(x, y, anchor="nw", image=imgtk, tags=(tag,))
                self.canvas.create_text(x + thumb_w // 2, y + thumb_h + 15,
                                        text=os.path.basename(path), anchor="n", font=("Segoe UI", 8))

                self.canvas.tag_bind(tag, "<Button-1>", lambda e, idx=i: self._on_select(idx))
                self.canvas.tag_bind(tag, "<Double-Button-1>", lambda e, idx=i: self._on_double_click(idx))

                col_count += 1
                if col_count >= max_cols:
                    col_count = 0
                    x = x0
                    y += row_h
                else:
                    x += col_w
            except Exception as e:
                print(f"Lỗi vẽ icon: {e}")
                continue

    def _draw_list_view(self):
        y = 10
        for i, path in enumerate(self.image_paths):
            name = os.path.basename(path)
            tag = f"list_{i}"
            self.canvas.create_text(20, y, anchor="nw", text=f"{i+1:02d}. {name}",
                                    font=("Segoe UI", 10), tags=(tag,))
            self.canvas.tag_bind(tag, "<Button-1>", lambda e, idx=i: self._on_select(idx))
            self.canvas.tag_bind(tag, "<Double-Button-1>", lambda e, idx=i: self._on_double_click(idx))
            y += 26

    # ===== ACTIONS =====
    def _on_select(self, idx):
        self.selected_index = idx
        path = self.image_paths[idx]
        self._update_preview(path)

    def _update_preview(self, path):
        try:
            img = read_bgr(path, ensure_size=True)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            pil.thumbnail((350, 200))
            self.tk_preview = ImageTk.PhotoImage(pil)
            self.preview_label.configure(image=self.tk_preview, text="")
            self.preview_label.image = self.tk_preview
        except Exception as e:
            self.preview_label.configure(text=f"Lỗi xem trước:\n{e}")

    def _on_double_click(self, idx):
        path = self.image_paths[idx]
        if self.main_app:
            self.main_app.load_image_to_editors(path)
        else:
            messagebox.showerror("Lỗi", "Không thể liên kết đến trình chỉnh sửa.")

    def choose_folder(self):
        new_dir = filedialog.askdirectory(initialdir=os.path.abspath(self.folder))
        if new_dir:
            self.folder = new_dir
            self.lbl_folder.config(text=self.folder)
            self._refresh_view()

    # ===== CHỨC NĂNG XỬ LÝ ẢNH (ĐÃ ĐIỀN ĐẦY ĐỦ) =====
    
    def _run_in_thread(self, fn):
        threading.Thread(target=fn, daemon=True).start()

    def _get_selected(self):
        if self.selected_index is None:
            messagebox.showwarning("Chưa chọn ảnh", "Hãy click chọn một ảnh trước.")
            return None
        return self.image_paths[self.selected_index]

    def run_show_all(self):
        """Đọc tập ảnh và hiển thị mỗi ảnh trên một cửa sổ riêng."""
        def job():
            cv2.startWindowThread()
            if len(self.image_paths) < 1: # Giảm từ 10 xuống 1 để dễ test
                messagebox.showerror("Lỗi", f"Cần ít nhất 1 ảnh trong thư mục (hiện có {len(self.image_paths)}).")
                return
            for i, p in enumerate(self.image_paths, 1):
                try:
                    img = read_bgr(p, ensure_size=True)
                    win = f"Ảnh {i}: {os.path.basename(p)}"
                    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(win, TARGET_W, TARGET_H)
                    cv2.imshow(win, img)
                except Exception as e:
                    print("Lỗi đọc ảnh:", e)
            messagebox.showinfo("Thông báo", "Đang hiển thị từng ảnh. Nhấn phím 0 để đóng tất cả.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        self._run_in_thread(job)

    def run_split_rgb(self):
        def job():
            cv2.startWindowThread()
            p = self._get_selected()
            if not p: return
            img = read_bgr(p)
            b, g, r = cv2.split(img)
            img_r = cv2.merge([np.zeros_like(b), np.zeros_like(g), r])
            img_g = cv2.merge([np.zeros_like(b), g, np.zeros_like(r)])
            img_b = cv2.merge([b, np.zeros_like(g), np.zeros_like(r)])
            cv2.imshow("Red", img_r)
            cv2.imshow("Green", img_g)
            cv2.imshow("Blue", img_b)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        self._run_in_thread(job)

    def run_gray(self):
        def job():
            cv2.startWindowThread()
            p = self._get_selected()
            if not p: return
            img = read_bgr(p)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Gray", gray)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        self._run_in_thread(job)

    def run_rotate(self):
        def job():
            cv2.startWindowThread()
            p = self._get_selected()
            if not p: return
            img = read_bgr(p)
            rotate_animation(img, steps=100, step_deg=5, delay_ms=100)
        self._run_in_thread(job)

    def run_crop(self):
        def job():
            cv2.startWindowThread()
            p = self._get_selected()
            if not p: return
            img = read_bgr(p)
            crop = center_crop_quarter(img)
            cv2.imshow("Crop", crop)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        self._run_in_thread(job)

    def run_export_all(self):
        def job():
            ok, fail = 0, 0
            for p in self.image_paths:
                try:
                    img = read_bgr(p)
                    base = os.path.splitext(os.path.basename(p))[0]
                    save_jpg_png(base, img)
                    ok += 1
                except:
                    fail += 1
            messagebox.showinfo("Xuất ảnh", f"Thành công: {ok}, Lỗi: {fail}\nThư mục: {os.path.abspath(OUTPUT_DIR)}")
        self._run_in_thread(job)

    def run_auto_demo(self):
        def job():
            cv2.startWindowThread()
            if len(self.image_paths) < 1: # Giảm từ 10 xuống 1
                messagebox.showerror("Lỗi", f"Cần ít nhất 1 ảnh để chạy AUTO DEMO (hiện có {len(self.image_paths)}).")
                return
            p0 = self.image_paths[0]
            img0 = read_bgr(p0)
            b, g, r = cv2.split(img0)
            cv2.imshow("Red", r)
            cv2.imshow("Green", g)
            cv2.imshow("Blue", b)
            cv2.waitKey(0)
            gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Gray", gray)
            cv2.waitKey(0)
            rotate_animation(img0)
            crop = center_crop_quarter(img0)
            cv2.imshow("Crop", crop)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            for p in self.image_paths:
                img = read_bgr(p)
                base = os.path.splitext(os.path.basename(p))[0]
                save_jpg_png(base, img)
            messagebox.showinfo("AUTO DEMO", f"Hoàn tất AUTO DEMO!\nĐã xuất {len(self.image_paths)} ảnh ra thư mục output.")
        self._run_in_thread(job)