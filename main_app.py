import tkinter as tk
from tkinter import ttk
import cv2
from tkinter import messagebox 
from gui.tab_gallery import TabGallery
from gui.tab_spatial import TabSpatial
from gui.tab_frequency import TabFrequency
from gui.tab_benchmark import TabBenchmark 
from gui.tab_morphology import MorphologyTab 

class MainApp(tk.Tk):
	def __init__(self):
		super().__init__()
		self.title("HỆ THỐNG XỬ LÝ ẢNH TỔNG HỢP - HUỲNH NGỌC TÀI")
		self.geometry("1400x900")
		self.minsize(1200, 700)

		self.notebook = ttk.Notebook(self)
		self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

		try:
			self.tab1 = TabGallery(self.notebook, self)
			self.tab2 = TabSpatial(self.notebook, self)
			self.tab3 = TabFrequency(self.notebook, self)
			self.tab4 = TabBenchmark(self.notebook, self)
			self.tab5 = MorphologyTab(self.notebook, self) 

			self.notebook.add(self.tab1, text='  🖼️ Thư viện (HW1)  ')
			self.notebook.add(self.tab2, text='  ✨ Lọc Không gian (HW2)  ')
			self.notebook.add(self.tab3, text='  📡 Lọc Tần số (HW3)  ')
			self.notebook.add(self.tab5, text='  🧬 Hình thái học (HW4)  ') 
			self.notebook.add(self.tab4, text='  📊 So sánh Hiệu năng  ')

		except ImportError as e:
			error_msg = str(e)
			if 'matplotlib' in error_msg:
				error_msg = ("Lỗi: Không tìm thấy thư viện 'matplotlib'.\n\n"
							 "Tab 'So sánh Hiệu năng' cần thư viện này.\n"
							 "Vui lòng chạy lệnh sau trong terminal:\n\n"
							 "pip install matplotlib\n\n"
							 f"Chi tiết lỗi: {e}")
			elif 'MorphologyTab' in error_msg or 'tab_morphology' in error_msg:
				error_msg = ("Lỗi: Không tìm thấy file hoặc lớp 'MorphologyTab'.\n\n"
							 "Vui lòng đảm bảo file 'gui/tab_morphology.py' tồn tại và có lớp 'MorphologyTab'.\n"
							 f"Chi tiết lỗi: {e}")
			else:
				error_msg = f"Đã xảy ra lỗi import không xác định: {e}"
                 
			self.withdraw() 
			messagebox.showerror("Lỗi Thiếu Thư viện/File", error_msg)
			self.destroy() 
			return
		except Exception as e:
			messagebox.showerror("Lỗi Khởi tạo", f"Đã xảy ra lỗi không xác định: {e}")
			self.destroy()
			return

	def load_image_to_editors(self, image_path):
		try:
			# 1. Đọc ảnh bằng CV2 (định dạng chuẩn của app)
			img_cv = cv2.imread(image_path)
			if img_cv is None:
				raise ValueError(f"Không thể đọc ảnh: {image_path}")

			# 2. Gửi ảnh đến Tab 2 (Lọc Không gian)
			self.tab2.set_new_image(img_cv)

			# 3. Gửi ảnh đến Tab 3 (Lọc Tần số)
			self.tab3.set_new_image(img_cv)
            
			# 4. Gửi ảnh đến Tab 5 (Hình thái học - HW4)
			self.tab5.set_new_image(img_cv)

			# 5. Tự động chuyển qua Tab 2
			self.notebook.select(self.tab2)

		except Exception as e:
			messagebox.showerror("Lỗi tải ảnh", f"Không thể tải ảnh vào trình chỉnh sửa.\nLỗi: {e}")

if __name__ == "__main__":
	app = MainApp()
	if app.winfo_exists():
		app.mainloop()
		