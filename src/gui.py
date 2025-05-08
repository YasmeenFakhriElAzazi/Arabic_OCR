import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import pytesseract
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
# لو عندك Tesseract في مكان مخصص حدد مساره هنا
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class ArabicOCRTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Arabic OCR Tool")
        self.root.configure(bg="#d3d3d3")

        # الواجهة
        self.setup_ui()

        # صور
        self.original_img = None
        self.processed_img = None

    def setup_ui(self):
        # أزرار التحكم
        button_width = 15  # Fixed width for all buttons
        tk.Button(self.root, text="Upload Image", width=button_width, command=self.upload_image).grid(row=0, column=0, padx=10, pady=5)
        tk.Button(self.root, text="Preprocess", width=button_width, command=self.preprocess_image).grid(row=1, column=0, padx=10, pady=5)
        tk.Button(self.root, text="Convert", width=button_width, command=self.convert_image).grid(row=2, column=0, padx=10, pady=5)
        tk.Button(self.root, text="Reset", width=button_width, command=self.reset_all).grid(row=3, column=0, padx=10, pady=5)

        # أماكن الصور
        self.original_canvas = tk.Label(self.root, text="Original Image", width=50, height=20, bg="lightgray")
        self.original_canvas.grid(row=0, column=1, rowspan=4, padx=10)

        self.processed_canvas = tk.Label(self.root, text="Processed Image", width=50, height=20, bg="white")
        self.processed_canvas.grid(row=0, column=2, rowspan=4, padx=10)

        # مكان النص الناتج
        self.text_box = tk.Text(self.root, height=5, width=100)
        self.text_box.grid(row=4, column=0, columnspan=3, padx=10, pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if not file_path:
            return
        self.original_img = cv2.imread(file_path)
        self.show_image(self.original_img, self.original_canvas)

    def preprocess_image(self):
        if self.original_img is None:
            messagebox.showerror("Error", "Upload an image first.")
            return
        gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        self.processed_img = thresh
        self.show_image(thresh, self.processed_canvas, is_gray=True)

    def reshape_arabic_text(self, text):
        # إعادة تشكيل النص العربي باستخدام arabic_reshaper
        reshaped_text = arabic_reshaper.reshape(text)
        # ضبط اتجاه النص من اليمين لليسار باستخدام bidi
        display_text = get_display(reshaped_text)
        return display_text

    def convert_image(self):
        if self.processed_img is None:
            messagebox.showerror("Error", "Preprocess the image first.")
            return
        # تحويل الصورة لنص عربي
        custom_config = '--psm 6 -l ara'
        text = pytesseract.image_to_string(self.processed_img, config=custom_config)
        reshaped_text = self.reshape_arabic_text(text)
        self.text_box.delete(1.0, tk.END)
        self.text_box.insert(tk.END, reshaped_text)

    def reset_all(self):
        self.original_img = None
        self.processed_img = None
        self.original_canvas.config(image='', text='Original Image')
        self.processed_canvas.config(image='', text='Processed Image')
        self.text_box.delete(1.0, tk.END)

    def show_image(self, img, widget, is_gray=False):
        if is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        max_width, max_height = 400, 300
        img_pil.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)
        widget.config(image=img_tk, text='')
        widget.image = img_tk  # لمنع حذف الصورة من الذاكرة

if __name__ == "__main__":
    root = tk.Tk()
    app = ArabicOCRTool(root)
    root.mainloop()
