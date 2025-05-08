import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2

class OCRGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Arabic OCR Tool")
        self.geometry("800x450")
        self.resizable(False, False)

        # --- Top: Upload row ---
        tk.Button(self, text="Upload Image", command=self.load_image)\
            .place(x=20, y=20, width=100, height=30)
        self.path_entry = tk.Entry(self)
        self.path_entry.place(x=140, y=20, width=620, height=30)

        # --- Left: Preprocess & Convert buttons + single canvas ---
        tk.Button(self, text="Preprocess", command=self.preprocess_image)\
            .place(x=20, y=70, width=100, height=30)
        tk.Button(self, text="Convert", command=self.convert_image)\
            .place(x=20, y=110, width=100, height=30)

        self.canvas_img = tk.Canvas(self, bg="lightgray")
        self.canvas_img.place(x=140, y=70, width=300, height=300)

        # --- Right: OCR text output ---
        self.text_out = tk.Text(self, wrap="word")
        self.text_out.place(x=460, y=70, width=300, height=300)

        # Internal image holders
        self.original = None
        self.processed = None

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp")])
        if not path: return
        self.path_entry.delete(0, tk.END)
        self.path_entry.insert(0, path)
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error","Failed to open image.")
            return
        self.original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._display(self.original)

    def preprocess_image(self):
        if self.original is None:
            messagebox.showwarning("Warning","Upload an image first.")
            return
        # stub: replace with your real preprocessing
        gray = cv2.cvtColor(self.original, cv2.COLOR_RGB2GRAY)
        _, th = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.processed = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
        self._display(self.processed)

    def convert_image(self):
        if self.processed is None:
            messagebox.showwarning("Warning","Preprocess first.")
            return
        # stub: replace with your OCR.run or pytesseract
        self.text_out.delete("1.0", tk.END)
        self.text_out.insert(tk.END,
            "مرحبا بالعالم\nThis is the OCR output placeholder."
        )

    def _display(self, rgb_img):
        """Draw a numpy RGB image onto our single canvas."""
        w, h = 300, 300
        img = Image.fromarray(rgb_img)
        img = img.resize((w,h), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(img)
        self.canvas_img.photo = photo
        self.canvas_img.create_image(0,0, anchor="nw", image=photo)

if __name__ == "__main__":
    app = OCRGui()
    app.mainloop()
