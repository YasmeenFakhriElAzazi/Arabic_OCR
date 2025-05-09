import cv2 as cv
import numpy as np
from scipy.ndimage import interpolation as inter
from PIL import Image as im


def binary_otsus(image: np.ndarray, apply_blur: bool = True, apply_morph: bool = False) -> np.ndarray:
    """
    تحويل صورة إلى صورة ثنائية (0 و 255) باستخدام خوارزمية Otsu.

    Parameters:
        image (np.ndarray): الصورة الأصلية (ملونة أو رمادية).
        apply_blur (bool): إذا كان True، يتم تطبيق تمويه Gaussian قبل التثبيت.
        apply_morph (bool): إذا كان True، يتم تطبيق Morphological Opening بعد التثبيت.

    Returns:
        np.ndarray: الصورة الثنائية الناتجة.
    """

    # تحويل الصورة إلى رمادية إذا كانت ملونة
    if len(image.shape) == 3:
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray_img = image

    # تطبيق تمويه Gaussian إذا تم تحديده
    if apply_blur:
        gray_img = cv.GaussianBlur(gray_img, (5, 5), 0)

    # تطبيق تثبيت Otsu
    _, binary_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # تطبيق Morphological Opening إذا طُلب
    if apply_morph:
        kernel = np.ones((3, 3), np.uint8)
        binary_img = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel)

    return binary_img


def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score


def deskew(binary_img):
    ht, wd = binary_img.shape
    # _, binary_img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

    # pix = np.array(img.convert('1').getdata(), np.uint8)
    bin_img = binary_img // 255.0

    delta = 0.1
    limit = 3
    angles = np.arange(-limit, limit + delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    # print('Best angle: {}'.formate(best_angle))

    # correct skew
    data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
    img = im.fromarray((255 * data).astype("uint8"))

    # img.save('skew_corrected.png')
    pix = np.array(img)
    return pix


def vexpand(gray_img, pad: int = 10, color: int = 255):
    """Add vertical padding (top and bottom) to a grayscale image."""
    h, w = gray_img.shape[:2] #Extracts the height (h) and width (w) of the input image
    color = np.clip(color, 0, 255)  # Ensure color is within valid range -> This prevents accidental invalid values like 300 or -50.
    pad_block = np.full((pad, w), fill_value=color, dtype=gray_img.dtype) 
    #Creates a new 2D array (pad_block) of size (pad, w) — a strip of pixels as wide as the image and pad pixels tall.
    # All pixels are set to the color value.
    return np.vstack([pad_block, gray_img, pad_block]) #Vertically stacks the padding on top, then the image, then the padding on bottom.


def hexpand(gray_img, pad: int = 10, color: int = 255):
    """Add horizontal padding (left and right) to a grayscale image."""
    h, w = gray_img.shape[:2]
    color = np.clip(color, 0, 255)
    pad_block = np.full((h, pad), fill_value=color, dtype=gray_img.dtype)
    return np.hstack([pad_block, gray_img, pad_block]) #Horizontally stacks the padding on left, then the image, then the padding on right.


def preprocess_and_label(image, pad: int = 0, apply_blur: bool = True, apply_morph: bool = False):
    """
    Optionally adds padding, applies Otsu binarization, and computes connected components.

    Parameters:
        image (np.ndarray): Input image (BGR or grayscale).
        pad (int): Padding (in pixels) to add to all sides. Set to 0 to skip padding.
        apply_blur (bool): Apply Gaussian blur before binarization.
        apply_morph (bool): Apply morphological opening after binarization.

    Returns:
        num_labels (int): Number of connected components found.
        labels (np.ndarray): Labeled image where each connected region has a unique ID.
    """

    # Apply padding if needed
    if pad > 0:
        gray_img = vexpand(gray_img, pad=pad)
        gray_img = hexpand(gray_img, pad=pad)

    # Binarize
    binary_img = binary_otsus(image, apply_blur=apply_blur, apply_morph=apply_morph)

    # Connected components
    num_labels, labels = cv.connectedComponents(binary_img)
    # Labels connected regions of non-zero pixels with unique integers.
    #Isolate individual characters or words:Each connected text region gets a unique label, helping in further processing like segmentation.

    return num_labels, labels
