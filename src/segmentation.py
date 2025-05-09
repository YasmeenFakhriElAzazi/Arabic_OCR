import numpy as np
import cv2 as cv
from preprocessing import binary_otsus, deskew
from utilities import projection, save_image


def preprocess(image):
    """
    مرحلة ما قبل التقسيم: تحويل إلى رمادي + قلب الألوان + binarization + تصحيح الميل
    """
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    inverted_img = cv.bitwise_not(gray_img)
    binary_img = binary_otsus(inverted_img, apply_blur=True)
    deskewed_img = deskew(binary_img)
    return deskewed_img


def projection_segmentation(clean_img, axis, cut=3):
    """
    تنفيذ خوارزمية projection segmentation على المحور المطلوب (horizontal / vertical)
    """
    segments = []
    start = -1
    cnt = 0

    projection_bins = projection(clean_img, axis)
    for idx, projection_bin in enumerate(projection_bins):
        if projection_bin != 0:
            cnt = 0
        if projection_bin != 0 and start == -1:
            start = idx
        if projection_bin == 0 and start != -1:
            cnt += 1
            if cnt >= cut:
                if axis == "horizontal":
                    segments.append(clean_img[max(start - 1, 0):idx, :])
                elif axis == "vertical":
                    segments.append(clean_img[:, max(start - 1, 0):idx])
                cnt = 0
                start = -1

    return segments


def line_segmentation(image, cut=3):
    """
    تقسيم الصورة إلى أسطر
    """
    clean_img = preprocess(image)
    lines = projection_segmentation(clean_img, axis="horizontal", cut=cut)
    return lines


def word_segmentation(line_image, cut=3):
    """
    تقسيم السطر الواحد إلى كلمات
    """
    words = projection_segmentation(line_image, axis="vertical", cut=cut)
    words.reverse()
    return words


def extract_words_from_image(img, save=False, cut=3):
    """
    تقسيم الصورة الكاملة إلى كلمات (داخل أسطر)
    """
    lines = line_segmentation(img, cut=cut)
    words = []

    for idx, line in enumerate(lines):
        line_words = word_segmentation(line, cut=cut)
        for w_idx, word_img in enumerate(line_words):
            if save:
                save_image(word_img, "words", f"word_{idx}_{w_idx}")
            words.append((word_img, line))  # ممكن تبسطيها لو مش محتاجين السطر

    return words


# مثال تشغيل
if __name__ == "__main__":
    img = cv.imread("../Dataset/scanned/capr196.png")
    word_imgs = extract_words_from_image(img, save=True, cut=6)
