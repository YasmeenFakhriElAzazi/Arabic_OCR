import numpy as np
import cv2 as cv
from preprocessing import binary_otsus, deskew, vexpand, hexpand, preprocess_and_label
from utilities import projection, save_image


def preprocess(image, padding=10, apply_blur=True, apply_morph=True):
    """
    Enhanced preprocessing pipeline incorporating the new preprocessing functions
    with additional improvements for Arabic OCR.
    """
    # Convert to grayscale and invert if needed (handled in binary_otsus)
    binary_img = binary_otsus(image, apply_blur=apply_blur, apply_morph=apply_morph)
    
    # Deskew the image
    deskewed_img = deskew(binary_img)
    
    # Add padding to help with edge cases
    if padding > 0:
        padded_img = vexpand(deskewed_img, pad=padding)
        padded_img = hexpand(padded_img, pad=padding)
    else:
        padded_img = deskewed_img
    
    # Optional: Get connected components (can be useful for advanced segmentation)
    _, labels = preprocess_and_label(padded_img, pad=0, apply_blur=False, apply_morph=False)
    
    return padded_img, labels


def projection_segmentation(clean_img, axis, cut=3, min_segment_size=10):
    """
    Improved projection segmentation with better handling of Arabic text characteristics.
    Now works with the enhanced preprocessing output.
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
                segment_size = idx - start
                if segment_size >= min_segment_size:  # Filter out tiny segments
                    if axis == "horizontal":
                        segment = clean_img[max(start - 1, 0):idx, :]
                    elif axis == "vertical":
                        segment = clean_img[:, max(start - 1, 0):idx]
                    
                    # Only add segment if it contains meaningful content
                    if np.mean(segment) < 250:  # Skip nearly white segments
                        segments.append(segment)
                cnt = 0
                start = -1

    # Handle case where text goes until end of image
    if start != -1:
        if axis == "horizontal":
            segment = clean_img[max(start - 1, 0):, :]
        elif axis == "vertical":
            segment = clean_img[:, max(start - 1, 0):]
        
        if segment.shape[0 if axis == "horizontal" else 1] >= min_segment_size:
            segments.append(segment)

    return segments


def line_segmentation(image, cut=3, padding=10):
    """
    Enhanced line segmentation that uses the new preprocessing
    """
    clean_img, _ = preprocess(image, padding=padding)
    lines = projection_segmentation(clean_img, axis="horizontal", cut=cut)
    return lines


def word_segmentation(line_image, cut=3, line_height_ratio=0.15):
    """
    Enhanced word segmentation with dynamic cut based on line height
    """
    # Calculate dynamic cut based on line height
    line_height = line_image.shape[0]
    dynamic_cut = max(cut, int(line_height * line_height_ratio))
    
    words = projection_segmentation(line_image, axis="vertical", cut=dynamic_cut)
    words.reverse()  # Maintain RTL order for Arabic
    
    return words


def extract_words_from_image(img, save=False, cut=3, padding=10, output_dir="output_segments"):
    """
    Complete word extraction pipeline with enhanced preprocessing
    Returns both word images and their metadata
    """
    lines = line_segmentation(img, cut=cut, padding=padding)
    words = []

    for line_idx, line in enumerate(lines):
        line_words = word_segmentation(line, cut=cut)
        
        for word_idx, word_img in enumerate(line_words):
            # Add some padding around the word (helps OCR)
            padded_word = cv.copyMakeBorder(word_img, 5, 5, 5, 5, cv.BORDER_CONSTANT, value=255)
            
            # Store word information with metadata
            word_data = {
                'image': padded_word,
                'line_num': line_idx,
                'word_num': word_idx,
                'original_line': line,
                'coordinates': (line_idx, word_idx)
            }
            
            words.append(word_data)
            
            if save:
                save_image(padded_word, f"{output_dir}/words", f"word_{line_idx}_{word_idx}")

    return words
