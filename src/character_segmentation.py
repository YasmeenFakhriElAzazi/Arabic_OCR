import cv2
import numpy as np
import os
from farasa.segmenter import FarasaSegmenter
import pytesseract

class ArabicCharacterSegmenter:
    def __init__(self, use_farasa=True):
        self.use_farasa = use_farasa
        if use_farasa:
            try:
                self.segmenter = FarasaSegmenter(interactive=True)
            except Exception as e:
                print(f"Failed to initialize Farasa: {e}")
                self.use_farasa = False
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

    def preprocess_word(self, word_img):
        """Convert to grayscale and binarize"""
        if len(word_img.shape) == 3:
            word_img = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(
            word_img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

    def segment_word(self, word_img):
        """Complete character segmentation pipeline"""
        processed = self.preprocess_word(word_img)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(processed, connectivity=8)
        
        # Sort right-to-left for Arabic
        components = sorted(
            [stats[i][:4] for i in range(1, num_labels)],
            key=lambda x: x[0],
            reverse=True
        )
        
        # Extract and pad characters
        characters = []
        for x, y, w, h in components:
            char = word_img[y:y+h, x:x+w]
            char = cv2.copyMakeBorder(char, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
            characters.append(char)
        return characters

def extract_words(image_path):
    """Basic word extraction using horizontal projections"""
    img = cv2.imread(image_path, 0)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return []
    
    # Binarize
    binary = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Simple horizontal projection-based line segmentation
    horizontal_proj = np.sum(binary, axis=1)
    threshold = np.mean(horizontal_proj) * 0.5
    lines = []
    start = None
    
    for i, val in enumerate(horizontal_proj):
        if val > threshold and start is None:
            start = i
        elif val <= threshold and start is not None:
            lines.append(binary[start:i, :])
            start = None
    
    if start is not None:  # Add last line if needed
        lines.append(binary[start:, :])
    
    return lines

# if __name__ == "__main__":
#     # Initialize
#     segmenter = ArabicCharacterSegmenter()
#     os.makedirs("output", exist_ok=True)
    
#     # Process image
#     image_path = "tests/1.png"  # Change to your image path
    
#     # Verify image exists
#     if not os.path.exists(image_path):
#         print(f"Error: Image file not found at {image_path}")
#         print("Current working directory:", os.getcwd())
#         exit()
    
#     # Step 1: Extract lines (words in this simple example)
#     words = extract_words(image_path)
#     print(f"Found {len(words)} words/lines")
    
#     # Step 2: Process each word
#     for i, word in enumerate(words):
#         # Segment characters
#         characters = segmenter.segment_word(word)
#         print(f"Word {i}: Found {len(characters)} characters")
        
#         # Save visualization
#         vis = cv2.cvtColor(word, cv2.COLOR_GRAY2BGR)
#         x_offset = 0
#         for j, char in enumerate(characters):
#             h, w = char.shape
#             cv2.rectangle(vis, (x_offset, 0), (x_offset+w, h), (0, 0, 255), 1)
#             cv2.putText(vis, str(j), (x_offset+2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
#             x_offset += w + 2
        
#         cv2.imwrite(f"output/word_{i}.png", vis)
        
#         # Save individual characters
#         for j, char in enumerate(characters):
#             cv2.imwrite(f"output/word_{i}_char_{j}.png", char)
    
#     print("Processing complete. Check output/ directory for results.")