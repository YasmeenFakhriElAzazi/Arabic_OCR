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
                print(f"Farasa initialization warning: {e}")
                self.use_farasa = False
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

    def preprocess_word(self, word_img):
        """Enhanced preprocessing specifically for Arabic text"""
        if len(word_img.shape) == 3:
            word_img = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
        
        # Contrast enhancement
        word_img = cv2.equalizeHist(word_img)
        
        # Adaptive thresholding with Arabic-optimized parameters
        binary = cv2.adaptiveThreshold(
            word_img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 8
        )
        
        # Morphological operations for Arabic text
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        
        return processed

    def segment_word(self, word_img):
        """Advanced segmentation for connected Arabic characters"""
        processed = self.preprocess_word(word_img)
        
        # Vertical projection analysis
        vertical_projection = np.sum(processed, axis=0)
        
        # Find segmentation points
        segmentation_points = []
        in_char = False
        char_start = 0
        
        for i, val in enumerate(vertical_projection):
            if val > 0 and not in_char:
                in_char = True
                char_start = i
            elif val == 0 and in_char:
                in_char = False
                # Only consider segments wider than 3 pixels
                if i - char_start > 3:  
                    segmentation_points.append((char_start, i))
        
        # Handle the last character if needed
        if in_char:
            segmentation_points.append((char_start, len(vertical_projection)))
        
        # Extract characters
        characters = []
        for start, end in segmentation_points:
            char = word_img[:, start:end]
            # Add padding and resize for consistency
            char = cv2.copyMakeBorder(char, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
            char = cv2.resize(char, (32, 64))  # Standard size for all characters
            characters.append(char)
        
        return characters

def extract_words(image_path):
    """Improved word extraction with Arabic text considerations"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return []
    
    # Convert to grayscale and enhance contrast
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # Binarization optimized for Arabic text
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 10
    )
    
    # Find contours of words/lines
    contours, _ = cv2.findContours(
        binary, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Sort contours right-to-left (for Arabic)
    contours = sorted(contours, key=lambda c: -cv2.boundingRect(c)[0])
    
    # Extract word images
    words = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        word = binary[y:y+h, x:x+w]
        words.append(word)
    
    return words

# if __name__ == "__main__":
#     # Initialize with error handling
#     try:
#         segmenter = ArabicCharacterSegmenter()
#         os.makedirs("output", exist_ok=True)
        
#         # Process image (change path as needed)
#         image_path = "../tests/0.png"
#         if not os.path.exists(image_path):
#             raise FileNotFoundError(f"Image not found: {image_path}")
        
#         # Extract words
#         words = extract_words(image_path)
        
#         # Simplified output format
#         print("\nWord Segmentation Results:")
#         print("-------------------------")
        
#         # Process each word
#         for i, word in enumerate(words):
#             characters = segmenter.segment_word(word)
#             print(f"Word {i+1}: {len(characters)} characters")
            
#             # Save visualization with character numbers
#             vis = cv2.cvtColor(word, cv2.COLOR_GRAY2BGR)
#             x_offset = 0
            
#             for j, char in enumerate(characters):
#                 h, w = char.shape[:2]
#                 cv2.rectangle(vis, (x_offset, 0), (x_offset+w, h), (0, 0, 255), 1)
#                 cv2.putText(vis, str(j+1), (x_offset+2, 15), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
#                 x_offset += w + 2
                
#                 # Save character
#                 cv2.imwrite(f"output/word_{i+1}_char_{j+1}.png", char)
            
#             cv2.imwrite(f"output/word_{i+1}_visualization.png", vis)
        
#         print("\nCharacter images saved in 'output' directory")
        
#     except Exception as e:
#         print(f"Error: {e}")