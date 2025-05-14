import cv2
import numpy as np
import os
from farasa.segmenter import FarasaSegmenter
import pytesseract
from PIL import Image

class ArabicTextSegmenter:
    def __init__(self, use_farasa=True):
        self.use_farasa = use_farasa
        if use_farasa:
            try:
                self.segmenter = FarasaSegmenter(interactive=True)
                print("Farasa segmenter initialized successfully")
            except Exception as e:
                print(f"Farasa initialization warning: {e}")
                self.use_farasa = False
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
        
        # Configure Tesseract for Arabic
        self.tesseract_config = r'--oem 3 --psm 6 -l ara'

    def extract_text_from_image(self, image_path):
        """Extract Arabic text from image using Tesseract OCR"""
        try:
            # Preprocess image for better OCR results
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.equalizeHist(gray)
            
            # Use PIL for Tesseract compatibility
            pil_img = Image.fromarray(enhanced)
            text = pytesseract.image_to_string(pil_img, config=self.tesseract_config)
            
            # Clean up text
            text = text.strip()
            return text
        except Exception as e:
            print(f"Error in text extraction: {e}")
            return ""

    def segment_text(self, text):
        """Segment Arabic text into words and characters using Farasa"""
        if not text:
            return []
            
        if self.use_farasa:
            try:
                # Farasa provides morphological segmentation
                segmented = self.segmenter.segment(text)
                
                # Process Farasa output to get characters
                # Farasa marks prefixes/suffixes with + and stems with |
                # Example: "بالكتاب" becomes "ب+ال+كتاب"
                words = []
                for word in segmented.split():
                    # Split into morphemes (prefixes, stems, suffixes)
                    morphemes = [m for m in word.split('+') if m]
                    
                    # Further split each morpheme into individual characters
                    characters = []
                    for morpheme in morphemes:
                        characters.extend(list(morpheme))
                    
                    words.append({
                        'original': word,
                        'morphemes': morphemes,
                        'characters': characters
                    })
                return words
            except Exception as e:
                print(f"Farasa segmentation error: {e}")
        
        # Fallback: simple character splitting if Farasa fails
        return [{
            'original': word,
            'morphemes': [word],
            'characters': list(word)
        } for word in text.split()]

    def process_image(self, image_path):
        """Complete processing pipeline for an image"""
        # Step 1: Extract text from image
        extracted_text = self.extract_text_from_image(image_path)
        print(f"Extracted Text: {extracted_text}")
        
        # Step 2: Segment text using Farasa
        segmented_words = self.segment_text(extracted_text)
        
        # Step 3: Visualize and return results
        results = []
        for word in segmented_words:
            results.append({
                'word': word['original'],
                'character_count': len(word['characters']),
                'characters': word['characters'],
                'morphemes': word['morphemes']
            })
        
        return results

if __name__ == "__main__":
    try:
        segmenter = ArabicTextSegmenter()
        
        # Example usage
        image_path = "../tests/2.png"  # Replace with your image path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        results = segmenter.process_image(image_path)
        
        print("\nText Segmentation Results:")
        print("=" * 50)
        for i, word in enumerate(results):
            print(f"\nWord {i+1}: {word['word']}")
            print(f"Morphemes: {' | '.join(word['morphemes'])}")
            print(f"Characters ({word['character_count']}): {' '.join(word['characters'])}")
        
    except Exception as e:
        print(f"Error: {e}")