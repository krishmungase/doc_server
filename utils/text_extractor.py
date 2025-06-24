import cv2
import numpy as np
import easyocr
from PIL import Image
import re
from collections import Counter

class TextExtractor:
    def __init__(self):
        # Initialize EasyOCR reader once
        self.reader = easyocr.Reader(['en'])
        
    def preprocess_image(self, image):
        """Enhanced preprocessing for better text extraction"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter to remove noise while keeping edges sharp
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # Apply Otsu's thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply morphological operations
            kernel = np.ones((1, 1), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Apply dilation to make text more prominent
            kernel = np.ones((2, 2), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            
            return thresh
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return image

    def remove_duplicate_text(self, texts):
        """Remove duplicate or similar texts with improved deduplication"""
        if not texts:
            return []
            
        # Convert all texts to uppercase for comparison
        texts = [t.upper() for t in texts]
        
        # Split texts into words and remove duplicates
        unique_texts = []
        seen_words = set()
        
        for text in texts:
            # Split text into words
            words = text.split()
            # Remove duplicate words
            unique_words = []
            for word in words:
                if word not in seen_words:
                    seen_words.add(word)
                    unique_words.append(word)
            
            # Join unique words back into text
            unique_text = ' '.join(unique_words)
            if unique_text:
                unique_texts.append(unique_text)
        
        # Remove texts that are substrings of other texts
        filtered_texts = []
        for text in unique_texts:
            is_substring = False
            for other_text in unique_texts:
                if text != other_text and text in other_text:
                    is_substring = True
                    break
            if not is_substring:
                filtered_texts.append(text)
        
        # If we have multiple texts, try to merge them intelligently
        if len(filtered_texts) > 1:
            # Sort by length to process longer texts first
            filtered_texts.sort(key=len, reverse=True)
            
            # Try to merge texts that are similar
            merged_texts = []
            for text in filtered_texts:
                if not any(self.is_similar_text(text, merged) for merged in merged_texts):
                    merged_texts.append(text)
            
            return merged_texts
        
        return filtered_texts

    def is_similar_text(self, text1, text2, threshold=0.8):
        """Check if two texts are similar using word overlap"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return False
            
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union > threshold

    def clean_text(self, text, field_type=None):
        """Enhanced text cleaning and formatting based on field type"""
        try:
            # Remove special characters but keep alphanumeric, spaces, and common punctuation
            text = re.sub(r'[^a-zA-Z0-9\s/]', '', text)
            
            # Remove extra spaces
            text = ' '.join(text.split())
            
            # Field-specific cleaning
            if field_type:
                if field_type in ['First name', 'Last name', 'name']:
                    # Remove numbers from names
                    text = re.sub(r'\d', '', text)
                    # Remove common OCR errors in names
                    text = text.replace('0', 'O').replace('1', 'I')
                elif field_type in ['License number', 'license_number']:
                    # Keep only alphanumeric for license numbers
                    text = re.sub(r'[^A-Z0-9]', '', text)
                elif field_type in ['Address', 'address']:
                    # Keep address format
                    text = re.sub(r'\s+', ' ', text)
                elif field_type in ['DOB', 'dob', 'Issue date', 'issue_date', 'Exp date', 'expiry_date']:
                    # Format dates
                    text = re.sub(r'[^0-9/]', '', text)
            
            # Convert to uppercase for consistency
            text = text.upper()
            
            # Remove duplicate words
            words = text.split()
            unique_words = []
            seen_words = set()
            for word in words:
                if word not in seen_words:
                    seen_words.add(word)
                    unique_words.append(word)
            
            return ' '.join(unique_words)
            
        except Exception as e:
            print(f"Error in text cleaning: {str(e)}")
            return text

    def extract_text_from_region(self, image, x, y, width, height, field_type=None, confidence_threshold=0.5):
        """Extract text from a specific region with improved accuracy"""
        try:
            # Add padding to the region
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + width + padding)
            y2 = min(image.shape[0], y + height + padding)
            
            # Extract region of interest
            roi = image[y1:y2, x1:x2]
            
            # Skip if ROI is too small
            if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                return ""
            
            # Try different preprocessing methods
            results = []
            
            # Original image
            results.extend(self.reader.readtext(roi))
            
            # Processed image
            processed_roi = self.preprocess_image(roi)
            results.extend(self.reader.readtext(processed_roi))
            
            # Inverted image
            inverted = cv2.bitwise_not(processed_roi)
            results.extend(self.reader.readtext(inverted))
            
            # Filter results by confidence and get unique texts
            filtered_texts = [text for _, text, conf in results if conf > confidence_threshold]
            
            if filtered_texts:
                # Clean and format text based on field type
                cleaned_texts = [self.clean_text(text, field_type) for text in filtered_texts]
                
                # Remove duplicates while preserving order
                seen = set()
                unique_texts = []
                for text in cleaned_texts:
                    if text not in seen:
                        seen.add(text)
                        unique_texts.append(text)
                
                # For 'details' field, join all unique lines with '\n'
                if field_type and field_type.lower() == 'details':
                    return '\n'.join(unique_texts)
                # For certain fields, use the longest text as it's likely to be most complete
                if field_type in ['Address', 'address', 'name', 'First name', 'Last name']:
                    final_text = max(unique_texts, key=len)
                else:
                    # For other fields, use the first text as it's likely to be most accurate
                    final_text = unique_texts[0]
                
                return final_text
            
            return ""
            
        except Exception as e:
            print(f"Error in text extraction: {str(e)}")
            return ""

    def extract_document_text(self, image, predictions):
        """Extract text from all predicted regions in the document"""
        extracted_text = {}
        
        try:
            for prediction in predictions:
                class_name = prediction["class"]
                x = int(prediction["x"] - prediction["width"]/2)
                y = int(prediction["y"] - prediction["height"]/2)
                width = int(prediction["width"])
                height = int(prediction["height"])
                confidence = prediction["confidence"]
                
                print(f"\nProcessing {class_name} region:")
                print(f"Coordinates: x={x}, y={y}, width={width}, height={height}")
                print(f"Confidence: {confidence}")
                
                # Extract text from the region
                text = self.extract_text_from_region(
                    image, x, y, width, height,
                    field_type=class_name,
                    confidence_threshold=0.5
                )
                
                if text:
                    extracted_text[class_name] = {
                        "text": text,
                        "confidence": confidence
                    }
                    
                    print(f"Extracted text: {text}")
                else:
                    print(f"No text extracted from {class_name}")
            
            return extracted_text
            
        except Exception as e:
            print(f"Error in document text extraction: {str(e)}")
            return extracted_text 