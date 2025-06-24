import cv2
import numpy as np
from scipy import ndimage
from typing import Tuple, Optional

class ImagePreprocessor:
    def __init__(self):
        """Initialize the image preprocessor with default parameters"""
        self.kernel_size = (5, 5)
        self.sigma = 1.5
        self.threshold_value = 127
        self.max_value = 255

    def apply_gaussian_blur(self, image: np.ndarray, kernel_size: Tuple[int, int] = None, 
                          sigma: float = None) -> np.ndarray:
        """
        Apply Gaussian blur to reduce noise in the image
        
        Args:
            image: Input image
            kernel_size: Size of Gaussian kernel (width, height)
            sigma: Standard deviation for Gaussian kernel
            
        Returns:
            Blurred image
        """
        if kernel_size is None:
            kernel_size = self.kernel_size
        if sigma is None:
            sigma = self.sigma
            
        return cv2.GaussianBlur(image, kernel_size, sigma)

    def apply_adaptive_threshold(self, image: np.ndarray, 
                               block_size: int = 11, 
                               c: int = 2) -> np.ndarray:
        """
        Apply adaptive thresholding to handle varying lighting conditions
        
        Args:
            image: Input image
            block_size: Size of pixel neighborhood
            c: Constant subtracted from mean
            
        Returns:
            Binary image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, block_size, c)

    def apply_otsu_threshold(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply Otsu's thresholding method
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (binary image, threshold value)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary, _
   
    def correct_skew(self, image: np.ndarray, 
                    delta: float = 1, 
                    limit: float = 5) -> Tuple[np.ndarray, float]:
        """
        Correct image skew using Hough transform
        
        Args:
            image: Input image
            delta: Step size for angle detection
            limit: Maximum angle to detect
            
        Returns:
            Tuple of (corrected image, detected angle)
        """
        # Convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply threshold to get binary image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Find all non-zero points
        coords = np.column_stack(np.where(thresh > 0))
        
        # Find minimum rotated rectangle
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle
        if angle < -45:
            angle = 90 + angle
            
        # Rotate image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), 
                                flags=cv2.INTER_CUBIC, 
                                borderMode=cv2.BORDER_REPLICATE)
        
        return rotated, angle

    def enhance_contrast(self, image: np.ndarray, 
                        clip_limit: float = 2.0, 
                        tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image: Input image
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
            
        Returns:
            Contrast enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        
        # Merge channels
        merged = cv2.merge((cl, a, b))
        
        # Convert back to BGR
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def remove_noise(self, image: np.ndarray, 
                    kernel_size: int = 3, 
                    iterations: int = 1) -> np.ndarray:
        """
        Remove noise using morphological operations
        
        Args:
            image: Input image
            kernel_size: Size of morphological kernel
            iterations: Number of iterations
            
        Returns:
            Denoised image
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        return closing

    def preprocess_pipeline(self, image: np.ndarray, 
                          apply_blur: bool = True,
                          apply_threshold: bool = True,
                          correct_skew: bool = True,
                          enhance_contrast: bool = True,
                          remove_noise: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            image: Input image
            apply_blur: Whether to apply Gaussian blur
            apply_threshold: Whether to apply thresholding
            correct_skew: Whether to correct skew
            enhance_contrast: Whether to enhance contrast
            remove_noise: Whether to remove noise
            
        Returns:
            Preprocessed image
        """
        processed = image.copy()
        
        if apply_blur:
            processed = self.apply_gaussian_blur(processed)
            
        if enhance_contrast:
            processed = self.enhance_contrast(processed)
            
        if correct_skew:
            processed, _ = self.correct_skew(processed)
            
        if apply_threshold:
            processed = self.apply_adaptive_threshold(processed)
            
        if remove_noise:
            processed = self.remove_noise(processed)
            
        return processed

# Example usage (commented out as requested)
"""
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = ImagePreprocessor()
    
    # Load image
    image = cv2.imread("sample.jpg")
    
    # Apply preprocessing pipeline
    processed = preprocessor.preprocess_pipeline(
        image,
        apply_blur=True,
        apply_threshold=True,
        correct_skew=True,
        enhance_contrast=True,
        remove_noise=True
    )
    
    # Display results
    cv2.imshow("Original", image)
    cv2.imshow("Processed", processed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
""" 