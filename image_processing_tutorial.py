import cv2
import numpy as np
from typing import Tuple, List

class ImageProcessingTutorial:
    def __init__(self):
        """Initialize with default parameters"""
        # Threshold values
        self.binary_threshold = 127
        self.max_value = 255
        
        # Gaussian blur parameters
        self.kernel_size = (5, 5)
        self.sigma = 1.5
        
        # Adaptive threshold parameters
        self.block_size = 11
        self.c = 2
        
        # Morphological operation parameters
        self.morph_kernel_size = 3
        self.iterations = 1

    def basic_thresholding(self, image: np.ndarray) -> np.ndarray:
        """
        Basic thresholding demonstration
        
        Args:
            image: Input image (grayscale)
            
        Returns:
            Binary image
        """
        # Simple binary thresholding
        _, binary = cv2.threshold(image, self.binary_threshold, self.max_value, 
                                cv2.THRESH_BINARY)
        
        # Binary inverse thresholding
        _, binary_inv = cv2.threshold(image, self.binary_threshold, self.max_value, 
                                    cv2.THRESH_BINARY_INV)
        
        # Truncate thresholding
        _, trunc = cv2.threshold(image, self.binary_threshold, self.max_value, 
                               cv2.THRESH_TRUNC)
        
        # To zero thresholding
        _, tozero = cv2.threshold(image, self.binary_threshold, self.max_value, 
                                cv2.THRESH_TOZERO)
        
        return binary

    def adaptive_thresholding(self, image: np.ndarray) -> np.ndarray:
        """
        Adaptive thresholding demonstration
        
        Args:
            image: Input image (grayscale)
            
        Returns:
            Binary image
        """
        # Mean adaptive thresholding
        mean_thresh = cv2.adaptiveThreshold(
            image, self.max_value,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            self.block_size,
            self.c
        )
        
        # Gaussian adaptive thresholding
        gaussian_thresh = cv2.adaptiveThreshold(
            image, self.max_value,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.block_size,
            self.c
        )
        
        return gaussian_thresh

    def otsu_thresholding(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Otsu's thresholding demonstration
        
        Args:
            image: Input image (grayscale)
            
        Returns:
            Tuple of (binary image, threshold value)
        """
        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Apply Otsu's thresholding
        threshold_value, binary = cv2.threshold(
            blur, 0, self.max_value,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        return binary, threshold_value

    def morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """
        Morphological operations demonstration
        
        Args:
            image: Input binary image
            
        Returns:
            Processed image
        """
        # Create kernel
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        
        # Erosion
        erosion = cv2.erode(image, kernel, iterations=self.iterations)
        
        # Dilation
        dilation = cv2.dilate(image, kernel, iterations=self.iterations)
        
        # Opening (erosion followed by dilation)
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Closing (dilation followed by erosion)
        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        # Gradient (difference between dilation and erosion)
        gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        
        return closing

    def noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """
        Noise reduction techniques demonstration
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        # Gaussian blur
        gaussian = cv2.GaussianBlur(image, self.kernel_size, self.sigma)
        
        # Median blur (good for salt and pepper noise)
        median = cv2.medianBlur(image, self.morph_kernel_size)
        
        # Bilateral filter (preserves edges)
        bilateral = cv2.bilateralFilter(image, 9, 75, 75)
        
        return bilateral

    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Edge detection demonstration
        
        Args:
            image: Input image
            
        Returns:
            Edge detected image
        """
        # Convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Sobel edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        
        # Canny edges
        canny = cv2.Canny(gray, 100, 200)
        
        # Laplacian edges
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        return canny

    def color_space_conversion(self, image: np.ndarray) -> np.ndarray:
        """
        Color space conversion demonstration
        
        Args:
            image: Input BGR image
            
        Returns:
            Processed image
        """
        # BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # BGR to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # BGR to Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return hsv

    def image_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Image enhancement techniques demonstration
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels
        merged = cv2.merge((cl, a, b))
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
        return enhanced

    def demonstrate_all(self, image_path: str):
        """
        Demonstrate all processing techniques on an image
        
        Args:
            image_path: Path to input image
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image")
            
        # Convert to grayscale for some operations
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply all techniques
        binary = self.basic_thresholding(gray)
        adaptive = self.adaptive_thresholding(gray)
        otsu, _ = self.otsu_thresholding(gray)
        morph = self.morphological_operations(binary)
        denoised = self.noise_reduction(image)
        edges = self.edge_detection(image)
        color_converted = self.color_space_conversion(image)
        enhanced = self.image_enhancement(image)
        
        # Display results
        cv2.imshow("Original", image)
        cv2.imshow("Binary", binary)
        cv2.imshow("Adaptive", adaptive)
        cv2.imshow("Otsu", otsu)
        cv2.imshow("Morphological", morph)
        cv2.imshow("Denoised", denoised)
        cv2.imshow("Edges", edges)
        cv2.imshow("Color Converted", color_converted)
        cv2.imshow("Enhanced", enhanced)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage
"""
if __name__ == "__main__":
    tutorial = ImageProcessingTutorial()
    tutorial.demonstrate_all("sample.jpg")
""" 