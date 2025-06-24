import cv2
import numpy as np
from typing import Tuple, List

class BasicImageProcessing:
    def __init__(self):
        """Initialize with default parameters"""
        # Gaussian Blur Parameters
        self.gaussian_kernel_size = (5, 5)  # Must be odd numbers
        self.gaussian_sigma = 1.5  # Standard deviation
        
        # Median Blur Parameters
        self.median_kernel_size = 5  # Must be odd number
        
        # Bilateral Filter Parameters
        self.bilateral_diameter = 9  # Diameter of pixel neighborhood
        self.bilateral_sigma_color = 75  # Filter sigma in color space
        self.bilateral_sigma_space = 75  # Filter sigma in coordinate space

    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur to reduce noise
        
        Args:
            image: Input image
            
        Returns:
            Blurred image
        """
        # Gaussian blur is used to reduce noise and smooth images
        # kernel_size: Size of Gaussian kernel (width, height)
        # sigma: Standard deviation in X direction
        blurred = cv2.GaussianBlur(image, self.gaussian_kernel_size, self.gaussian_sigma)
        return blurred

    def apply_median_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Apply median blur to remove salt and pepper noise
        
        Args:
            image: Input image
            
        Returns:
            Blurred image
        """
        # Median blur is effective for removing salt and pepper noise
        # kernel_size: Size of median kernel (must be odd)
        blurred = cv2.medianBlur(image, self.median_kernel_size)
        return blurred

    def apply_bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filter to preserve edges while removing noise
        
        Args:
            image: Input image
            
        Returns:
            Filtered image
        """
        # Bilateral filter preserves edges while removing noise
        # d: Diameter of pixel neighborhood
        # sigmaColor: Filter sigma in color space
        # sigmaSpace: Filter sigma in coordinate space
        filtered = cv2.bilateralFilter(
            image,
            self.bilateral_diameter,
            self.bilateral_sigma_color,
            self.bilateral_sigma_space
        )
        return filtered

    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert BGR image to grayscale
        
        Args:
            image: Input BGR image
            
        Returns:
            Grayscale image
        """
        # Convert BGR to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def apply_basic_threshold(self, image: np.ndarray, threshold: int = 127) -> np.ndarray:
        """
        Apply basic binary thresholding
        
        Args:
            image: Input grayscale image
            threshold: Threshold value (0-255)
            
        Returns:
            Binary image
        """
        # Basic thresholding converts grayscale to binary
        # threshold: Threshold value
        # maxval: Maximum value to assign
        # type: Type of thresholding
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return binary

    def apply_adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding
        
        Args:
            image: Input grayscale image
            
        Returns:
            Binary image
        """
        # Adaptive thresholding handles varying lighting conditions
        # maxValue: Maximum value to assign
        # adaptiveMethod: Method to calculate threshold
        # thresholdType: Type of thresholding
        # blockSize: Size of neighborhood
        # C: Constant subtracted from mean
        binary = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # blockSize
            2    # C
        )
        return binary

    def apply_morphological_operations(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply basic morphological operations
        
        Args:
            image: Input binary image
            
        Returns:
            Tuple of (eroded, dilated, opened) images
        """
        # Create kernel for morphological operations
        kernel = np.ones((3, 3), np.uint8)
        
        # Erosion: Removes small objects
        eroded = cv2.erode(image, kernel, iterations=1)
        
        # Dilation: Expands objects
        dilated = cv2.dilate(image, kernel, iterations=1)
        
        # Opening: Erosion followed by dilation
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        return eroded, dilated, opened

    def demonstrate_basic_processing(self, image_path: str):
        """
        Demonstrate basic image processing steps
        
        Args:
            image_path: Path to input image
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image")
        
        # Convert to grayscale
        gray = self.convert_to_grayscale(image)
        
        # Apply different blur techniques
        gaussian_blurred = self.apply_gaussian_blur(gray)
        median_blurred = self.apply_median_blur(gray)
        bilateral_filtered = self.apply_bilateral_filter(gray)
        
        # Apply thresholding
        basic_threshold = self.apply_basic_threshold(gray)
        adaptive_threshold = self.apply_adaptive_threshold(gray)
        
        # Apply morphological operations
        eroded, dilated, opened = self.apply_morphological_operations(basic_threshold)
        
        # Display results
        cv2.imshow("Original", image)
        cv2.imshow("Grayscale", gray)
        cv2.imshow("Gaussian Blur", gaussian_blurred)
        cv2.imshow("Median Blur", median_blurred)
        cv2.imshow("Bilateral Filter", bilateral_filtered)
        cv2.imshow("Basic Threshold", basic_threshold)
        cv2.imshow("Adaptive Threshold", adaptive_threshold)
        cv2.imshow("Eroded", eroded)
        cv2.imshow("Dilated", dilated)
        cv2.imshow("Opened", opened)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage
"""
if __name__ == "__main__":
    processor = BasicImageProcessing()
    processor.demonstrate_basic_processing("sample.jpg")
""" 