"""
Image Preprocessor for TrustExtract-N
======================================
Dedicated image preprocessing pipeline for scanned government documents.
Handles deskewing, denoising, contrast enhancement, adaptive thresholding,
and orientation detection while preserving coordinate mappings.
"""

import math
from typing import Tuple, Optional, Dict, Any
from PIL import Image
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class ImagePreprocessor:
    """
    Preprocesses scanned document images for optimal OCR performance.
    Handles common degradation patterns in Indian government notifications:
    - Skewed/rotated scans
    - Low contrast / faded text
    - Noise and artifacts (stamps, signatures)
    - Uneven illumination
    """

    def __init__(
        self,
        deskew: bool = True,
        denoise: bool = True,
        enhance_contrast: bool = True,
        adaptive_threshold: bool = True,
        target_dpi: int = 250,
    ):
        self.deskew = deskew
        self.denoise = denoise
        self.enhance_contrast = enhance_contrast
        self.adaptive_threshold = adaptive_threshold
        self.target_dpi = target_dpi

    def preprocess(self, pil_image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Full preprocessing pipeline for a scanned page image.

        Args:
            pil_image: PIL Image of the scanned page.

        Returns:
            Tuple of (preprocessed PIL Image, metadata dict with transform info).
        """
        metadata = {
            "original_size": pil_image.size,
            "deskew_angle": 0.0,
            "operations_applied": [],
        }

        if not CV2_AVAILABLE:
            # Minimal fallback: convert to grayscale
            gray_img = pil_image.convert("L")
            metadata["operations_applied"].append("grayscale_only")
            return gray_img, metadata

        # Convert PIL → OpenCV
        cv_img = np.array(pil_image.convert("RGB"))
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        # 1. Convert to grayscale
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        metadata["operations_applied"].append("grayscale")

        # 2. Deskew
        if self.deskew:
            gray, angle = self._deskew_image(gray)
            metadata["deskew_angle"] = angle
            if abs(angle) > 0.1:
                metadata["operations_applied"].append(f"deskew({angle:.2f}°)")

        # 3. Denoise
        if self.denoise:
            gray = self._denoise_image(gray)
            metadata["operations_applied"].append("denoise")

        # 4. Contrast enhancement (CLAHE)
        if self.enhance_contrast:
            gray = self._enhance_contrast(gray)
            metadata["operations_applied"].append("clahe_contrast")

        # 5. Adaptive thresholding
        if self.adaptive_threshold:
            gray = self._adaptive_threshold(gray)
            metadata["operations_applied"].append("adaptive_threshold")

        metadata["processed_size"] = (gray.shape[1], gray.shape[0])

        # Convert back to PIL
        result_img = Image.fromarray(gray)
        return result_img, metadata

    def preprocess_for_ocr(self, pil_image: Image.Image) -> Image.Image:
        """
        Simplified preprocessing that returns only the processed image.
        Convenience method for OCR engine integration.
        """
        processed_img, _ = self.preprocess(pil_image)
        return processed_img

    @staticmethod
    def _deskew_image(gray: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Deskews a grayscale image using Hough line transform.
        Returns (deskewed image, detected angle in degrees).
        """
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using probabilistic Hough transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=100,
            minLineLength=gray.shape[1] // 4,
            maxLineGap=20
        )

        if lines is None or len(lines) == 0:
            return gray, 0.0

        # Calculate angles of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            # Only consider near-horizontal lines (within 15 degrees)
            if abs(angle) < 15:
                angles.append(angle)

        if not angles:
            return gray, 0.0

        # Use median angle to avoid outliers
        median_angle = float(np.median(angles))

        # Only deskew if angle is significant but not too large
        if abs(median_angle) < 0.3 or abs(median_angle) > 10:
            return gray, 0.0

        # Rotate image
        h, w = gray.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            gray, rotation_matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated, median_angle

    @staticmethod
    def _denoise_image(gray: np.ndarray) -> np.ndarray:
        """
        Denoises using non-local means denoising.
        Parameters tuned for typical scanned document noise.
        """
        # fastNlMeansDenoising params: h=10 (filter strength), templateWindowSize=7, searchWindowSize=21
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        return denoised

    @staticmethod
    def _enhance_contrast(gray: np.ndarray) -> np.ndarray:
        """
        Enhances contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Effective for documents with uneven illumination or faded text.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

    @staticmethod
    def _adaptive_threshold(gray: np.ndarray) -> np.ndarray:
        """
        Applies Gaussian adaptive thresholding.
        Better than global thresholding for documents with varying
        background brightness (letterheads, stamps, etc.).
        """
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=8
        )
        return binary

    @staticmethod
    def detect_orientation(pil_image: Image.Image) -> int:
        """
        Detects page orientation (0, 90, 180, 270 degrees).
        Uses text line density heuristic: correct orientation
        has maximum horizontal text density.

        Returns rotation angle needed to correct orientation.
        """
        if not CV2_AVAILABLE:
            return 0

        gray = np.array(pil_image.convert("L"))
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Calculate horizontal and vertical projection profiles
        h_proj = np.sum(binary, axis=1)
        v_proj = np.sum(binary, axis=0)

        # In correct orientation, horizontal projection has more variance
        # (text lines create clear peaks)
        h_variance = np.var(h_proj)
        v_variance = np.var(v_proj)

        if h_variance > v_variance * 1.5:
            return 0  # Already correct
        elif v_variance > h_variance * 1.5:
            return 90  # Rotated 90 degrees
        else:
            return 0  # Ambiguous, assume correct
