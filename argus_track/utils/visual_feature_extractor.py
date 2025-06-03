# argus_track/utils/visual_feature_extractor.py (NEW FILE)

"""
Visual Feature Extractor - Extract and compare visual features for object matching
Uses multiple feature descriptors to match objects based on appearance
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

from ..core import Detection


@dataclass
class VisualFeatures:
    """Container for visual features of an object"""
    histogram_hsv: np.ndarray          # HSV color histogram
    histogram_gray: np.ndarray         # Grayscale histogram  
    texture_features: np.ndarray       # LBP texture descriptor
    shape_features: np.ndarray         # Contour-based shape features
    size_features: np.ndarray          # Size and aspect ratio
    detection_bbox: np.ndarray         # Original bounding box
    extraction_quality: float         # Quality score (0-1)


class VisualFeatureExtractor:
    """
    Extract visual features from object detections for appearance-based matching
    
    Features extracted:
    1. Color histograms (HSV and grayscale)
    2. Texture features (Local Binary Patterns)
    3. Shape features (contours, aspect ratio)
    4. Size features (normalized)
    """
    
    def __init__(self, 
                 min_bbox_size: int = 20,
                 hist_bins: int = 32,
                 lbp_radius: int = 1,
                 lbp_points: int = 8):
        """
        Initialize visual feature extractor
        
        Args:
            min_bbox_size: Minimum bbox size for feature extraction
            hist_bins: Number of bins for histograms
            lbp_radius: Radius for LBP texture analysis
            lbp_points: Number of points for LBP
        """
        self.min_bbox_size = min_bbox_size
        self.hist_bins = hist_bins
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points
        self.logger = logging.getLogger(f"{__name__}.VisualFeatureExtractor")
        
        # Feature weights for similarity calculation
        self.feature_weights = {
            'color_hsv': 0.3,
            'color_gray': 0.2,
            'texture': 0.3,
            'shape': 0.1,
            'size': 0.1
        }
        
        self.logger.info("Visual Feature Extractor initialized")
        self.logger.info(f"  Histogram bins: {hist_bins}")
        self.logger.info(f"  LBP parameters: radius={lbp_radius}, points={lbp_points}")
    
    def extract_features(self, frame: np.ndarray, detection: Detection) -> Optional[VisualFeatures]:
        """
        Extract visual features from a detection
        
        Args:
            frame: Full frame image
            detection: Detection object with bounding box
            
        Returns:
            VisualFeatures object or None if extraction failed
        """
        try:
            bbox = detection.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Validate bounding box
            if not self._validate_bbox(bbox, frame.shape):
                return None
            
            # Extract region of interest
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0 or min(roi.shape[:2]) < self.min_bbox_size:
                return None
            
            # Extract different types of features
            hist_hsv = self._extract_hsv_histogram(roi)
            hist_gray = self._extract_gray_histogram(roi)
            texture_features = self._extract_texture_features(roi)
            shape_features = self._extract_shape_features(roi)
            size_features = self._extract_size_features(bbox)
            
            # Calculate extraction quality
            quality = self._calculate_extraction_quality(roi, bbox)
            
            features = VisualFeatures(
                histogram_hsv=hist_hsv,
                histogram_gray=hist_gray,
                texture_features=texture_features,
                shape_features=shape_features,
                size_features=size_features,
                detection_bbox=bbox,
                extraction_quality=quality
            )
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Feature extraction failed: {e}")
            return None
    
    def compare_features(self, features1: VisualFeatures, features2: VisualFeatures) -> float:
        """
        Compare two sets of visual features
        
        Args:
            features1: First set of features
            features2: Second set of features
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        try:
            similarities = {}
            
            # Color similarity (HSV)
            if features1.histogram_hsv is not None and features2.histogram_hsv is not None:
                similarities['color_hsv'] = self._compare_histograms(
                    features1.histogram_hsv, features2.histogram_hsv
                )
            
            # Color similarity (Grayscale)
            if features1.histogram_gray is not None and features2.histogram_gray is not None:
                similarities['color_gray'] = self._compare_histograms(
                    features1.histogram_gray, features2.histogram_gray
                )
            
            # Texture similarity
            if features1.texture_features is not None and features2.texture_features is not None:
                similarities['texture'] = self._compare_feature_vectors(
                    features1.texture_features, features2.texture_features
                )
            
            # Shape similarity
            if features1.shape_features is not None and features2.shape_features is not None:
                similarities['shape'] = self._compare_feature_vectors(
                    features1.shape_features, features2.shape_features
                )
            
            # Size similarity
            if features1.size_features is not None and features2.size_features is not None:
                similarities['size'] = self._compare_feature_vectors(
                    features1.size_features, features2.size_features
                )
            
            # Weighted average of similarities
            total_weight = 0
            weighted_sum = 0
            
            for feature_type, similarity in similarities.items():
                if similarity is not None:
                    weight = self.feature_weights.get(feature_type, 0.1)
                    weighted_sum += similarity * weight
                    total_weight += weight
            
            if total_weight > 0:
                final_similarity = weighted_sum / total_weight
                
                # Quality adjustment
                quality_factor = min(features1.extraction_quality, features2.extraction_quality)
                final_similarity *= quality_factor
                
                return min(1.0, max(0.0, final_similarity))
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Feature comparison failed: {e}")
            return 0.0
    
    def _extract_hsv_histogram(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """Extract HSV color histogram"""
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Calculate histograms for H, S, V channels
            hist_h = cv2.calcHist([hsv], [0], None, [self.hist_bins], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [self.hist_bins], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [self.hist_bins], [0, 256])
            
            # Normalize and concatenate
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()
            hist_v = cv2.normalize(hist_v, hist_v).flatten()
            
            return np.concatenate([hist_h, hist_s, hist_v])
            
        except Exception as e:
            self.logger.warning(f"HSV histogram extraction failed: {e}")
            return None
    
    def _extract_gray_histogram(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """Extract grayscale histogram"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [self.hist_bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist
            
        except Exception as e:
            self.logger.warning(f"Gray histogram extraction failed: {e}")
            return None
    
    def _extract_texture_features(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """Extract texture features using Local Binary Patterns"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Simple LBP implementation
            lbp = self._local_binary_pattern(gray, self.lbp_points, self.lbp_radius)
            
            # Calculate LBP histogram
            hist, _ = np.histogram(lbp.ravel(), bins=self.lbp_points + 2, 
                                 range=(0, self.lbp_points + 2), density=True)
            
            return hist
            
        except Exception as e:
            self.logger.warning(f"Texture feature extraction failed: {e}")
            return None
    
    def _local_binary_pattern(self, image: np.ndarray, points: int, radius: int) -> np.ndarray:
        """Simplified Local Binary Pattern implementation"""
        rows, cols = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = image[i, j]
                pattern = 0
                
                # Sample points around the center
                for p in range(points):
                    angle = 2 * np.pi * p / points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    
                    if 0 <= x < rows and 0 <= y < cols:
                        if image[x, y] > center:
                            pattern |= (1 << p)
                
                lbp[i, j] = pattern
        
        return lbp
    
    def _extract_shape_features(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """Extract shape-based features"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Find contours
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return np.array([0.0, 0.0, 0.0, 0.0])  # Default shape features
            
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate shape features
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 1.0
            
            # Extent (area / bounding box area)
            extent = area / (w * h) if w * h > 0 else 0.0
            
            # Solidity (area / convex hull area)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.0
            
            return np.array([aspect_ratio, extent, solidity, perimeter / area if area > 0 else 0.0])
            
        except Exception as e:
            self.logger.warning(f"Shape feature extraction failed: {e}")
            return np.array([0.0, 0.0, 0.0, 0.0])
    
    def _extract_size_features(self, bbox: np.ndarray) -> np.ndarray:
        """Extract normalized size features"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Normalize features (assuming typical image size of 1920x1080)
        normalized_width = width / 1920.0
        normalized_height = height / 1080.0
        normalized_area = area / (1920.0 * 1080.0)
        
        return np.array([normalized_width, normalized_height, normalized_area, aspect_ratio])
    
    def _calculate_extraction_quality(self, roi: np.ndarray, bbox: np.ndarray) -> float:
        """Calculate quality score for feature extraction"""
        try:
            # Size quality (larger regions are better)
            size = roi.shape[0] * roi.shape[1]
            size_quality = min(1.0, size / (self.min_bbox_size * self.min_bbox_size * 4))
            
            # Contrast quality (higher contrast is better)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            contrast = np.std(gray) / 255.0
            contrast_quality = min(1.0, contrast * 2)
            
            # Sharpness quality (edge strength)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            sharpness_quality = min(1.0, edge_density * 10)
            
            # Combined quality
            quality = (size_quality + contrast_quality + sharpness_quality) / 3.0
            return max(0.1, min(1.0, quality))  # Clamp between 0.1 and 1.0
            
        except Exception:
            return 0.5  # Default quality
    
    def _validate_bbox(self, bbox: np.ndarray, frame_shape: tuple) -> bool:
        """Validate bounding box dimensions"""
        x1, y1, x2, y2 = bbox
        height, width = frame_shape[:2]
        
        # Check bounds
        if x1 < 0 or y1 < 0 or x2 >= width or y2 >= height:
            return False
        
        # Check minimum size
        if (x2 - x1) < self.min_bbox_size or (y2 - y1) < self.min_bbox_size:
            return False
        
        return True
    
    def _compare_histograms(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Compare two histograms using correlation"""
        try:
            # Use correlation method (returns -1 to 1, we convert to 0 to 1)
            correlation = cv2.compareHist(hist1.astype(np.float32), 
                                        hist2.astype(np.float32), 
                                        cv2.HISTCMP_CORREL)
            return max(0.0, correlation)  # Convert to 0-1 range
            
        except Exception:
            return 0.0
    
    def _compare_feature_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compare two feature vectors using cosine similarity"""
        try:
            if len(vec1) != len(vec2):
                return 0.0
            
            vec1 = vec1.reshape(1, -1)
            vec2 = vec2.reshape(1, -1)
            
            # Cosine similarity
            similarity = cosine_similarity(vec1, vec2)[0, 0]
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception:
            return 0.0
    
    def get_feature_statistics(self) -> Dict[str, str]:
        """Get statistics about feature extraction"""
        return {
            'histogram_bins': self.hist_bins,
            'lbp_parameters': f"radius={self.lbp_radius}, points={self.lbp_points}",
            'min_bbox_size': self.min_bbox_size,
            'feature_weights': str(self.feature_weights)
        }