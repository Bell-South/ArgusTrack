# argus_track/utils/gps_motion_predictor.py

"""
GPS Motion Prediction for Track Fragmentation Prevention
=======================================================

Calculates vehicle movement from GPS data and predicts where static objects
should appear in the next frame to prevent track fragmentation.

Key Components:
1. GPS Motion Calculator - Converts GPS lat/lon changes to vehicle movement
2. Screen Displacement Predictor - Converts vehicle movement to pixel displacement
3. Position Predictor - Predicts where objects should appear on screen

Camera Model: GoPro Hero 11 Wide Mode
- Horizontal FOV: 122°
- Resolution: 1920x1080
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core import GPSData


@dataclass
class MotionPredictionConfig:
    """Configuration for GPS motion prediction"""

    # === OBJECT DISTANCE ESTIMATION ===
    default_object_distance_m: float = 30.0  # Assume 30m distance for all objects

    # === GPS PROCESSING ===
    gps_smoothing_window: int = 3  # Moving average window for GPS smoothing
    min_gps_accuracy_m: float = 3.0  # Use prediction when GPS accuracy < 3m
    max_gps_accuracy_m: float = 5.0  # Fallback to existing logic when > 5m
    min_movement_threshold_m: float = 0.001  # Ignore GPS movements smaller than 10cm

    # === CAMERA PARAMETERS (GoPro Hero 11 Wide) ===
    horizontal_fov_degrees: float = 122.0  # Horizontal field of view
    vertical_fov_degrees: float = 69.0  # Vertical field of view
    image_width: int = 1920  # Screen resolution width
    image_height: int = 1080  # Screen resolution height

    # === PREDICTION PARAMETERS ===
    prediction_tolerance_px: float = 15.0  # Tight matching tolerance for predictions
    max_prediction_distance_px: float = 100.0  # Max pixel movement to consider valid
    rotation_enabled: bool = False  # Enable rotation prediction (start False)

    # === DEBUG SETTINGS ===
    enable_debug_logging: bool = True  # Log motion calculations
    enable_prediction_visualization: bool = True  # Show predictions in debug overlay


@dataclass
class VehicleMovement:
    """Represents calculated vehicle movement between two GPS points"""

    # Movement in meters (world coordinates)
    translation_x_m: float  # East-West movement (+ = East)
    translation_y_m: float  # North-South movement (+ = North)
    distance_moved_m: float  # Total distance moved

    # Rotation
    heading_change_deg: float  # Change in heading (+ = clockwise)

    # Metadata
    time_delta_s: float  # Time between GPS points
    speed_ms: float  # Calculated speed
    gps_accuracy_m: float  # Combined GPS accuracy
    is_valid: bool = True  # Whether movement is trustworthy


@dataclass
class ScreenDisplacement:
    """Predicted pixel displacement on screen"""

    displacement_x_px: float  # Horizontal pixel movement (+ = right)
    displacement_y_px: float  # Vertical pixel movement (+ = down)
    confidence: float  # Prediction confidence [0-1]
    method_used: str  # Which calculation method was used


class GPSMotionCalculator:
    """
    Calculates vehicle movement from GPS data using geodetic calculations
    """

    def __init__(self, config: MotionPredictionConfig):
        """Initialize GPS motion calculator"""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.GPSMotionCalculator")

        # GPS smoothing buffer
        self.gps_history: List[GPSData] = []

        # Earth radius for geodetic calculations (WGS84)
        self.EARTH_RADIUS_M = 6378137.0

        self.logger.info("GPS Motion Calculator initialized")
        if config.enable_debug_logging:
            self.logger.info(
                f"  Object distance assumption: {config.default_object_distance_m}m"
            )
            self.logger.info(
                f"  GPS accuracy thresholds: {config.min_gps_accuracy_m}m - {config.max_gps_accuracy_m}m"
            )

    def calculate_vehicle_movement(
        self, gps_prev: GPSData, gps_current: GPSData
    ) -> VehicleMovement:
        """
        Calculate vehicle movement between two GPS points

        Args:
            gps_prev: Previous GPS data point
            gps_current: Current GPS data point

        Returns:
            VehicleMovement object with calculated movement
        """
        # Add current GPS to smoothing buffer
        self._update_gps_history(gps_current)

        # Use smoothed GPS if available
        smoothed_prev = self._get_smoothed_gps(gps_prev)
        smoothed_current = self._get_smoothed_gps(gps_current)

        # Calculate time delta
        time_delta = smoothed_current.timestamp - smoothed_prev.timestamp

        if time_delta <= 0:
            # Estimate time delta based on frame interval (6 frames at 60fps = 0.1s)
            estimated_time_delta = 6.0 / 60.0  # 0.1 seconds
            self.logger.debug(
                f"Zero time delta detected, using estimated: {estimated_time_delta}s"
            )
            time_delta = estimated_time_delta

        # Continue with existing calculation logic...
        translation_x, translation_y = self._calculate_translation_meters(
            smoothed_prev, smoothed_current
        )

        # Calculate total distance and speed
        distance_moved = math.sqrt(translation_x**2 + translation_y**2)
        speed_ms = distance_moved / time_delta

        # Calculate heading change
        heading_change = self._calculate_heading_change(smoothed_prev, smoothed_current)

        # Determine combined GPS accuracy
        combined_accuracy = max(smoothed_prev.accuracy, smoothed_current.accuracy)

        # Validate movement
        is_valid = self._validate_movement(distance_moved, speed_ms, combined_accuracy)

        movement = VehicleMovement(
            translation_x_m=translation_x,
            translation_y_m=translation_y,
            distance_moved_m=distance_moved,
            heading_change_deg=heading_change,
            time_delta_s=time_delta,
            speed_ms=speed_ms,
            gps_accuracy_m=combined_accuracy,
            is_valid=is_valid,
        )

        if self.config.enable_debug_logging and is_valid:
            self.logger.debug(
                f"Vehicle movement: {distance_moved:.2f}m, "
                f"Speed: {speed_ms*3.6:.1f} km/h, "
                f"Heading change: {heading_change:.1f}°"
            )

        return movement

    def _calculate_translation_meters(
        self, gps_prev: GPSData, gps_current: GPSData
    ) -> Tuple[float, float]:
        """
        Calculate translation in meters using Haversine-based local coordinates

        Returns:
            (translation_x_m, translation_y_m) where:
            - x is East-West (+ = East)
            - y is North-South (+ = North)
        """
        # Convert latitude/longitude differences to meters
        lat1_rad = math.radians(gps_prev.latitude)
        lat2_rad = math.radians(gps_current.latitude)

        # Calculate differences
        dlat_rad = lat2_rad - lat1_rad
        dlon_rad = math.radians(gps_current.longitude - gps_prev.longitude)

        # Convert to meters using local approximation
        # (accurate for small distances < 1km)
        translation_y = dlat_rad * self.EARTH_RADIUS_M  # North-South
        translation_x = dlon_rad * self.EARTH_RADIUS_M * math.cos(lat1_rad)  # East-West

        return translation_x, translation_y

    def _calculate_heading_change(
        self, gps_prev: GPSData, gps_current: GPSData
    ) -> float:
        """Calculate change in vehicle heading"""
        heading_prev = gps_prev.heading
        heading_current = gps_current.heading

        # Handle heading wraparound (0° = 360°)
        heading_change = heading_current - heading_prev

        # Normalize to [-180, 180] range
        while heading_change > 180:
            heading_change -= 360
        while heading_change < -180:
            heading_change += 360

        return heading_change

    def _update_gps_history(self, gps_data: GPSData):
        """Update GPS history for smoothing"""
        self.gps_history.append(gps_data)

        # Keep only recent history
        max_history = self.config.gps_smoothing_window * 2
        if len(self.gps_history) > max_history:
            self.gps_history = self.gps_history[-max_history:]

    def _get_smoothed_gps(self, gps_data: GPSData) -> GPSData:
        """Get smoothed GPS data using moving average"""
        if len(self.gps_history) < self.config.gps_smoothing_window:
            return gps_data

        # Find recent GPS points around this timestamp
        target_time = gps_data.timestamp
        recent_points = []

        for hist_gps in self.gps_history[-self.config.gps_smoothing_window :]:
            time_diff = abs(hist_gps.timestamp - target_time)
            if time_diff < 1.0:  # Within 1 second
                recent_points.append(hist_gps)

        if len(recent_points) < 2:
            return gps_data

        # Calculate weighted average (more recent = higher weight)
        total_weight = 0
        weighted_lat = 0
        weighted_lon = 0
        weighted_heading = 0

        for i, point in enumerate(recent_points):
            weight = i + 1  # Linear weighting
            total_weight += weight
            weighted_lat += point.latitude * weight
            weighted_lon += point.longitude * weight
            weighted_heading += point.heading * weight

        # Create smoothed GPS data
        smoothed_gps = GPSData(
            timestamp=gps_data.timestamp,
            latitude=weighted_lat / total_weight,
            longitude=weighted_lon / total_weight,
            altitude=gps_data.altitude,
            heading=weighted_heading / total_weight,
            accuracy=gps_data.accuracy,
        )

        return smoothed_gps

    def _validate_movement(
        self, distance_moved: float, speed_ms: float, gps_accuracy: float
    ) -> bool:
        """Validate if calculated movement is trustworthy"""

        # Check GPS accuracy
        if gps_accuracy > self.config.max_gps_accuracy_m:
            if self.config.enable_debug_logging:
                self.logger.debug(
                    f"Movement invalid: GPS accuracy too low ({gps_accuracy:.1f}m)"
                )
            return False

        # Check minimum movement threshold
        if distance_moved < self.config.min_movement_threshold_m:
            if self.config.enable_debug_logging:
                self.logger.debug(f"Movement below threshold: {distance_moved:.3f}m")
            return False

        # Check reasonable speed (under 200 km/h = 55 m/s)
        if speed_ms > 55.0:
            if self.config.enable_debug_logging:
                self.logger.debug(
                    f"Movement invalid: Speed too high ({speed_ms*3.6:.1f} km/h)"
                )
            return False

        return True

    def _create_invalid_movement(self, reason: str) -> VehicleMovement:
        """Create invalid movement object with reason"""
        if self.config.enable_debug_logging:
            self.logger.debug(f"Invalid movement: {reason}")

        return VehicleMovement(
            translation_x_m=0.0,
            translation_y_m=0.0,
            distance_moved_m=0.0,
            heading_change_deg=0.0,
            time_delta_s=0.0,
            speed_ms=0.0,
            gps_accuracy_m=999.0,
            is_valid=False,
        )


class ScreenDisplacementPredictor:
    """
    Predicts pixel displacement on screen based on vehicle movement
    """

    def __init__(self, config: MotionPredictionConfig):
        """Initialize screen displacement predictor"""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ScreenDisplacementPredictor")

        # Calculate derived camera parameters
        self.horizontal_fov_rad = math.radians(config.horizontal_fov_degrees)
        self.vertical_fov_rad = math.radians(config.vertical_fov_degrees)

        # Pixels per radian
        self.pixels_per_rad_horizontal = config.image_width / self.horizontal_fov_rad
        self.pixels_per_rad_vertical = config.image_height / self.vertical_fov_rad

        self.logger.info("Screen Displacement Predictor initialized")
        if config.enable_debug_logging:
            self.logger.info(
                f"  Camera FOV: {config.horizontal_fov_degrees}° x {config.vertical_fov_degrees}°"
            )
            self.logger.info(
                f"  Pixels per radian: H={self.pixels_per_rad_horizontal:.1f}, V={self.pixels_per_rad_vertical:.1f}"
            )

    def predict_screen_displacement(
        self,
        vehicle_movement: VehicleMovement,
        object_distance_m: Optional[float] = None,
    ) -> ScreenDisplacement:
        """
        Predict pixel displacement based on vehicle movement

        Args:
            vehicle_movement: Calculated vehicle movement
            object_distance_m: Distance to object (uses default if None)

        Returns:
            ScreenDisplacement with predicted pixel movement
        """
        if not vehicle_movement.is_valid:
            return ScreenDisplacement(
                displacement_x_px=0.0,
                displacement_y_px=0.0,
                confidence=0.0,
                method_used="invalid_movement",
            )

        # Use default object distance if not specified
        if object_distance_m is None:
            object_distance_m = self.config.default_object_distance_m

        # Calculate angular displacement due to translation
        angular_displacement_x, angular_displacement_y = (
            self._calculate_angular_displacement(vehicle_movement, object_distance_m)
        )

        # Convert angular displacement to pixel displacement
        displacement_x_px = angular_displacement_x * self.pixels_per_rad_horizontal
        displacement_y_px = angular_displacement_y * self.pixels_per_rad_vertical

        # Calculate confidence based on GPS accuracy and movement size
        confidence = self._calculate_prediction_confidence(
            vehicle_movement, displacement_x_px, displacement_y_px
        )

        # Validate prediction
        if self._is_prediction_reasonable(displacement_x_px, displacement_y_px):
            method_used = "translation_only"
        else:
            displacement_x_px = 0.0
            displacement_y_px = 0.0
            confidence = 0.0
            method_used = "prediction_too_large"

        if self.config.enable_debug_logging and confidence > 0.5:
            self.logger.debug(
                f"Screen prediction: ({displacement_x_px:.1f}, {displacement_y_px:.1f})px, "
                f"confidence: {confidence:.2f}"
            )

        return ScreenDisplacement(
            displacement_x_px=displacement_x_px,
            displacement_y_px=displacement_y_px,
            confidence=confidence,
            method_used=method_used,
        )

    def _calculate_angular_displacement(
        self, vehicle_movement: VehicleMovement, object_distance_m: float
    ) -> Tuple[float, float]:
        """
        Calculate angular displacement of static objects due to vehicle movement

        Key insight: When vehicle moves, static objects appear to move in opposite direction
        """
        # Vehicle movement in world coordinates
        vehicle_x = vehicle_movement.translation_x_m  # East-West
        vehicle_y = vehicle_movement.translation_y_m  # North-South

        # For static objects, apparent movement is opposite to vehicle movement
        apparent_x = -vehicle_x  # Vehicle moves East → objects appear to move West
        apparent_y = -vehicle_y  # Vehicle moves North → objects appear to move South

        # Convert to angular displacement (small angle approximation)
        # Angular displacement = linear displacement / distance
        angular_x_rad = apparent_x / object_distance_m
        angular_y_rad = apparent_y / object_distance_m

        return angular_x_rad, angular_y_rad

    def _calculate_prediction_confidence(
        self,
        vehicle_movement: VehicleMovement,
        displacement_x_px: float,
        displacement_y_px: float,
    ) -> float:
        """Calculate confidence in the prediction"""

        # Base confidence from GPS accuracy
        accuracy_confidence = 1.0 - min(
            1.0, vehicle_movement.gps_accuracy_m / self.config.max_gps_accuracy_m
        )

        # Movement confidence (very small movements are less reliable)
        movement_confidence = min(
            1.0, vehicle_movement.distance_moved_m / 1.0
        )  # Confident above 1m movement

        # Pixel displacement confidence (very large displacements are suspicious)
        max_reasonable_px = self.config.max_prediction_distance_px
        pixel_displacement = math.sqrt(displacement_x_px**2 + displacement_y_px**2)
        displacement_confidence = max(0.0, 1.0 - pixel_displacement / max_reasonable_px)

        # Combined confidence
        combined_confidence = (
            accuracy_confidence * movement_confidence * displacement_confidence
        )

        return max(0.0, min(1.0, combined_confidence))

    def _is_prediction_reasonable(
        self, displacement_x_px: float, displacement_y_px: float
    ) -> bool:
        """Check if predicted displacement is reasonable"""

        # Check individual axis limits
        if abs(displacement_x_px) > self.config.max_prediction_distance_px:
            return False
        if abs(displacement_y_px) > self.config.max_prediction_distance_px:
            return False

        # Check total displacement
        total_displacement = math.sqrt(displacement_x_px**2 + displacement_y_px**2)
        if total_displacement > self.config.max_prediction_distance_px:
            return False

        return True


class MotionPredictor:
    """
    Main class that combines GPS motion calculation and screen displacement prediction
    """

    def __init__(self, config: MotionPredictionConfig):
        """Initialize motion predictor"""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.MotionPredictor")

        # Initialize components
        self.gps_calculator = GPSMotionCalculator(config)
        self.screen_predictor = ScreenDisplacementPredictor(config)

        # Statistics
        self.prediction_count = 0
        self.successful_predictions = 0
        self.total_prediction_error = 0.0

        self.logger.info("Motion Predictor initialized")

    def predict_object_positions(
        self,
        gps_prev: GPSData,
        gps_current: GPSData,
        current_track_positions: Dict[int, np.ndarray],
    ) -> Dict[int, Dict]:
        """
        Predict where existing tracked objects should appear in the current frame

        Args:
            gps_prev: Previous GPS data
            gps_current: Current GPS data
            current_track_positions: Dict of {track_id: last_known_position}

        Returns:
            Dict of {track_id: prediction_info} with predicted positions
        """
        predictions = {}

        # Calculate vehicle movement
        vehicle_movement = self.gps_calculator.calculate_vehicle_movement(
            gps_prev, gps_current
        )

        if not vehicle_movement.is_valid:
            # Return empty predictions - fall back to existing tracking
            return predictions

        # Calculate screen displacement
        screen_displacement = self.screen_predictor.predict_screen_displacement(
            vehicle_movement
        )

        if screen_displacement.confidence < 0.3:
            # Low confidence - don't use predictions
            return predictions

        # Apply displacement to each tracked object
        for track_id, last_position in current_track_positions.items():
            predicted_position = np.array(
                [
                    last_position[0] + screen_displacement.displacement_x_px,
                    last_position[1] + screen_displacement.displacement_y_px,
                ]
            )

            # Store prediction info
            predictions[track_id] = {
                "predicted_position": predicted_position,
                "last_position": last_position,
                "displacement": np.array(
                    [
                        screen_displacement.displacement_x_px,
                        screen_displacement.displacement_y_px,
                    ]
                ),
                "confidence": screen_displacement.confidence,
                "vehicle_movement": vehicle_movement,
                "screen_displacement": screen_displacement,
            }

            self.prediction_count += 1

        if self.config.enable_debug_logging and predictions:
            self.logger.debug(
                f"Generated {len(predictions)} position predictions with "
                f"confidence {screen_displacement.confidence:.2f}"
            )

        return predictions

    def get_prediction_statistics(self) -> Dict:
        """Get motion prediction statistics"""
        success_rate = (
            self.successful_predictions / max(1, self.prediction_count)
        ) * 100
        avg_error = self.total_prediction_error / max(1, self.successful_predictions)

        return {
            "total_predictions": self.prediction_count,
            "successful_predictions": self.successful_predictions,
            "success_rate_percent": success_rate,
            "average_prediction_error_px": avg_error,
            "gps_smoothing_window": self.config.gps_smoothing_window,
            "object_distance_m": self.config.default_object_distance_m,
        }

    def update_prediction_accuracy(self, prediction_error_px: float):
        """Update prediction accuracy statistics"""
        self.successful_predictions += 1
        self.total_prediction_error += prediction_error_px


def create_motion_prediction_config(
    object_distance_m: float = 30.0,
    gps_accuracy_threshold_m: float = 3.0,
    prediction_tolerance_px: float = 15.0,
    enable_debug: bool = True,
) -> MotionPredictionConfig:
    """
    Create motion prediction configuration with common parameters

    Args:
        object_distance_m: Assumed distance to objects in meters
        gps_accuracy_threshold_m: GPS accuracy threshold for using predictions
        prediction_tolerance_px: Tolerance for matching predictions to detections
        enable_debug: Enable debug logging and visualization

    Returns:
        MotionPredictionConfig object
    """
    return MotionPredictionConfig(
        default_object_distance_m=object_distance_m,
        min_gps_accuracy_m=gps_accuracy_threshold_m,
        max_gps_accuracy_m=gps_accuracy_threshold_m + 2.0,
        prediction_tolerance_px=prediction_tolerance_px,
        enable_debug_logging=enable_debug,
        enable_prediction_visualization=enable_debug,
    )
