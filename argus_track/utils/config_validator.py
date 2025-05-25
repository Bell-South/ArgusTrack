"""Configuration validation utilities"""

from typing import Dict, List, Any, Optional
from dataclasses import fields
import yaml
import json
from pathlib import Path

from ..config import TrackerConfig, DetectorConfig, CameraConfig


class ConfigValidator:
    """Validate and sanitize configuration parameters"""
    
    @staticmethod
    def validate_tracker_config(config: TrackerConfig) -> List[str]:
        """
        Validate tracker configuration
        
        Args:
            config: TrackerConfig instance
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate thresholds
        if not 0 <= config.track_thresh <= 1:
            errors.append(f"track_thresh must be between 0 and 1, got {config.track_thresh}")
        
        if not 0 <= config.match_thresh <= 1:
            errors.append(f"match_thresh must be between 0 and 1, got {config.match_thresh}")
        
        # Validate buffer sizes
        if config.track_buffer < 1:
            errors.append(f"track_buffer must be at least 1, got {config.track_buffer}")
        
        if config.track_buffer > 300:
            errors.append(f"track_buffer is very large ({config.track_buffer}), this may cause memory issues")
        
        # Validate area threshold
        if config.min_box_area < 0:
            errors.append(f"min_box_area must be non-negative, got {config.min_box_area}")
        
        # Validate static detection parameters
        if config.static_threshold <= 0:
            errors.append(f"static_threshold must be positive, got {config.static_threshold}")
        
        if config.min_static_frames < 1:
            errors.append(f"min_static_frames must be at least 1, got {config.min_static_frames}")
        
        return errors
    
    @staticmethod
    def validate_detector_config(config: DetectorConfig) -> List[str]:
        """
        Validate detector configuration
        
        Args:
            config: DetectorConfig instance
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check paths exist
        if not Path(config.model_path).exists():
            errors.append(f"Model path does not exist: {config.model_path}")
        
        if not Path(config.config_path).exists():
            errors.append(f"Config path does not exist: {config.config_path}")
        
        # Validate thresholds
        if not 0 <= config.confidence_threshold <= 1:
            errors.append(f"confidence_threshold must be between 0 and 1, got {config.confidence_threshold}")
        
        if not 0 <= config.nms_threshold <= 1:
            errors.append(f"nms_threshold must be between 0 and 1, got {config.nms_threshold}")
        
        # Validate target classes
        if config.target_classes is not None and not config.target_classes:
            errors.append("target_classes is empty, no objects will be detected")
        
        return errors
    
    @staticmethod
    def validate_camera_config(config: CameraConfig) -> List[str]:
        """
        Validate camera configuration
        
        Args:
            config: CameraConfig instance
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate camera matrix
        if len(config.camera_matrix) != 3 or len(config.camera_matrix[0]) != 3:
            errors.append("camera_matrix must be a 3x3 matrix")
        
        # Validate distortion coefficients
        if len(config.distortion_coeffs) < 4:
            errors.append("distortion_coeffs must have at least 4 elements")
        
        # Validate image dimensions
        if config.image_width <= 0:
            errors.append(f"image_width must be positive, got {config.image_width}")
        
        if config.image_height <= 0:
            errors.append(f"image_height must be positive, got {config.image_height}")
        
        return errors
    
    @staticmethod
    def sanitize_config(config_dict: Dict[str, Any], 
                       config_class: type) -> Dict[str, Any]:
        """
        Sanitize configuration dictionary
        
        Args:
            config_dict: Raw configuration dictionary
            config_class: Target configuration class
            
        Returns:
            Sanitized configuration dictionary
        """
        # Get valid field names
        valid_fields = {f.name for f in fields(config_class)}
        
        # Filter out invalid fields
        sanitized = {
            k: v for k, v in config_dict.items() 
            if k in valid_fields
        }
        
        # Add missing fields with defaults
        for field in fields(config_class):
            if field.name not in sanitized and field.default is not None:
                sanitized[field.name] = field.default
        
        return sanitized
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], 
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
            
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = ConfigValidator.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged


class ConfigLoader:
    """Load and validate configuration from various sources"""
    
    @staticmethod
    def load_from_file(filepath: str) -> Dict[str, Any]:
        """
        Load configuration from file
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        if path.suffix in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {path.suffix}")
    
    @staticmethod
    def create_tracker_config(config_source: Optional[str] = None,
                            overrides: Optional[Dict[str, Any]] = None) -> TrackerConfig:
        """
        Create validated TrackerConfig
        
        Args:
            config_source: Path to configuration file
            overrides: Dictionary of override values
            
        Returns:
            Validated TrackerConfig instance
        """
        # Load base configuration
        if config_source:
            config_dict = ConfigLoader.load_from_file(config_source)
        else:
            config_dict = {}
        
        # Apply overrides
        if overrides:
            config_dict = ConfigValidator.merge_configs(config_dict, overrides)
        
        # Sanitize configuration
        config_dict = ConfigValidator.sanitize_config(config_dict, TrackerConfig)
        
        # Create config instance
        config = TrackerConfig(**config_dict)
        
        # Validate
        errors = ConfigValidator.validate_tracker_config(config)
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return config