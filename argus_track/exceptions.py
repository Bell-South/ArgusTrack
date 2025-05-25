"""Custom exceptions for Argus Track"""


class ArgusTrackError(Exception):
    """Base exception for Argus Track"""
    pass


class DetectorError(ArgusTrackError):
    """Raised when detector operations fail"""
    pass


class TrackerError(ArgusTrackError):
    """Raised when tracker operations fail"""
    pass


class ConfigurationError(ArgusTrackError):
    """Raised when configuration is invalid"""
    pass


class GPSError(ArgusTrackError):
    """Raised when GPS operations fail"""
    pass


class VideoError(ArgusTrackError):
    """Raised when video processing fails"""
    pass