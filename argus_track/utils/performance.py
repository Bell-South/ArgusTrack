"""
Performance monitoring utilities for ArgusTrack
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    fps: float = 0.0
    frame_time: float = 0.0
    detection_time: float = 0.0
    tracking_time: float = 0.0
    total_time: float = 0.0
    frame_count: int = 0
    
    def reset(self):
        """Reset all metrics to zero"""
        self.fps = 0.0
        self.frame_time = 0.0
        self.detection_time = 0.0
        self.tracking_time = 0.0
        self.total_time = 0.0
        self.frame_count = 0

class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.start_time: Optional[float] = None
        self.frame_times: List[float] = []
        self.detection_times: List[float] = []
        self.tracking_times: List[float] = []
        
    def start_frame(self):
        """Start timing a frame"""
        self.start_time = time.time()
        
    def end_frame(self):
        """End timing a frame and update metrics"""
        if self.start_time is not None:
            frame_time = time.time() - self.start_time
            self.frame_times.append(frame_time)
            self.metrics.frame_count += 1
            self.start_time = None
            
    def record_detection_time(self, detection_time: float):
        """Record detection time for current frame"""
        self.detection_times.append(detection_time)
        
    def record_tracking_time(self, tracking_time: float):
        """Record tracking time for current frame"""
        self.tracking_times.append(tracking_time)
        
    def update_metrics(self):
        """Update average metrics"""
        if self.frame_times:
            self.metrics.frame_time = sum(self.frame_times) / len(self.frame_times)
            self.metrics.fps = 1.0 / self.metrics.frame_time if self.metrics.frame_time > 0 else 0.0
            
        if self.detection_times:
            self.metrics.detection_time = sum(self.detection_times) / len(self.detection_times)
            
        if self.tracking_times:
            self.metrics.tracking_time = sum(self.tracking_times) / len(self.tracking_times)
            
        self.metrics.total_time = self.metrics.detection_time + self.metrics.tracking_time
        
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        self.update_metrics()
        return self.metrics
        
    def reset(self):
        """Reset all timing data"""
        self.metrics.reset()
        self.frame_times.clear()
        self.detection_times.clear()
        self.tracking_times.clear()
        self.start_time = None
        
    def print_stats(self):
        """Print performance statistics"""
        self.update_metrics()
        print(f"Performance Stats:")
        print(f"  FPS: {self.metrics.fps:.2f}")
        print(f"  Frame Time: {self.metrics.frame_time*1000:.2f}ms")
        print(f"  Detection Time: {self.metrics.detection_time*1000:.2f}ms")
        print(f"  Tracking Time: {self.metrics.tracking_time*1000:.2f}ms")
        print(f"  Total Frames: {self.metrics.frame_count}")