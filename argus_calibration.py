#!/usr/bin/env python3
"""
Create a sample stereo calibration file for Argus Track
This creates argus_calibration.pkl with reasonable GoPro Hero 11 parameters
"""

import numpy as np
import pickle
from pathlib import Path

def create_sample_calibration():
    """Create sample stereo calibration for GoPro Hero 11"""
    
    # GoPro Hero 11 approximate parameters
    image_width = 2704
    image_height = 2028
    focal_length = 1400  # pixels (approximate)
    baseline = 0.12  # 12cm baseline (approximate for dual GoPro setup)
    
    # Camera intrinsic matrices (assumed identical cameras)
    cx = image_width / 2
    cy = image_height / 2
    
    camera_matrix = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Distortion coefficients (typical GoPro values)
    dist_coeffs = np.array([-0.3, 0.1, 0, 0, 0], dtype=np.float64)
    
    # Stereo parameters (assuming well-aligned cameras)
    R = np.eye(3, dtype=np.float64)  # No rotation between cameras
    T = np.array([[baseline], [0], [0]], dtype=np.float64)  # Horizontal translation
    
    # Essential and Fundamental matrices (computed from R, T, K)
    E = np.array([
        [0, 0, 0],
        [0, 0, -baseline],
        [0, baseline, 0]
    ], dtype=np.float64)
    
    # Simplified fundamental matrix (for demonstration)
    F = np.linalg.inv(camera_matrix.T) @ E @ np.linalg.inv(camera_matrix)
    
    # Create calibration dictionary
    calibration_data = {
        'camera_matrix_left': camera_matrix,
        'camera_matrix_right': camera_matrix,
        'dist_coeffs_left': dist_coeffs,
        'dist_coeffs_right': dist_coeffs,
        'R': R,
        'T': T,
        'E': E,
        'F': F,
        'baseline': baseline,
        'image_width': image_width,
        'image_height': image_height,
        # Optional matrices for rectification (can be computed later)
        'P1': None,
        'P2': None,
        'Q': None
    }
    
    return calibration_data

def main():
    """Create and save the calibration file"""
    print("Creating sample stereo calibration for Argus Track...")
    
    # Create calibration
    calibration = create_sample_calibration()
    
    # Save to file
    output_path = Path("argus_calibration.pkl")
    
    with open(output_path, 'wb') as f:
        pickle.dump(calibration, f)
    
    print(f"✅ Created calibration file: {output_path}")
    print(f"   Baseline: {calibration['baseline']}m")
    print(f"   Image size: {calibration['image_width']}x{calibration['image_height']}")
    print(f"   Focal length: {calibration['camera_matrix_left'][0,0]:.0f}px")
    
    # Verify the file can be loaded
    try:
        with open(output_path, 'rb') as f:
            loaded = pickle.load(f)
        print("✅ Calibration file verified and can be loaded")
        
        # Print summary
        print("\n📋 Calibration Summary:")
        print(f"   Left camera matrix shape: {loaded['camera_matrix_left'].shape}")
        print(f"   Right camera matrix shape: {loaded['camera_matrix_right'].shape}")
        print(f"   Rotation matrix shape: {loaded['R'].shape}")
        print(f"   Translation vector shape: {loaded['T'].shape}")
        print(f"   Baseline distance: {loaded['baseline']:.3f}m")
        
    except Exception as e:
        print(f"❌ Error verifying calibration file: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())