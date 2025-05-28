# convert_calibration.py
import pickle
import numpy as np

def convert_depthsnap_to_argus(input_file, output_file):
    """
    Convert DepthSnap calibration format to ArgusTrack format
    """
    # Load DepthSnap calibration
    with open(input_file, 'rb') as f:
        depthsnap_calib = pickle.load(f)
    
    # Extract stereo parameters
    stereo = depthsnap_calib['stereo']
    
    # Convert to ArgusTrack format
    argus_calib = {
        'camera_matrix_left': stereo['left_camera_matrix'],
        'camera_matrix_right': stereo['right_camera_matrix'],
        'dist_coeffs_left': stereo['left_dist_coeffs'],
        'dist_coeffs_right': stereo['right_dist_coeffs'],
        'R': stereo['R'],
        'T': stereo['T'],
        'E': stereo['E'],
        'F': stereo['F'],
        'R1': stereo['R1'],
        'R2': stereo['R2'],
        'P1': stereo['P1'],
        'P2': stereo['P2'],
        'Q': stereo['Q'],
        'baseline': stereo['baseline'],
        'image_size': stereo['image_size'],
        'roi1': stereo['roi1'],
        'roi2': stereo['roi2'],
        'calibration_error': stereo['calibration_error']
    }
    
    # Save in ArgusTrack format
    with open(output_file, 'wb') as f:
        pickle.dump(argus_calib, f)
    
    print(f"Converted calibration saved to: {output_file}")
    print(f"Baseline: {argus_calib['baseline']:.2f}mm")
    print(f"Image size: {argus_calib['image_size']}")
    print(f"Calibration error: {argus_calib['calibration_error']:.6f}")

if __name__ == "__main__":
    convert_depthsnap_to_argus(
        '../DepthSnap/results/camera_calibration.pkl',
        'argus_calibration.pkl'
    )