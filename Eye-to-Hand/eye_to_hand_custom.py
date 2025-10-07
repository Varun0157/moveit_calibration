import numpy as np
import yaml
import cv2
from scipy.spatial.transform import Rotation

"""
Reference : https://forum.opencv.org/t/eye-to-hand-calibration/5690/9 
"""

def matrix_to_transformation(matrix_flat):
    """Convert flat 16-element list to 4x4 transformation matrix."""
    return np.array(matrix_flat).reshape(4, 4)

def load_calibration_samples(yaml_file):
    """Load T_base2gripper and T_cam2target samples from a YAML file."""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    base2gripper_transforms = []
    cam2target_transforms = []
    
    for sample in data:
        b2g_matrix = matrix_to_transformation(sample['effector_wrt_world'])
        base2gripper_transforms.append(b2g_matrix)
        
        c2t_matrix = matrix_to_transformation(sample['object_wrt_sensor'])
        cam2target_transforms.append(c2t_matrix)
    
    return base2gripper_transforms, cam2target_transforms

def check_sample_diversity(base2gripper_transforms):
    """Check if robot poses have sufficient diversity."""
    positions = np.array([T[:3, 3] for T in base2gripper_transforms])
    pos_std = np.std(positions, axis=0)
    print(f"\nPosition standard deviations (x,y,z): {pos_std}")
    if np.all(pos_std < 0.001):
        print("WARNING: All gripper positions are nearly identical.")
        return False
        
    quats = np.array([Rotation.from_matrix(T[:3, :3]).as_quat() for T in base2gripper_transforms])
    quat_std = np.std(quats, axis=0)
    print(f"Orientation (quaternion) standard deviations: {quat_std}")
    if np.all(quat_std < 0.01):
         print("WARNING: All gripper orientations are very similar.")
    return True

def calibrate_eye_hand(R_base2gripper, t_base2gripper, R_cam2target, t_cam2target, eye_to_hand=True):
    """
    A corrected wrapper for cv2.calibrateHandEye to handle both scenarios.
    For eye-to-hand, it correctly inverts the camera-to-target poses.
    """
    if eye_to_hand:
        R_target2cam, t_target2cam = [], []
        for R, t in zip(R_cam2target, t_cam2target):
            R_inv = R.T
            t_inv = -R_inv @ t
            R_target2cam.append(R_inv)
            t_target2cam.append(t_inv)
    else:
        R_target2cam = R_cam2target
        t_target2cam = t_cam2target

    R, t = cv2.calibrateHandEye(
        R_gripper2base=R_base2gripper,
        t_gripper2base=t_base2gripper,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=cv2.CALIB_HAND_EYE_DANIILIDIS
    )
    return R, t

def compute_reprojection_error(base2gripper_transforms, cam2target_transforms, T_base2cam, T_gripper2target):
    """
    Calculates the reprojection error for the eye-to-hand calibration.

    This function checks the consistency of the transformation loop for each sample:
    T_base2cam * T_cam2target should be equal to T_base2gripper * T_gripper2target.
    """
    translation_errors = []
    rotation_errors = []

    print("\n" + "="*60)
    print("COMPUTING REPROJECTION ERROR")
    print("="*60)

    for i in range(len(base2gripper_transforms)):
        T_b2g = base2gripper_transforms[i]
        T_c2t = cam2target_transforms[i]

        # Pose of the target relative to the robot base, via the CAMERA path
        T_b2t_via_cam = T_base2cam @ T_c2t
        
        # Pose of the target relative to the robot base, via the GRIPPER path
        T_b2t_via_gripper = T_b2g @ T_gripper2target

        # --- Calculate Translation Error ---
        trans_cam = T_b2t_via_cam[:3, 3]
        trans_gripper = T_b2t_via_gripper[:3, 3]
        t_error = np.linalg.norm(trans_cam - trans_gripper)
        translation_errors.append(t_error)

        # --- Calculate Rotation Error ---
        R_cam = T_b2t_via_cam[:3, :3]
        R_gripper = T_b2t_via_gripper[:3, :3]
        
        # The rotation error is the angle of the relative rotation between the two matrices
        R_relative = R_cam @ R_gripper.T
        r_err_rad = np.linalg.norm(Rotation.from_matrix(R_relative).as_rotvec())
        rotation_errors.append(np.rad2deg(r_err_rad))
        
        print(f"Sample {i+1:2d}: Translation Error = {t_error*1000:6.3f} mm, Rotation Error = {np.rad2deg(r_err_rad):6.3f} degrees")

    rms_t_error_mm = np.sqrt(np.mean(np.square(translation_errors))) * 1000
    rms_r_error_deg = np.sqrt(np.mean(np.square(rotation_errors)))
    
    print("-" * 60)
    print(f"RMS Translation Error: {rms_t_error_mm:.4f} mm")
    print(f"RMS Rotation Error:    {rms_r_error_deg:.4f} degrees")
    print("="*60)
    
    return rms_t_error_mm, rms_r_error_deg

if __name__ == "__main__":
    yaml_file = "best_samples.yaml"
    
    print("Loading calibration samples for EYE-TO-HAND...")
    base2gripper_transforms, cam2target_transforms = load_calibration_samples(yaml_file)
    print(f"Loaded {len(base2gripper_transforms)} samples.")
    
    if not check_sample_diversity(base2gripper_transforms):
        print("\nCannot perform calibration due to insufficient pose diversity.")
    else:
        # 1. Decompose the 4x4 matrices into lists of R and t
        R_base2gripper = [T[:3, :3] for T in base2gripper_transforms]
        t_base2gripper = [T[:3, 3].reshape(3, 1) for T in base2gripper_transforms]
        
        R_cam2target = [T[:3, :3] for T in cam2target_transforms]
        t_cam2target = [T[:3, 3].reshape(3, 1) for T in cam2target_transforms]
        
        print("\n" + "="*60)
        print("SOLVING EYE-TO-HAND CALIBRATION (AX = XB)")
        print("="*60)
        
        R_g2t, t_g2t = calibrate_eye_hand(
            R_base2gripper, t_base2gripper, R_cam2target, t_cam2target, eye_to_hand=True
        )
        
        T_gripper2target = np.eye(4)
        T_gripper2target[:3, :3] = R_g2t
        T_gripper2target[:3, 3] = t_g2t.flatten()
        
        print("\nIntermediate Result: Gripper to Target Transformation (X):")
        print(np.round(T_gripper2target, 4))
        
        T_b2g_sample = base2gripper_transforms[0]
        T_c2t_sample_inv = np.linalg.inv(cam2target_transforms[0])
        T_base2cam = T_b2g_sample @ T_gripper2target @ T_c2t_sample_inv
        
        print("\n" + "#"*60)
        print("FINAL RESULT: Base to Camera Transformation")
        print("#"*60)
        print(np.round(T_base2cam, 4))
        print("\nTranslation (x, y, z) in meters:", np.round(T_base2cam[:3, 3], 4))
        
        compute_reprojection_error(
            base2gripper_transforms,
            cam2target_transforms,
            T_base2cam,
            T_gripper2target
        )