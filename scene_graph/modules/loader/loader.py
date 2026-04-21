import cv2
import numpy as np

def load_rgbd_frame(rgb_path, depth_path):
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, -1).astype(float) / 1000.0  # scale to meters
    return rgb, depth


def load_intrinsics(K_path):
    return np.loadtxt(K_path)


def load_pose(pose_path):
    return np.loadtxt(pose_path)
