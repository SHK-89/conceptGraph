import numpy as np

def backproject(depth, mask, K):
    """
    Converts 2D masked depth into 3D point cloud in camera coordinates.

    Parameters:
        depth (H, W)
        mask (H, W)
        K 3x3 intrinsics
    """
    ys, xs = np.where(mask)
    zs = depth[ys, xs]

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    xs3d = (xs - cx) * zs / fx
    ys3d = (ys - cy) * zs / fy
    pts = np.vstack((xs3d, ys3d, zs)).T

    return pts
