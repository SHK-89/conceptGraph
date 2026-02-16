import numpy as np

def geometric_sim(A, B, thresh=0.05):
    if len(A)==0 or len(B)==0:
        return 0
    dists = np.linalg.norm(A[:,None,:] - B[None,:,:], axis=2)
    return (dists.min(axis=1) < thresh).mean()

def semantic_sim(fA, fB):
    cos = np.dot(fA, fB)
    return (cos + 1) / 2

def match_detections(detections, objects, threshold=1.0):
    """
    Single-frame association (very simple):
    Each detection becomes a new object.

    UVO has per-frame graph; no temporal fusion.
    """
    return {i: None for i in range(len(detections))}
