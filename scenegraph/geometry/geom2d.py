import numpy as np

def bbox_center(bbox):
    x, y, w, h = bbox
    return np.array([x + w/2, y + h/2])

def iou_2d(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = w1 * h1
    area2 = w2 * h2
    return inter / (area1 + area2 - inter + 1e-6)

def spatial_relation_heuristic(cA, cB):
    """
    Heuristic 2D relation based on centroid positions.
    """
    dx = cB[0] - cA[0]
    dy = cB[1] - cA[1]

    if abs(dx) > abs(dy):
        return "A is left of B" if dx > 0 else "A is right of B"
    else:
        return "A is above B" if dy > 0 else "A is below B"


def union_bbox(bboxA, bboxB):
    """Return the minimal bounding box covering both bboxes."""
    x1 = min(bboxA[0], bboxB[0])
    y1 = min(bboxA[1], bboxB[1])
    x2 = max(bboxA[0] + bboxA[2], bboxB[0] + bboxB[2])
    y2 = max(bboxA[1] + bboxA[3], bboxB[1] + bboxB[3])
    return [x1, y1, x2 - x1, y2 - y1]

def crop_pair_region(rgb, bboxA, bboxB):
    """Crop the region that contains both objects."""
    x, y, w, h = union_bbox(bboxA, bboxB)
    x, y, w, h = map(int, (x, y, w, h))
    crop = rgb[y:y+h, x:x+w]
    return crop, (x, y)

def remap_center(center, offset):
    """Convert global pixel coordinates to the cropped-region coordinates."""
    ox, oy = offset
    return (center[0] - ox, center[1] - oy)
