import numpy as np

def bbox_center(bbox):
    x, y, w, h = bbox
    return np.array([x + w/2, y + h/2])

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
