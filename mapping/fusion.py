def fuse(objects, detections, assoc):
    """
    With no camera pose and no temporal consistency,
    each detection becomes a new object node.
    """
    for i, det in enumerate(detections):
        objects.append({
            "points": det["points"],
            "feat": det["feat"],
            "views": [det["rgb_crop"]],
            "caption": None
        })
    return objects
