def point_in_bbox(point, bbox):

    x, y = point
    x1, y1, x2, y2 = bbox

    return x1 <= x <= x2 and y1 <= y <= y2


def map_gaze_to_object(gaze_bbox, objects):

    x1, y1, x2, y2 = gaze_bbox

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    for obj_id, obj in enumerate(objects):

        bbox = obj.get("box")

        if bbox is None or len(bbox) != 4:
            continue

        if point_in_bbox((cx, cy), bbox):

            return obj_id, (cx, cy)

    return None, (cx, cy)