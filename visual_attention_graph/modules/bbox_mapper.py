from visual_attention_graph.utils.geometry import iou


class BoundingBoxMapper:

    def map_bbox(self, frame, gaze_bbox, annotations):

        objects = annotations.get(str(frame))

        if objects is None:
            return None

        best_score = 0
        best_obj = None

        for obj_id, obj in enumerate(objects):

            obj_bbox = obj.get("box")

            if obj_bbox is None or len(obj_bbox) != 4:
                continue

            score = iou(gaze_bbox, obj_bbox)
            if score > best_score:
                print(
                    f"Frame {frame}: IoU between gaze bbox {gaze_bbox} and object {obj_id} bbox {obj_bbox} is {score:.4f}")

                best_score = score
                best_obj = obj_id

        if best_score < 0.1:
            return None

        return best_obj