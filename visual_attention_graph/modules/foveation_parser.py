import ast


class FoveationParser:

    def __init__(self, bbox_mapper):
        self.bbox_mapper = bbox_mapper

    def extract_objects(self, row, annotations):

        results = []

        bbox_dict = ast.literal_eval(row["gt_object_multiple_bbox"])

        for frame, bbox_list in bbox_dict.items():

            frame_objects = annotations.get(str(frame))

            if frame_objects is None:
                continue

            if bbox_list is None or bbox_list == []:
                continue

            for bbox in bbox_list:

                obj_id = self.bbox_mapper.map_bbox(
                    frame,
                    bbox,
                    annotations
                )

                if obj_id is None:
                    continue

                obj_bbox = frame_objects[obj_id]["box"]

                results.append(
                    {
                        "frame": int(frame),
                        "object_id": obj_id,
                        "bbox": obj_bbox
                    }
                )

        return results


