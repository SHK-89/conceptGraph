import ast


class FoveationProcessor:
    def __init__(self):
        pass

    def extract_objects(self, row):

        bbox_dict = ast.literal_eval(row["gt_object_multiple_bbox"])

        objects = set()

        for frame, bboxes in bbox_dict.items():

            for bbox in bboxes:

                obj_id = self._bbox_to_object_id(bbox)

                if obj_id is not None:
                    objects.add(obj_id)

        return list(objects)

    def _bbox_to_object_id(self, bbox):

        # implement mapping bbox -> object id
        # depending on your UVO annotation

        return bbox["object_id"]