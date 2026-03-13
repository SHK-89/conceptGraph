import ast

class ScanpathBuilder:

    def __init__(self, parser):
        self.parser = parser


    def build_scanpath(self, participant_df, annotations):

        scanpath = []

        for _, row in participant_df.iterrows():

            if row["event"] != "FOV":
                continue

            objects = self.parser.extract_objects(row, annotations)

            for obj in objects:

                scanpath.append(obj)

        # order by frame
        scanpath.sort(key=lambda x: x["frame"])

        # remove repeated objects
        cleaned = []

        prev_obj = None

        for step in scanpath:

            obj_id = step["object_id"]

            if obj_id != prev_obj:

                cleaned.append(step)

            prev_obj = obj_id

        return cleaned