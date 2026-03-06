class SemanticScanpathBuilder:

    def __init__(self):
        pass

    def build_scanpath(self, participant_df, processor):

        scanpath = []

        for _, row in participant_df.iterrows():

            objects = processor.extract_objects(row)

            if len(objects) > 0:
                scanpath.append(objects)

        return scanpath