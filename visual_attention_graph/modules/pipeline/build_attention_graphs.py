from collections import defaultdict
from visual_attention_graph.modules.foveation_processor import FoveationProcessor
from visual_attention_graph.modules.gaze_loader import GazeDataLoader


class VisualAttentionGraphPipeline:

    def __init__(self, csv_path):

        self.loader = GazeDataLoader(csv_path)

        self.processor = FoveationProcessor()

    def run(self):

        df = self.loader.load()

        graphs = {}

        for video_id, video_df in df.groupby("video_id"):

            object_ids = self._collect_objects(video_df)

            matrix_builder = TransitionMatrixBuilder(object_ids)

            for participant, p_df in video_df.groupby("participant_id"):

                scanpath = SemanticScanpathBuilder().build_scanpath(
                    p_df,
                    self.processor
                )

                matrix_builder.update(scanpath)

            graph = AttentionGraphBuilder(
                matrix_builder.matrix,
                object_ids
            ).build_graph()

            graphs[video_id] = graph

        return graphs

    def _collect_objects(self, df):

        objs = set()

        for _, row in df.iterrows():

            objects = self.processor.extract_objects(row)

            objs.update(objects)

        return objs