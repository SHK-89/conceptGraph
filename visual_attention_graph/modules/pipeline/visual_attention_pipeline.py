from collections import defaultdict

from visual_attention_graph.modules.gaze_loader import GazeDataLoader
from visual_attention_graph.modules.annotation_loader import AnnotationLoader
from visual_attention_graph.modules.bbox_mapper import BoundingBoxMapper
from visual_attention_graph.modules.foveation_parser import FoveationParser
from visual_attention_graph.modules.scanpath_builder import ScanpathBuilder
from visual_attention_graph.modules.transition_matrix import TransitionMatrix
from visual_attention_graph.modules.attention_graph import AttentionGraphBuilder


class VisualAttentionPipeline:

    def __init__(self, gaze_csv, annotation_dir):

        self.loader = GazeDataLoader(gaze_csv)

        self.annotation_loader = AnnotationLoader(annotation_dir)

        mapper = BoundingBoxMapper()

        parser = FoveationParser(mapper)

        self.scanpath_builder = ScanpathBuilder(parser)

    def run(self):

        df = self.loader.load()

        graphs = {}

        for video, video_df in df.groupby("filename"):

            annotations = self.annotation_loader.load(video)

            transition_matrix = TransitionMatrix()

            for subject, participant_df in video_df.groupby("subject"):

                scanpath = self.scanpath_builder.build_scanpath(
                    participant_df,
                    annotations
                )

                transition_matrix.update(scanpath)

            graph = AttentionGraphBuilder().build(
                transition_matrix.edges()
            )

            graphs[video] = graph

        return graphs