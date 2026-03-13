from visual_attention_graph.modules.bbox_mapper import BoundingBoxMapper
from visual_attention_graph.modules.foveation_parser import FoveationParser
from visual_attention_graph.modules.scanpath_builder import ScanpathBuilder
from visual_attention_graph.modules.transition_matrix import TransitionMatrix
from visual_attention_graph.modules.graph_builder import GraphBuilder

import pandas as pd
import pickle
import os

class AttentionPipeline:

    def __init__(self, gaze_csv, annotation_dir):

        self.df = pd.read_csv(gaze_csv)
        self.annotation_dir = annotation_dir

        mapper = BoundingBoxMapper()
        parser = FoveationParser(mapper)

        self.scanpath_builder = ScanpathBuilder(parser)
        self.graph_builder = GraphBuilder()

    def run_single_video_OLD(self, video_name):

        video_df = self.df[self.df["filename"] == video_name]

        annotation_path = os.path.join(
            self.annotation_dir,
            f"{video_name[:-4]}.pkl"
        )
        print("Loading annotations from:", annotation_path)
        print("Annotation file exists:", os.path.exists(annotation_path))
        print("Annotation file path:", annotation_path)
        with open(annotation_path, "rb") as f:
            annotations = pickle.load(f)

        print("Video:", video_name)
        print("Total rows:", len(video_df))
        print("Unique participants:", video_df["subject"].nunique())
        print("Total annotations:", sum(len(v) for v in annotations.values()))
        print("annotation names:", list(annotations.keys())[:])

        scanpath = self.scanpath_builder.build_scanpath(
            video_df,
            annotations
        )

        graph = self.graph_builder.build(scanpath)

        return graph

    def run_single_video(self, video_name):

        video_df = self.df[self.df["filename"].str.strip() == video_name]

        annotation_path = os.path.join(
            self.annotation_dir,
            f"{video_name[:-4]}.pkl"
        )

        with open(annotation_path, "rb") as f:

            annotations = pickle.load(f)

        all_scanpaths = []

        # --------------------------------
        # build scanpath per participant
        # --------------------------------

        for subject_id, participant_df in video_df.groupby("subject"):

            scanpath = self.scanpath_builder.build_scanpath(
                participant_df,
                annotations
            )

            if len(scanpath) > 1:
                all_scanpaths.append(scanpath)

        # --------------------------------
        # build attention graph
        # --------------------------------

        graph = self.graph_builder.build(all_scanpaths)

        return graph
    def run_all_videos(self):

        df = self.loader.load()

        videos = df["filename"].unique()

        graphs = {}

        print("\nTotal videos:", len(videos))

        for video in videos:
            graph = self.run_single_video(video)

            graphs[video] = graph

        return graphs