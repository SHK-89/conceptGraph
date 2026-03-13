import cv2
import pandas as pd
import pickle
import ast
import os

from visual_attention_graph.modules.bbox_mapper import BoundingBoxMapper
from visual_attention_graph.debug.gaze_object_visualizer import GazeObjectVisualizer


class DebugVideoRunner:

    def __init__(self, gaze_csv, annotation_dir, video_dir):

        self.gaze_csv = gaze_csv
        self.annotation_dir = annotation_dir
        self.video_dir = video_dir

        self.mapper = BoundingBoxMapper()
        self.visualizer = GazeObjectVisualizer()


    def run(self, video_name, output_dir):

        video_path = os.path.join(self.video_dir, video_name)

        annotation_path = os.path.join(
            self.annotation_dir,
            f"{video_name[:-4]}.pkl"
        )

        output_video = os.path.join(
            output_dir,
            "debug_attention_mapping.mp4"
        )

        df = pd.read_csv(self.gaze_csv)

        video_df = df[
            (df["filename"] == video_name) &
            (df["event"] == "FOV")
        ]

        with open(annotation_path, "rb") as f:
            annotations = pickle.load(f)

        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        writer = cv2.VideoWriter(
            output_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        frame_id = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            objects = annotations.get(str(frame_id), [])

            frame = self.visualizer.draw_objects(frame, objects)

            rows = video_df[
                (video_df["frame_start"] <= frame_id) &
                (video_df["frame_end"] >= frame_id)
            ]

            for _, row in rows.iterrows():

                bbox_dict = ast.literal_eval(
                    row["gt_object_multiple_bbox"]
                )

                if str(frame_id) not in bbox_dict:
                    continue

                for gaze_bbox in bbox_dict[str(frame_id)]:

                    frame = self.visualizer.draw_gaze(frame, gaze_bbox)

                    obj_id = self.mapper.map_bbox(
                        frame_id,
                        gaze_bbox,
                        annotations
                    )

                    if obj_id is not None:

                        bbox = objects[obj_id]["box"]

                        frame = self.visualizer.draw_match(frame, bbox)

            writer.write(frame)

            frame_id += 1

        cap.release()
        writer.release()

        print("Debug video saved:", output_video)