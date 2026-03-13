import os
import pickle
import pandas as pd


class FrameObjectAnalyzer:

    def __init__(self, annotation_dir):

        self.annotation_dir = annotation_dir


    def load_annotation(self, file_path):

        with open(file_path, "rb") as f:
            annotation = pickle.load(f)

        return annotation


    def count_objects_per_frame(self, annotation, video_name):

        rows = []

        for frame, objects in annotation.items():

            rows.append({
                "video": video_name,
                "frame": int(frame),
                "num_objects": len(objects)
            })

        return rows


    def analyze_all_frames(self):

        all_rows = []

        for file in os.listdir(self.annotation_dir):

            if not file.endswith(".pkl"):
                continue

            file_path = os.path.join(self.annotation_dir, file)

            annotation = self.load_annotation(file_path)

            rows = self.count_objects_per_frame(annotation, file)

            all_rows.extend(rows)

        df = pd.DataFrame(all_rows)

        df = df.sort_values(["video", "frame"])

        return df


    def summarize_by_video(self, frame_df):

        video_df = (
            frame_df.groupby("video")["num_objects"]
            .agg(
                max_objects="max",
                total_frames="count"
            )
            .sort_values("max_objects", ascending=False)
        )

        return video_df


    def save_results(self, frame_df, video_df, output_dir="outputs"):

        os.makedirs(output_dir, exist_ok=True)

        frame_path = os.path.join(output_dir, "frame_object_counts.csv")
        video_path = os.path.join(output_dir, "video_object_statistics.csv")

        frame_df.to_csv(frame_path, index=False)
        video_df.to_csv(video_path)

        print("Saved:", frame_path)
        print("Saved:", video_path)