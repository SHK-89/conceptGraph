import os
from typing import Iterator, Tuple


class UVOVideoDataset:
    """
    Iterates over valid (video_name, video_path, annotation_path) tuples.
    """

    def __init__(self, video_dir: str, ann_dir: str):
        self.video_dir = video_dir
        self.ann_dir = ann_dir

        self.video_files = sorted([
            f for f in os.listdir(video_dir)
            if f.endswith(".mp4")
        ])

    def __iter__(self) -> Iterator[Tuple[str, str, str]]:
        for filename in self.video_files:
            video_name = os.path.splitext(filename)[0]

            video_path = os.path.join(self.video_dir, filename)
            ann_path = os.path.join(self.ann_dir, f"{video_name}.pkl")

            if not os.path.exists(ann_path):
                continue

            yield video_name, video_path, ann_path

    def __len__(self):
        return len(self.video_files)
