from dataclasses import dataclass

@dataclass
class BatchConfig:
    video_dir: str
    ann_dir: str
    gaze_csv: str
    output_root: str

