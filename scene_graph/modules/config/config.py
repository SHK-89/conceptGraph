# modules/config.py

from dataclasses import dataclass
from typing import Optional


@dataclass
class BatchConfig:
    video_dir: str
    ann_dir: str
    output_root: str

    max_frames: int = 90
    max_videos: Optional[int] = None
    device: str = "cuda"

    draw_scene_graph: bool = False
    draw_overlay: bool = False
    enable_caption: bool = False
    quiet_mode: bool = True
    store_crop: bool = True

    resume: bool = True
