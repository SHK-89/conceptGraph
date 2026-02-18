from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from scenegraph.core.models import DetectedObject, RelationEdge


@dataclass
class VideoSceneGraph:
    video_name: str
    objects: List[DetectedObject]
    edges: List[RelationEdge]
    frame_objects: Dict[int, List[DetectedObject]]
    frames: Dict[int, np.ndarray]

    # ----------------------------
    # Utility Methods
    # ----------------------------

    def get_objects_in_frame(self, frame_idx: int):
        return self.frame_objects.get(frame_idx, [])

    def to_dict(self):
        return {
            "video_name": self.video_name,
            "objects": [o.to_dict() for o in self.objects],
            "edges": [e.to_dict() for e in self.edges],
        }

    def num_objects(self):
        return len(self.objects)

    def num_edges(self):
        return len(self.edges)
