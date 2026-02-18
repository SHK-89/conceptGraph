from dataclasses import dataclass, asdict
from typing import Tuple, Optional
import numpy as np


@dataclass
class DetectedObject:
    frame: int
    obj_id: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    caption: str
    feature: np.ndarray
    crop: Optional[np.ndarray] = None

    def to_dict(self):
        d = asdict(self)
        d["feature"] = self.feature.tolist()  # numpy → serializable
        d["crop"] = None  # don't serialize full images
        return d


@dataclass
class RelationEdge:
    frame: int
    src_id: int
    dst_id: int
    relation: str

    def to_dict(self):
        return asdict(self)
