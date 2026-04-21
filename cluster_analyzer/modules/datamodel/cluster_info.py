from dataclasses import dataclass
from typing import List
import re

from cluster_analyzer.modules.utility import normalize_relation


@dataclass
class ClusterInfo:
    cluster_id: str
    size: int
    name: str
    representatives: List[str]
    members: List[str]

    def normalized_members(self) -> List[str]:
        return [normalize_relation(x) for x in self.members]

    def normalized_representatives(self) -> List[str]:
        return [normalize_relation(x) for x in self.representatives]
