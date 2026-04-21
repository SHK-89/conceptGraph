from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ClusteringEvaluationResult:
    k: int
    silhouette: Optional[float]
    normalized_entropy: float
    inertia: Optional[float]
    cluster_sizes: Dict[int, int]