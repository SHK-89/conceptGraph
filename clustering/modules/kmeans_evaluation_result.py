from dataclasses import dataclass
from typing import Optional, Dict
@dataclass
class KMeansEvaluationResult:
    k: int
    silhouette: Optional[float]
    normalized_entropy: float
    combined_score: Optional[float]
    adjusted_score: Optional[float]
    inertia: float
    cluster_sizes: Dict[int, int]
    num_tiny_clusters: int
    tiny_cluster_fraction: float
    average_cluster_size: float
    min_cluster_size: int