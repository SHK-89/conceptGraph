from typing import List, Dict
import numpy as np
from sklearn_extra.cluster import KMedoids

class ClusterInterpreter:
    @staticmethod
    def build_cluster_members(
        predicates: List[str],
        labels: np.ndarray,
    ) -> Dict[int, List[str]]:
        clusters: Dict[int, List[str]] = {}
        for predicate, label in zip(predicates, labels):
            clusters.setdefault(int(label), []).append(predicate)

        for cluster_id in clusters:
            clusters[cluster_id] = sorted(clusters[cluster_id])

        return clusters

    @staticmethod
    def get_cluster_representatives_from_medoids(
        predicates: List[str],
        model: KMedoids,
    ) -> Dict[int, str]:
        representatives = {}
        for cluster_id, medoid_index in enumerate(model.medoid_indices_):
            representatives[int(cluster_id)] = predicates[int(medoid_index)]
        return representatives

    @staticmethod
    def get_top_representatives_per_cluster(
        predicates: List[str],
        labels: np.ndarray,
        distance_matrix: np.ndarray,
        model: KMedoids,
        top_n: int = 5,
    ) -> Dict[int, List[str]]:
        """
        For each cluster, return top_n representatives based on distance
        to the cluster medoid. The first one is usually the medoid itself.
        """
        reps: Dict[int, List[str]] = {}

        for cluster_id, medoid_index in enumerate(model.medoid_indices_):
            cluster_indices = np.where(labels == cluster_id)[0]

            # Sort members of this cluster by distance to the medoid
            sorted_cluster_indices = sorted(
                cluster_indices,
                key=lambda idx: distance_matrix[idx, medoid_index]
            )

            reps[int(cluster_id)] = [
                predicates[int(idx)] for idx in sorted_cluster_indices[:top_n]
            ]

        return reps