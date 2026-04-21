import numpy as np
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids

from clustering.modules.clustering_evaluation_result import ClusteringEvaluationResult
from clustering.modules.entropy_scorer import EntropyScorer


class KMedoidsCosineEvaluator:
    """
    Runs K-Medoids on a precomputed cosine-distance matrix
    and evaluates the result with silhouette and entropy.
    """

    def __init__(
        self,
        random_state: int = 0,
        max_iter: int = 300,
        init: str = "k-medoids++",
        method: str = "pam",
    ):
        self.random_state = random_state
        self.max_iter = max_iter
        self.init = init
        self.method = method

    def fit_and_evaluate(
        self,
        vectors: np.ndarray,
        distance_matrix: np.ndarray,
        k: int,
    ):
        if k < 2:
            raise ValueError("k must be at least 2.")

        model = KMedoids(
            n_clusters=k,
            metric="precomputed",
            init=self.init,
            method=self.method,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )

        model.fit(distance_matrix)
        labels = model.labels_

        silhouette = None
        if len(np.unique(labels)) >= 2:
            silhouette = float(
                silhouette_score(distance_matrix, labels, metric="precomputed")
            )

        normalized_entropy = EntropyScorer.normalized_cluster_entropy(labels, k)

        cluster_sizes = {
            int(cluster_id): int(np.sum(labels == cluster_id))
            for cluster_id in range(k)
        }

        inertia = None
        if hasattr(model, "inertia_"):
            inertia = float(model.inertia_)

        result = ClusteringEvaluationResult(
            k=k,
            silhouette=silhouette,
            normalized_entropy=normalized_entropy,
            inertia=inertia,
            cluster_sizes=cluster_sizes,
        )

        return model, result