from clustering.modules.kmedois_cosine_evaluator import KMedoidsCosineEvaluator
import numpy as np

class GridSearchRunner:
    """
    Runs clustering for a range of k values.
    Selection is based on silhouette only.
    """

    def __init__(self, k_min: int, k_max: int):
        if k_min < 2:
            raise ValueError("k_min must be at least 2.")
        if k_max < k_min:
            raise ValueError("k_max must be >= k_min.")
        self.k_min = k_min
        self.k_max = k_max

    def run(
        self,
        evaluator: KMedoidsCosineEvaluator,
        vectors: np.ndarray,
        distance_matrix: np.ndarray,
    ):
        all_results = []
        models_by_k = {}

        for k in range(self.k_min, self.k_max + 1):
            model, result = evaluator.fit_and_evaluate(
                vectors=vectors,
                distance_matrix=distance_matrix,
                k=k,
            )
            all_results.append(result)
            models_by_k[k] = model

        valid = [r for r in all_results if r.silhouette is not None]
        if not valid:
            raise RuntimeError("No valid silhouette scores found.")

        best_result = max(valid, key=lambda r: r.silhouette)
        best_model = models_by_k[best_result.k]

        return best_model, best_result, all_results