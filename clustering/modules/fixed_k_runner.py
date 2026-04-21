import numpy as np

from clustering.modules.kmedois_cosine_evaluator import KMedoidsCosineEvaluator


class FixedKRunner:
    """
    Runs clustering for a single fixed k.
    """

    def __init__(self, fixed_k: int):
        if fixed_k < 2:
            raise ValueError("fixed_k must be at least 2.")
        self.fixed_k = fixed_k

    def run(
        self,
        evaluator: KMedoidsCosineEvaluator,
        vectors: np.ndarray,
        distance_matrix: np.ndarray,
    ):
        model, result = evaluator.fit_and_evaluate(
            vectors=vectors,
            distance_matrix=distance_matrix,
            k=self.fixed_k,
        )
        return model, result, [result]