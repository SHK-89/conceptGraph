import math

import numpy as np
class EntropyScorer:
    """
    Computes normalized entropy of cluster-size distribution.

    H = -sum(p_i * log(p_i))
    H_norm = H / log(k)

    Returns 0 if k <= 1.
    """

    @staticmethod
    def normalized_cluster_entropy(labels: np.ndarray, k: int) -> float:
        if k <= 1:
            return 0.0

        counts = np.bincount(labels, minlength=k).astype(np.float64)
        probs = counts / counts.sum()

        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = math.log(k)

        if max_entropy == 0:
            return 0.0

        return float(entropy / max_entropy)
