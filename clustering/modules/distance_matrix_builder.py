import numpy as np
from sklearn.metrics import pairwise_distances

class DistanceMatrixBuilder:
    """
    Builds pairwise distance matrices from embeddings.
    """

    def __init__(self, metric: str = "cosine"):
        self.metric = metric

    def build(self, vectors: np.ndarray) -> np.ndarray:
        return pairwise_distances(vectors, metric=self.metric)