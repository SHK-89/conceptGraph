import numpy as np
class EmbeddingPreprocessor:
    """
    Optional preprocessing for embeddings before clustering.
    """

    def __init__(self, l2_normalize: bool = True):
        self.l2_normalize = l2_normalize

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        x = vectors.copy()

        if self.l2_normalize:
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            x = x / norms

        return x

