import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class EmbeddingProjector:
    """
    Projects embeddings to 2D for visualization.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def project_pca(self, vectors: np.ndarray, n_components: int = 2) -> np.ndarray:
        projector = PCA(n_components=n_components, random_state=self.random_state)
        return projector.fit_transform(vectors)

    def project_tsne(
        self,
        vectors: np.ndarray,
        n_components: int = 2,
        perplexity: float = 30.0,
        max_iter: int = 1000,
    ) -> np.ndarray:
        projector = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            max_iter=max_iter,
            init="pca",
            learning_rate="auto",
            random_state=self.random_state,
        )
        return projector.fit_transform(vectors)
