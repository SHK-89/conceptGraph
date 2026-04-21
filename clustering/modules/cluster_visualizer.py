from typing import Tuple, List
import numpy as np
from matplotlib import pyplot as plt
class ClusterVisualizer:
    """
    Saves cluster scatter plots in 2D.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 9)):
        self.figsize = figsize

    def plot_clusters_2d(
        self,
        points_2d: np.ndarray,
        labels: np.ndarray,
        predicates: List[str],
        save_path: str,
        title: str,
        annotate: bool = False,
    ) -> None:
        plt.figure(figsize=self.figsize)
        print("Plotting clusters...")
        unique_clusters = sorted(np.unique(labels))
        for cluster_id in unique_clusters:
            mask = labels == cluster_id
            plt.scatter(
                points_2d[mask, 0],
                points_2d[mask, 1],
                label=f"Cluster {cluster_id}",
                alpha=0.8,
            )

        if annotate:
            for i, predicate in enumerate(predicates):
                plt.annotate(
                    predicate,
                    (points_2d[i, 0], points_2d[i, 1]),
                    fontsize=8,
                    alpha=0.8,
                )

        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

