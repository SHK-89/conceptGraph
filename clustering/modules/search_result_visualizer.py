from typing import Tuple, List, Optional

from matplotlib import pyplot as plt
import os

from clustering.modules.clustering_evaluation_result import ClusteringEvaluationResult


class SearchResultVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        self.figsize = figsize

    def plot_metric_vs_k(
        self,
        ks: List[int],
        values: List[Optional[float]],
        ylabel: str,
        title: str,
        save_path: str,
    ) -> None:
        filtered = [(k, v) for k, v in zip(ks, values) if v is not None]
        if not filtered:
            return

        x, y = zip(*filtered)

        plt.figure(figsize=self.figsize)
        plt.plot(x, y, marker="o")
        plt.xlabel("Number of clusters (k)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(list(x))
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_all_metrics(
        self,
        all_results: List[ClusteringEvaluationResult],
        output_dir: str,
    ) -> None:
        ks = [r.k for r in all_results]
        silhouettes = [r.silhouette for r in all_results]
        entropies = [r.normalized_entropy for r in all_results]

        self.plot_metric_vs_k(
            ks,
            silhouettes,
            ylabel="Silhouette Score",
            title="Silhouette vs k",
            save_path=os.path.join(output_dir, "silhouette_vs_k.png"),
        )

        self.plot_metric_vs_k(
            ks,
            entropies,
            ylabel="Normalized Entropy",
            title="Entropy vs k",
            save_path=os.path.join(output_dir, "entropy_vs_k.png"),
        )