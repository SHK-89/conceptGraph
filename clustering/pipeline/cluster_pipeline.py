import os
from typing import Dict, Optional

from clustering.modules.cluster_visualizer import ClusterVisualizer
from clustering.modules.distance_matrix_builder import DistanceMatrixBuilder
from clustering.modules.fixed_k_runner import FixedKRunner
from clustering.modules.grid_search_runner import GridSearchRunner
from clustering.modules.kmedois_cosine_evaluator import KMedoidsCosineEvaluator
from clustering.modules.result_saver import ResultSaver
from clustering.modules.search_result_visualizer import SearchResultVisualizer
from embedding.modules.embedding_json_loader import EmbeddingJSONLoader
from embedding.modules.embedding_preprocessor import EmbeddingPreprocessor
from embedding.modules.embedding_projector import EmbeddingProjector


class PredicateClusteringPipeline:
    def __init__(
            self,
            embedding_json_path: str,
            output_dir: str,
            mode: str = "fixed",  # "fixed" or "search"
            fixed_k: Optional[int] = 20, # works in fixed mode
            k_min: int = 10, # works in search mode
            k_max: int = 25, # works in search mode
            l2_normalize: bool = True,
            distance_metric: str = "cosine",  #
            random_state: int = 0,
            max_iter: int = 300,
            tsne_perplexity: float = 30.0,
            tsne_max_iter: int = 1000,
    ):
        self.embedding_json_path = embedding_json_path
        self.output_dir = output_dir
        self.mode = mode
        self.fixed_k = fixed_k
        self.tsne_perplexity = tsne_perplexity
        self.tsne_max_iter = tsne_max_iter

        self.loader = EmbeddingJSONLoader(embedding_json_path)
        self.preprocessor = EmbeddingPreprocessor(l2_normalize=l2_normalize)
        self.distance_builder = DistanceMatrixBuilder(metric=distance_metric)

        self.evaluator = KMedoidsCosineEvaluator(
            random_state=random_state,
            max_iter=max_iter,
            init="k-medoids++",
            method="pam",
        )

        if mode == "fixed":
            if fixed_k is None:
                raise ValueError("fixed_k must be provided in fixed mode.")
            self.runner = FixedKRunner(fixed_k=fixed_k)

        elif mode == "search":
            self.runner = GridSearchRunner(k_min=k_min, k_max=k_max)

        else:
            raise ValueError("mode must be 'fixed' or 'search'.")

        self.projector = EmbeddingProjector(random_state=random_state)
        self.cluster_visualizer = ClusterVisualizer()
        self.search_visualizer = SearchResultVisualizer()

    def run(self) -> Dict:
        os.makedirs(self.output_dir, exist_ok=True)

        predicates, vectors = self.loader.load()
        processed_vectors = self.preprocessor.transform(vectors)
        distance_matrix = self.distance_builder.build(processed_vectors)

        best_model, best_result, all_results = self.runner.run(
            evaluator=self.evaluator,
            vectors=processed_vectors,
            distance_matrix=distance_matrix,
        )

        labels = best_model.labels_

        results_path = os.path.join(self.output_dir, "clustering_results.json")
        final_clusters_path = os.path.join(self.output_dir, "final_clusters.json")
        cluster_report_json = os.path.join(self.output_dir, "cluster_report.json")

        ResultSaver.save_results(
            save_path=results_path,
            best_result=best_result,
            all_results=all_results,
        )

        ResultSaver.save_final_clusters(
            save_path=final_clusters_path,
            predicates=predicates,
            labels=labels,
            model=best_model,
            result=best_result,
        )

        if len(all_results) > 1:
            self.search_visualizer.plot_all_metrics(
                all_results=all_results,
                output_dir=self.output_dir,
            )

        # PCA visualization
        pca_points = self.projector.project_pca(processed_vectors)
        self.cluster_visualizer.plot_clusters_2d(
            points_2d=pca_points,
            labels=labels,
            predicates=predicates,
            save_path=os.path.join(self.output_dir, "clusters_pca.png"),
            title=f"K-Medoids Clusters (PCA 2D) - best k={best_result.k}",
            annotate=False,
        )
        # t-SNE visualization
        tsne_points = self.projector.project_tsne(
            processed_vectors,
            perplexity=self.tsne_perplexity,
            max_iter=self.tsne_max_iter,
        )
        self.cluster_visualizer.plot_clusters_2d(
            points_2d=tsne_points,
            labels=labels,
            predicates=predicates,
            save_path=os.path.join(self.output_dir, "clusters_tsne.png"),
            title=f"K-Medoids Clusters (t-SNE 2D) - best k={best_result.k}",
            annotate=False,
        )
        ResultSaver.save_cluster_report_json(
        save_path=cluster_report_json,
        predicates=predicates,
        labels=labels,
        distance_matrix=distance_matrix,
        model=best_model,
        top_n_representatives=5,
    )
        return {
        "mode": self.mode,
        "selected_k": best_result.k,
        "silhouette": best_result.silhouette,
        "normalized_entropy": best_result.normalized_entropy,
        "inertia": best_result.inertia,
        "results_path": results_path,
        "pca_plot_path": os.path.join(self.output_dir, "clusters_pca.png"),
        "tsne_plot_path": os.path.join(self.output_dir, "clusters_tsne.png"),
        "cluster_report_json": os.path.join(self.output_dir, "cluster_report.json"),
        "final_clusters_path": os.path.join(self.output_dir, "final_clusters.json"),
        "global_pca_plot": os.path.join(self.output_dir, "clusters_pca_global.png"),
        "global_tsne_plot": os.path.join(self.output_dir, "clusters_tsne_global.png"),
    }
