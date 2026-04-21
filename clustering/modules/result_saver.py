import csv
import json
from dataclasses import asdict
from typing import List, Dict

import numpy as np
from sklearn_extra.cluster import KMedoids
from clustering.modules.cluster_interpreter import ClusterInterpreter
from clustering.modules.clustering_evaluation_result import ClusteringEvaluationResult


class ResultSaver:
    @staticmethod
    def save_results(
            save_path: str,
            best_result: ClusteringEvaluationResult,
            all_results: List[ClusteringEvaluationResult],
    ) -> None:
        payload = {
            "best_result": asdict(best_result),
            "all_results": [asdict(r) for r in all_results],
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def save_final_clusters(
            save_path: str,
            predicates: List[str],
            labels: np.ndarray,
            model: KMedoids,
            result: ClusteringEvaluationResult,
    ) -> None:
        clusters = ClusterInterpreter.build_cluster_members(predicates, labels)
        representatives = ClusterInterpreter.get_cluster_representatives_from_medoids(
            predicates=predicates,
            model=model,
        )

        assignments = [
            {"predicate": predicate, "cluster": int(label)}
            for predicate, label in zip(predicates, labels)
        ]

        payload = {
            "k": result.k,
            "silhouette": result.silhouette,
            "normalized_entropy": result.normalized_entropy,
            "inertia": result.inertia,
            "cluster_sizes": result.cluster_sizes,
            "medoid_indices": [int(i) for i in model.medoid_indices_],
            "representatives": representatives,
            "assignments": assignments,
            "clusters": clusters,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def save_cluster_report_json(
            save_path: str,
            predicates: List[str],
            labels: np.ndarray,
            distance_matrix: np.ndarray,
            model: KMedoids,
            top_n_representatives: int = 5,
    ) -> None:
        """
        Saves a report in the format:
        {
          "4": {
            "size": 86,
            "representatives": [...],
            "members": [...]
          },
          ...
        }
        """
        clusters = ClusterInterpreter.build_cluster_members(predicates, labels)
        representatives = ClusterInterpreter.get_top_representatives_per_cluster(
            predicates=predicates,
            labels=labels,
            distance_matrix=distance_matrix,
            model=model,
            top_n=top_n_representatives,
        )

        payload = {}
        for cluster_id in sorted(clusters.keys()):
            payload[str(cluster_id)] = {
                "size": len(clusters[cluster_id]),
                "representatives": representatives.get(cluster_id, []),
                "members": clusters[cluster_id],
            }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
