from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from cluster_analyzer.modules.cluster_comparison_analyzer import ClusterComparisonAnalyzer
from cluster_analyzer.modules.loader.cluster_report_loader import ClusterReportLoader
from cluster_analyzer.modules.loader.relation_list_loader import RelationListLoader
from cluster_analyzer.modules.mapper.relation_cluster_mapper import RelationClusterMapper
from cluster_analyzer.modules.visualization.plot_generator import PlotGenerator


class RelationPreferencePipeline:
    def __init__(self, cluster_report_path: str | Path, scene_relations_path: str | Path, temporal_relations_path: str | Path, output_dir: str | Path = "cluster_preference_outputs"):
        self.cluster_report_path = Path(cluster_report_path)
        self.scene_relations_path = Path(scene_relations_path)
        self.temporal_relations_path = Path(temporal_relations_path)
        self.output_dir = Path(output_dir)

    def run(self) -> Dict[str, Any]:
        clusters = ClusterReportLoader(self.cluster_report_path).load()
        scene_relations, _ = RelationListLoader(self.scene_relations_path).load()
        temporal_relations, n_participants = RelationListLoader(self.temporal_relations_path).load()

        print("clusters:", len(clusters))
        print("scene_relations:", len(scene_relations))
        print("temporal_relations:", len(temporal_relations))

        mapper = RelationClusterMapper(clusters, include_representatives=True)
        scene_assignments, unmapped_scene = mapper.map_relations(scene_relations, source="scene")
        temporal_assignments, unmapped_temporal = mapper.map_relations(temporal_relations, source="temporal")

        analyzer = ClusterComparisonAnalyzer(clusters)
        cluster_df, relation_df = analyzer.compare(scene_assignments, temporal_assignments, n_participants)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        cluster_csv = self.output_dir / "cluster_comparison_metrics.csv"
        relation_csv = self.output_dir / "relation_comparison_metrics.csv"
        summary_json = self.output_dir / "summary.json"
        unmapped_json = self.output_dir / "unmapped_relations.json"

        cluster_df.to_csv(cluster_csv, index=False, encoding="utf-8")
        relation_df.to_csv(relation_csv, index=False, encoding="utf-8")

        with summary_json.open("w", encoding="utf-8") as f:
            json.dump({
                "n_scene_relations_input": len(scene_relations),
                "n_temporal_relations_input": len(temporal_relations),
                "n_scene_assignments": len(scene_assignments),
                "n_temporal_assignments": len(temporal_assignments),
                "top_clusters_by_p_saccade_given_cluster": cluster_df.head(20).to_dict(orient="records"),
            }, f, indent=2, ensure_ascii=False)

        with unmapped_json.open("w", encoding="utf-8") as f:
            json.dump({
                "unmapped_scene_count": len(unmapped_scene),
                "unmapped_temporal_count": len(unmapped_temporal),
                "unmapped_scene_examples": sorted(set(unmapped_scene))[:100],
                "unmapped_temporal_examples": sorted(set(unmapped_temporal))[:100],
            }, f, indent=2, ensure_ascii=False)

        plotter = PlotGenerator(self.output_dir)
        outputs = {
            "cluster_metrics_csv": str(cluster_csv),
            "relation_metrics_csv": str(relation_csv),
            "summary_json": str(summary_json),
            "unmapped_json": str(unmapped_json),
            #"cluster_relation_count_heatmap_temporal": str(cluster_relation_count_heatmap_temporal),
            "scene_vs_temporal_counts": str(plotter.plot_scene_vs_temporal_counts(cluster_df)),
            "relation_heatmap": str(plotter.plot_cluster_relation_count_heatmap(relation_df)),
            "p_saccade_given_cluster": str(plotter.plot_top_metric(cluster_df, len(scene_assignments), len(temporal_assignments), n_participants, "p_saccade_given_cluster", "Top Clusters by P(saccade | cluster) %", "P(saccade | cluster) %", "p_saccade_given_cluster.png")),
            #"to_clusters": str(plotter.plot_top_metric(cluster_df, len(scene_assignments), len(temporal_assignments), "top_clusters", "Top Clusters by P(saccade | cluster)", "P(saccade | cluster)", "top_clusters.png")),
            "enrichment_over_availability": str(plotter.plot_top_metric(cluster_df, len(scene_assignments), len(temporal_assignments), n_participants,"enrichment_over_availability", "Top Clusters by Enrichment over Scene Availability", "Enrichment", "enrichment_over_availability.png")),
            "availability_vs_selection_scatter": str(plotter.plot_availability_vs_selection(cluster_df)),
            "wordclouds_temporal_count": [str(p) for p in plotter.generate_cluster_wordclouds(relation_df, metric="n_temporal", top_k_clusters=12, min_value=1.0)],
            "wordclouds_temporal_share": [str(p) for p in plotter.generate_cluster_wordclouds(relation_df, metric="p_saccade_given_cluster", top_k_clusters=12, min_value=0.001)],
            "wordclouds_relation_selection": [str(p) for p in plotter.generate_cluster_wordclouds(relation_df, metric="p_saccade_given_relation", top_k_clusters=12, min_value=0.001)],
            "wordclouds_cluster_report_members": [str(p) for p in plotter.generate_cluster_report_wordclouds(clusters, max_clusters=10)],

        }
        return outputs
