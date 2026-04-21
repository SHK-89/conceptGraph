from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import pandas as pd

from cluster_analyzer.modules.datamodel.assigment import Assignment
from cluster_analyzer.modules.datamodel.cluster_info import ClusterInfo
from cluster_analyzer.modules.utility import safe_div, top_counter_string, top_counter_item


class ClusterComparisonAnalyzer:
    def __init__(self, clusters: Dict[str, ClusterInfo]):
        self.clusters = clusters

    def compare(self, scene_assignments: List[Assignment], temporal_assignments: List[Assignment], n_participants: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        scene_cluster_counts = Counter(a.cluster_id for a in scene_assignments)
        temporal_cluster_counts = Counter(a.cluster_id for a in temporal_assignments)

        scene_rel_by_cluster: Dict[str, Counter] = defaultdict(Counter)
        temp_rel_by_cluster: Dict[str, Counter] = defaultdict(Counter)
        for a in scene_assignments:
            scene_rel_by_cluster[a.cluster_id][a.relation] += 1
        for a in temporal_assignments:
            temp_rel_by_cluster[a.cluster_id][a.relation] += 1

        total_scene = len(scene_assignments)
        total_temp = len(temporal_assignments)

        cluster_rows = []
        relation_rows = []
        for cluster_id, cluster in self.clusters.items():
            n_all = scene_cluster_counts.get(cluster_id, 0)
            n_temp = temporal_cluster_counts.get(cluster_id, 0)
            p_all = safe_div(n_all, total_scene)
            p_temp = safe_div(n_temp, total_temp)
            #p_saccade_given_cluster = safe_div(n_temp, n_all)
            p_saccade_given_cluster = safe_div(n_temp * 1000, n_all * n_participants)
            enrichment = safe_div(p_temp, p_all)
            scene_counter = scene_rel_by_cluster.get(cluster_id, Counter())
            temp_counter = temp_rel_by_cluster.get(cluster_id, Counter())

            cluster_rows.append({
                "cluster_id": cluster_id,
                "cluster_size": cluster.size,
                "cluster_name": cluster.name, #TODO: SHK add name later
                #"representatives": " | ".join(cluster.representatives[:5]),
                "representatives" : top_counter_item(temp_counter, 1),
                "n_all_scene": n_all,
                "n_temporal": n_temp,
                "p_all_scene": p_all,
                "p_temporal": p_temp,
                "p_saccade_given_cluster": p_saccade_given_cluster,
                "enrichment_over_availability": enrichment,
                "top_scene_relations": top_counter_string(scene_counter, None),
                "top_temporal_relations": top_counter_string(temp_counter, None),
            })

            for relation in sorted(set(scene_counter) | set(temp_counter)):
                n_all_rel = scene_counter.get(relation, 0)
                n_temp_rel = temp_counter.get(relation, 0)
                p_all_given_cluster = safe_div(n_all_rel, n_all)
                p_saccade_given_cluster = safe_div(n_temp_rel, n_temp)
                p_saccade_given_relation = safe_div(n_temp_rel * 1000, n_all_rel * n_participants)
                #p_saccade_given_relation = safe_div(n_temp_rel, n_all_rel)
                relation_rows.append({
                    "cluster_id": cluster_id,
                    "cluster_size": cluster.size,
                    "cluster_name": cluster.name,
                    "relation": relation,
                    "n_all_scene": n_all_rel,
                    "n_temporal": n_temp_rel,
                    "p_all_given_cluster": p_all_given_cluster,
                    "p_saccade_given_cluster": p_saccade_given_cluster,
                    "p_saccade_given_relation": p_saccade_given_relation,
                    "temporal_minus_scene_share": p_saccade_given_cluster - p_all_given_cluster,
                })

        cluster_df = pd.DataFrame(cluster_rows).sort_values(["p_saccade_given_cluster", "n_temporal"], ascending=[False, False]).reset_index(drop=True)
        relation_df = pd.DataFrame(relation_rows).sort_values(["p_saccade_given_relation", "n_temporal"], ascending=[False, False]).reset_index(drop=True)
        return cluster_df, relation_df
