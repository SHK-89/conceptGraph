from __future__ import annotations

from typing import Dict, List, Iterable, Tuple

from cluster_analyzer.modules.datamodel.assigment import Assignment
from cluster_analyzer.modules.datamodel.cluster_info import ClusterInfo
from cluster_analyzer.modules.utility import normalize_relation


class RelationClusterMapper:
    def __init__(self, clusters: Dict[str, ClusterInfo], include_representatives: bool = True):
        self.clusters = clusters
        self.include_representatives = include_representatives
        self.lookup = self._build_lookup()

    def _build_lookup(self) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for cluster_id, cluster in self.clusters.items():
            for member in cluster.normalized_members():
                lookup[member] = cluster_id
            if self.include_representatives:
                for rep in cluster.normalized_representatives():
                    lookup.setdefault(rep, cluster_id)
        return lookup

    def map_relations(self, relations: Iterable[str], source: str) -> Tuple[List[Assignment], List[str]]:
        assignments: List[Assignment] = []
        unmapped: List[str] = []
        for rel in relations:
            norm = normalize_relation(rel)
            cluster_id = self.lookup.get(norm)
            if cluster_id is None:
                unmapped.append(norm)
                continue
            assignments.append(Assignment(norm, cluster_id, source))
        return assignments, unmapped
