from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from cluster_analyzer.modules.datamodel.cluster_info import ClusterInfo


class ClusterReportLoader:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> Dict[str, ClusterInfo]:
        with self.path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        out: Dict[str, ClusterInfo] = {}
        for cluster_id, info in raw.items():
            out[str(cluster_id)] = ClusterInfo(
                cluster_id=str(cluster_id),
                size=int(info["size"]),
                name=info["name"], #TODO: add name later
                representatives=list(info.get("representatives", [])),
                members=list(info.get("members", [])),
            )
        return out