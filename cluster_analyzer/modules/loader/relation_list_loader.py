from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from cluster_analyzer.modules.utility import normalize_relation


class RelationListLoader:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> Tuple[List[str], int | None]:
        with self.path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        # Format 1: all_predicates_dict.json → plain list, no n_participants
        if isinstance(raw, list):
            return [normalize_relation(x) for x in raw], None

        # Format 2: temporal_relation_dict.json → {"n_participants": int, "relations": [...]}
        if isinstance(raw, dict):
            if "relations" in raw:
                relations = [normalize_relation(x) for x in raw["relations"]]
                n_participants = raw.get("n_participants")  # None if key missing
                return relations, n_participants
            raise ValueError(f"Dict format not supported: expected 'relations' key, got keys: {list(raw.keys())}")

        # if "relations" in raw:
        # return [normalize_relation(x) for x in raw["relations"]]
        raise ValueError("Unsupported relation file format.")
