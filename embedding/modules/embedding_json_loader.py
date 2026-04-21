import json
from typing import List, Tuple

import numpy as np


class EmbeddingJSONLoader:
    """
    Loads embeddings from JSON.

    Expected format:
    {
      "predicates": ["above", "adjacent", ...],
      "vectors": [[...], [...], ...]
    }

    Or:
    {
      "above": [0.1, 0.2, ...],
      "adjacent": [0.3, 0.4, ...]
    }
    """

    def __init__(self, json_path: str):
        self.json_path = json_path

    def load(self) -> Tuple[List[str], np.ndarray]:
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "predicates" in data and "vectors" in data:
            predicates = data["predicates"]
            vectors = np.asarray(data["vectors"], dtype=np.float32)
        else:
            predicates = list(data.keys())
            vectors = np.asarray([data[p] for p in predicates], dtype=np.float32)

        if len(predicates) != len(vectors):
            raise ValueError("Mismatch between number of predicates and number of vectors.")

        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2D array of shape [N, D].")

        return predicates, vectors
