from __future__ import annotations

import re
from collections import Counter

def normalize_relation(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def top_counter_string(counter: Counter, top_k: int | None = 8) -> str:
    items = counter.most_common() if top_k is None else counter.most_common(top_k)
    return " | ".join(f"{k}:{v}" for k, v in items)

def top_counter_item(counter: Counter, top_k: int | None = 8) -> str:
    items = counter.most_common() if top_k is None else counter.most_common(top_k)
    return " | ".join(f"{key}" for key, _ in items)















