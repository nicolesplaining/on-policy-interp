"""Small IO / reproducibility helpers used across subsystems."""
from __future__ import annotations

import json
import os
import random
from typing import Dict, Iterable, Iterator, List


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (incl. CUDA) for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_jsonl(path: str) -> List[Dict]:
    """Read a ``.jsonl`` file into a list of dicts."""
    items: List[Dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def iter_jsonl(path: str) -> Iterator[Dict]:
    """Stream a ``.jsonl`` file record by record."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, records: Iterable[Dict]) -> int:
    """Write ``records`` to ``path`` as JSONL; returns the count written."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            n += 1
    return n


def write_json(path: str, obj) -> None:
    """Write ``obj`` to ``path`` as pretty JSON, creating parent dirs."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path: str):
    with open(path) as f:
        return json.load(f)
