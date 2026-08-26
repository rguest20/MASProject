"""
agents/cognition/semantic_utils.py
Pure semantic utility functions (vector ops + lightweight text keywording).
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from config import DIMS

DIM = DIMS

_STOP = set(
    """
a an the and or of for to by in on at as with without into from over under out
is are was were be been being it this that these those there here then than if
but not no yes do does did done can could will would should may might must have
has had i you he she we they them us our your their its which who whom whose
""".split()
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")


def rand_vec(dim: int = DIM, span: float = 0.1):
    return [random.uniform(-span, span) for _ in range(dim)]


def add(u, v):
    return [a + b for a, b in zip(u, v)]


def sub(u, v):
    return [a - b for a, b in zip(u, v)]


def scale(u, s: float):
    return [a * s for a in u]


def lerp(u, v, a: float):
    return add(u, scale(sub(v, u), a))


def norm(u):
    s = math.sqrt(sum(x * x for x in u)) or 1.0
    return [x / s for x in u]


def cos_sim(u, v):
    du = math.sqrt(sum(x * x for x in u)) or 1.0
    dv = math.sqrt(sum(x * x for x in v)) or 1.0
    return sum(a * b for a, b in zip(u, v)) / (du * dv)


def extract_keywords(text: str, k: int = 12):
    if not text:
        return []
    toks = [t.lower() for t in _WORD_RE.findall(text)]
    toks = [t for t in toks if t not in _STOP and len(t) >= 3]
    freq = Counter(toks)
    return [w for w, _ in freq.most_common(k)]


def sniff_dictionary_keys(raw_json: str, max_keys: int = 400):
    try:
        obj = json.loads(raw_json)
        if not isinstance(obj, dict):
            return []
        keys = list(obj.keys())
        random.shuffle(keys)
        return keys[:max_keys]
    except Exception:
        return []


__all__ = [
    "DIM",
    "rand_vec",
    "add",
    "sub",
    "scale",
    "lerp",
    "norm",
    "cos_sim",
    "extract_keywords",
    "sniff_dictionary_keys",
]
