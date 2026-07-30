"""Semantic system for managing agent semantic memory and associations."""

from __future__ import annotations

import random
import math
import re
import numpy as np
from config import DIMS
from agents.cognition.semantic_utils import add, sub, scale
from typing import Optional


class SemanticVectorMixin:
    def ensure_language_fields(self):
        """Ensure language-related semantic fields exist on the shared store."""
        owner = self.owner
        if hasattr(owner, "_ensure_semantic"):
            owner._ensure_semantic()
        sem = getattr(owner, "semantic", None)
        if not isinstance(sem, dict):
            owner.semantic = {}
            sem = owner.semantic

        sem.setdefault("concept_tokens", {})
        self.concept_tokens = sem["concept_tokens"]
        sem.setdefault("rel_family", {})
        sem.setdefault("ref_family", {})

        if not hasattr(owner, "numeric_semantic") or not isinstance(owner.numeric_semantic, dict):
            # token -> anchor vector (used for filtering + numeric reasoning)
            owner.numeric_semantic = {}

    def ensure_numeric_token(self, tok: str, *, digit: Optional[int] = None, base: Optional[int] = None):
        """Register a numeric token so semantic learning can treat it specially.

        - Ensures a stable numeric anchor vector exists (owner.numeric_semantic[tok])
        - Ensures the semantic vector exists (self.vectors[tok])
        """
        if not isinstance(tok, str) or not tok.strip():
            return
        self.ensure_language_fields()

        owner = self.owner
        tok = tok.strip().lower()

        if tok not in self.vectors:
            self.ensure_vec(tok)

        if tok in owner.numeric_semantic:
            return

        try:
            digit = int(digit) if digit is not None else None
        except Exception:
            digit = None

        try:
            base = int(base) if base is not None else None
        except Exception:
            base = None

        if digit is None:
            digit = 0
        if base is None or base <= 1:
            base = 10

        dim = self._semantic_dim()
        anchor = [(digit / max(1, base - 1)) * 0.2] + [0.0] * (dim - 1)
        anchor = [x + random.uniform(-0.02, 0.02) for x in anchor]

        owner.numeric_semantic[tok] = list(anchor)
        self.vectors[tok] = list(anchor)

    def _ensure_vec(self, tok, scale=0.25):
        vecs = self.vectors
        if tok not in vecs:
            vecs[tok] = self._randvec(scale=scale)
        return vecs[tok]

    def _l_ensure_vec(self, tok, scale=0.25):
        return self._ensure_vec(tok, scale)

    def _r_ensure_vec(self, tok, scale=0.25):
        return self._ensure_vec(tok, scale)

    def _sanitize_vector_dims(self):
        dim = self._semantic_dim()
        vecs = self.vectors
        for tok, v in list(vecs.items()):
            if not isinstance(v, (list, tuple)):
                vecs[tok] = self._randvec()
                continue
            if len(v) != dim:
                new_v = list(v[:dim])
                if len(new_v) < dim:
                    new_v += [random.uniform(-0.1, 0.1) for _ in range(dim - len(new_v))]
                vecs[tok] = new_v

    def _centroid(self, vecs):
        vecs = [v for v in vecs if v]
        if not vecs:
            dim = self._semantic_dim()
            return [0.0] * dim
        dim = len(vecs[0])
        return [
            sum(v[d] for v in vecs) / len(vecs)
            for d in range(dim)
        ]

    def _semantic_dim(self):
        return getattr(self, "semantic_dim", 32)

    def _randvec(self, scale=1.0):
        dim = self._semantic_dim()
        return [random.uniform(-scale, scale) for _ in range(dim)]

    def clip_vector(self, vec, max_norm: float = 5.0):
        """Safety clamp for semantic vectors.

        Keeps norms bounded and sanitises NaNs/Infs.
        """
        if not isinstance(vec, (list, tuple)):
            return vec

        v = np.array(vec, dtype=float)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

        n = np.linalg.norm(v)
        if n == 0.0:
            return self._randvec(scale=1.0)

        if n > max_norm:
            v = v * (max_norm / n)

        return v.tolist()

    def ensure_vec(self, tok, scale: float = 0.25):
        """Ensure a semantic vector exists for a token and return it."""
        vecs = self.vectors
        if tok not in vecs:
            vecs[tok] = self.clip_vector(self._randvec(scale=scale))
        return vecs[tok]

    def _semantic_neighbors(self, tok, k=8, max_radius=None):
        vecs = self.vectors
        
        if tok not in vecs:
            return []

        v = vecs[tok]
        if not isinstance(v, (list, tuple)) or not v:
            return []

        dim = len(v)
        candidates = []

        for other, w in vecs.items():
            if other == tok:
                continue
            if self.owner.identity_system._is_identity_like(other):
                continue
            if hasattr(self.owner, "vocab") and other not in self.owner.vocab:
                continue
            if not isinstance(w, (list, tuple)) or not w:
                continue

            d2 = 0.0
            m = min(dim, len(w))
            for i in range(1, m):
                dv = v[i] - w[i]
                d2 += dv * dv

            candidates.append((other, d2))

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[1])
        out = []
        for other, d2 in candidates:
            if max_radius is not None and (d2 ** 0.5) > max_radius:
                continue
            out.append(other)
            if len(out) >= k:
                break
        return out

    def _ensure_flavour_entry(self, tok):
        if tok not in self.semantic_flavour:
            self.semantic_flavour[tok] = {
                "objectness": 0.0,
                "processness": 0.0,
                "relationness": 0.0,
                "transformness": 0.0,
                "causativeness": 0.0,
                "temporalness": 0.0,
            }
        return self.semantic_flavour[tok]

    def _get_flavour_attractor(self, channel, dim=None):
        base = self.latent_attractors.get(channel)
        if base is None or not isinstance(base, (list, tuple)):
            return None
        v = list(base)
        if dim is None:
            return v
        if len(v) > dim:
            return v[:dim]
        if len(v) < dim:
            extra = [random.uniform(-0.01, 0.01) for _ in range(dim - len(v))]
            v.extend(extra)
        return v

    def update_token_flavour(self, tokens, context_gain=0.1):
        if context_gain is None:
            context_gain = 1.0
        for i, tok in enumerate(tokens):
            entry = self._ensure_flavour_entry(tok)
            entry["objectness"] += 0.05 * context_gain
            if isinstance(tok, str) and tok.startswith("rel"):
                entry["relationness"] += 0.15 * context_gain
            if i > 0:
                prev = tokens[i - 1]
                if isinstance(prev, str) and (prev.startswith("tol") or tok.startswith("tol")):
                    entry["processness"] += 0.12 * context_gain
                if isinstance(tok, str) and tok.startswith("muk"):
                    entry["transformness"] += 0.10 * context_gain
            if tok == "why":
                entry["causativeness"] += 0.15 * context_gain
            if isinstance(tok, str) and (tok.endswith("su") or tok.endswith("rin")):
                entry["temporalness"] += 0.05 * context_gain
            total = sum(entry.values())
            if total > 0:
                for k in entry:
                    entry[k] /= total

    def apply_flavour_drift(self, tok, vec, lr=0.01):
        f = self.semantic_flavour.get(tok)
        if not f:
            return vec
        axes = self.flavour_axes
        newv = list(vec)
        for flavour, weight in f.items():
            axis = axes.get(flavour)
            if axis is None or weight <= 0.0:
                continue
            for i in range(len(newv)):
                newv[i] += lr * weight * axis[i]
        return self.clip_vector(newv)

    def apply_flavour_attractors(self):
        if not self.semantic_flavour:
            return
        owner_sem = getattr(self.owner, "semantic", {})
        gain = float(owner_sem.get("flavour_gain", self.flavour_gain))
        if gain <= 0.0:
            return
        vecs = self.vectors
        for tok, flavour in list(self.semantic_flavour.items()):
            v = vecs.get(tok)
            if not isinstance(v, (list, tuple)):
                continue
            dim = len(v)
            pull = [0.0] * dim
            total_weight = 0.0
            for channel, w in flavour.items():
                if w <= 0.01:
                    continue
                att = self._get_flavour_attractor(channel, dim=dim)
                if att is None:
                    continue
                total_weight += w
                for i in range(dim):
                    pull[i] += w * (att[i] - v[i])
            if total_weight <= 0.0:
                continue
            step_scale = gain / max(1e-6, total_weight)
            new_v = [
                vi + step_scale * pi
                for vi, pi in zip(v, pull)
            ]
            vecs[tok] = new_v
