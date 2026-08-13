"""Semantic system for managing agent semantic memory and associations."""

from __future__ import annotations

import random
import math
import re
import numpy as np
from config import DIMS
from agents.cognition.semantic_utils import add, sub, scale
from typing import Optional


class SemanticAssociationMixin:
    def observe_tokens(self, tokens, gain=None):
        """Main co-occurrence learning step over a token sequence."""
        if not tokens:
            return

        owner = self.owner

        # HARD NUMERIC / IDENTITY ISOLATION
        cleaned = []
        for t in tokens:
            if hasattr(owner, "is_numeric_token") and owner.is_numeric_token(t):
                continue
            if hasattr(owner, "is_identity_token") and owner.is_identity_token(t):
                continue
            cleaned.append(t)

        if len(cleaned) < 2:
            return

        # Flavour update inside semantic system
        self.update_token_flavour(cleaned, context_gain=gain)

        numeric_sem = getattr(owner, "numeric_semantic", {})
        raw = list(cleaned)
        filtered = [t for t in raw if isinstance(t, str) and t not in numeric_sem]
        if len(filtered) > 15:
            filtered = random.sample(filtered, 15)

        # Ensure shared store exists and compute effective gain
        if hasattr(owner, "_ensure_semantic"):
            owner._ensure_semantic()
        sem = getattr(owner, "semantic", {})
        base_gain = sem.get("cooccur_gain", 0.10)
        gain = base_gain if gain is None else gain

        # ensure vecs + recency + optional flavour drift
        for t in filtered:
            self.ensure_vec(t)
            vec = self.vectors.get(t)
            if vec is not None:
                self.vectors[t] = self.apply_flavour_drift(t, vec)
            if hasattr(owner, "recent_tokens"):
                owner.recent_tokens.append(t)

        # bloom noise for non-numeric, non-identity tokens
        numeric_sem = getattr(owner, "numeric_semantic", {})
        for t in filtered:
            if t in numeric_sem:
                continue
            if hasattr(owner, "is_identity_token") and owner.is_identity_token(t):
                continue
            v = self.vectors.get(t)
            if v is None:
                continue
            noise = [
                random.uniform(-0.02, 0.02) if i == 0
                else random.uniform(-0.15, 0.15)
                for i, _ in enumerate(v)
            ]
            self.vectors[t] = [a + b for a, b in zip(v, noise)]

        # pairwise links
        n = len(filtered)
        for i in range(n):
            for j in range(i + 1, n):
                self.link(filtered[i], filtered[j], gain)

        # trim recency
        if hasattr(owner, "recent_tokens"):
            if len(owner.recent_tokens) > 64:
                owner.recent_tokens = owner.recent_tokens[-64:]

        # identity grounding
        if hasattr(owner, "identity_system"):
            for t in filtered:
                if isinstance(t, str) and t.startswith("a") and t[1:].isdigit():
                    owner.identity_system.mark_identity_token(t)

    def _true_degree(self, tok):
        row = self.links.get(tok, {})
        deg = 0
        for entry in row.values():
            if isinstance(entry, dict):
                if abs(entry.get("w", 0.0)) > 1e-6:
                    deg += 1
            else:
                if abs(entry) > 1e-6:
                    deg += 1
        return deg

    def _compute_usefulness(self, a, b):
        """Compute how 'useful' the link a <-> b is.

        Uses:
        - vector similarity
        - novelty vs neighbourhood
        - communication frequency (utterance usage_count)
        - existing link weight
        """

        vecs = self.vectors
        links = self.links
        deg = len(links.get(a, {}))
        deg_penalty = 1.0 / (1.0 + deg)

        va = vecs[a]
        vb = vecs[b]
        sim = float(np.dot(va, vb) /
                    (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))

        neigh = self._semantic_neighbors(a, k=2)
        valid_neigh = [n for n in neigh if n in vecs]
        if len(valid_neigh) < 1:
            novelty = 0.0
        else:
            try:
                mean_vec = np.mean([vecs[n] for n in valid_neigh], axis=0)
                novelty = 1.0 / (1.0 + np.linalg.norm(np.array(vecs[a]) - mean_vec))
            except Exception:
                novelty = 0.0

        usage = 1
        um = getattr(self.owner, "utterance_memory", None)
        if isinstance(um, dict):
            usage = um.get("usage_count", {}).get(b, 1)
        comm_factor = math.log1p(usage)

        link_entry = links.get(a, {}).get(b)
        if isinstance(link_entry, dict):
            w = link_entry.get("w", 0.0)
        else:
            w = float(link_entry or 0.0)

        return (
            0.50 * sim +
            0.30 * novelty +
            0.20 * comm_factor
        ) * (1.0 + 0.25 * w) * deg_penalty

    def link(self, a, b, w):
        """Bidirectional semantic link with numeric exclusion & degree cap."""

        if not a or not b or a == b:
            return

        owner = self.owner
        if hasattr(owner, "is_numeric_token") and (
            owner.is_numeric_token(a) or owner.is_numeric_token(b)
        ):
            return

        self.ensure_vec(a)
        self.ensure_vec(b)

        links = self.links
        row_a = links.setdefault(a, {})
        row_b = links.setdefault(b, {})

        now = getattr(owner, "current_generation", 0)
        max_degree = 15

        deg = self._true_degree(a)
        if deg >= max_degree:
            candidates = [
                (k, v) for k, v in row_a.items()
                if isinstance(v, dict) and abs(v.get("w", 0.0)) > 1e-6
            ]
            if not candidates:
                return

            new_use = self._compute_usefulness(a, b)
            weakest_key, weakest_data = min(
                candidates,
                key=lambda kv: kv[1].get("use", 0.0)
            )
            weakest_use = weakest_data.get("use", 0.0)
            if new_use <= weakest_use:
                return

            row_a.pop(weakest_key, None)
            if weakest_key in links:
                links[weakest_key].pop(a, None)

        if b not in row_a:
            row_a[b] = {"w": 0.0, "use": 0.0, "last": now}
        if a not in row_b:
            row_b[a] = {"w": 0.0, "use": 0.0, "last": now}

        row_a[b]["w"] += w
        row_b[a]["w"] += w

        row_a[b]["use"] = self._compute_usefulness(a, b)
        row_b[a]["use"] = self._compute_usefulness(b, a)

        row_a[b]["last"] = now
        row_b[a]["last"] = now

    def decay_links(self, decay: float = 0.995, prune_threshold: float = 1e-4):
        """Reduce old link weights and prune tiny edges."""
        links = self.links
        for a, nbrs in list(links.items()):
            for b, entry in list(nbrs.items()):
                if isinstance(entry, dict):
                    w = entry.get("w", 0.0)
                else:
                    w = float(entry)
                new_w = w * decay
                if abs(new_w) < prune_threshold:
                    del nbrs[b]
                    continue
                if isinstance(entry, dict):
                    entry["w"] = new_w
                    entry["age"] = entry.get("age", 0) + 1
                    nbrs[b] = entry
                else:
                    nbrs[b] = new_w
            if not nbrs:
                del links[a]

    def gravity_step(self, gravity: float):
        """Pull linked tokens together one step using weighted edges."""
        vecs = self.vectors
        links = self.links

        scored = []
        for tok, nbrs in links.items():
            if not nbrs:
                continue
            total = 0.0
            for entry in nbrs.values():
                if isinstance(entry, dict):
                    total += abs(entry.get("w", 0.0))
                else:
                    total += abs(entry)
            scored.append((tok, total))

        K = 12
        top_tokens = [
            tok for tok, score in sorted(scored, key=lambda x: x[1], reverse=True)[:K]
        ]

        for a in top_tokens:
            nbrs = links.get(a, {})
            va = vecs.get(a)
            if va is None:
                continue
            for b, entry in nbrs.items():
                if isinstance(entry, dict):
                    w = entry.get("w", 0.0)
                else:
                    w = float(entry or 0.0)
                if w == 0:
                    continue
                vb = vecs.get(b)
                if vb is None:
                    continue
                dv = sub(vb, va)
                new_va = add(va, scale(dv, gravity * w))
                vecs[a] = new_va
                va = new_va

        self.decay_links()

    def pick_neighbor(self, token):
        """Weighted-random neighbor based on link strength."""
        nbrs = self.links.get(token, {})
        if not nbrs:
            return None

        words = list(nbrs.keys())
        weights = []
        for v in nbrs.values():
            if isinstance(v, dict):
                w = v.get("w", 0.0)
            else:
                w = float(v or 0.0)
            weights.append(max(1e-6, w))
        return random.choices(words, weights=weights)[0]

    def export_bundle(self, max_keys: int = 3):
        """Export a small semantic packet for teaching."""
        vecs = self.vectors
        if not vecs:
            return None
        keys = list(vecs.keys())
        random.shuffle(keys)
        chosen = keys[:max_keys]
        return {
            "tokens": chosen,
            "vecs": {t: vecs[t] for t in chosen},
        }

    def attempt_prediction(self, bundle):
        """Synthesize a predicted semantic vector from a bundle."""
        if not bundle:
            return None
        toks = bundle.get("tokens", [])
        if not toks:
            return None

        preds = []
        for t in toks:
            if t not in self.vectors:
                curiosity = getattr(self.owner, "traits", {}).get("curiosity", 0.5)
                if random.random() < curiosity:
                    self.vectors[t] = self._randvec(scale=1.0)
                else:
                    continue
            preds.append(self.vectors[t])

        if not preds:
            return None

        acc = np.array(preds[0], copy=True)
        for v in preds[1:]:
            acc = add(acc, v)
        return scale(acc, 1.0 / len(preds))

    def receive_semantic_seeds(self, seeds):
        """Integrate external semantic seeds (dictionary-style) into memory.

        Expected format:
            seeds = {
                "words":    [...],           # word strings
                "synonyms": [(a, b), ...],  # positive links
                "antonyms": [(a, b), ...],  # negative links
            }
        """
        if not seeds or not isinstance(seeds, dict):
            return

        words = seeds.get("words") or []
        syns = seeds.get("synonyms") or []
        ants = seeds.get("antonyms") or []

        # Ensure vectors exist for all words
        for w in words:
            if not isinstance(w, str) or not w.strip():
                continue
            self.ensure_vec(w)

        # Link synonyms with positive weight
        for a, b in syns:
            if not a or not b:
                continue
            self.link(a, b, +0.10)

        # Link antonyms with negative weight (inverse pull)
        for a, b in ants:
            if not a or not b:
                continue
            self.link(a, b, -0.10)
