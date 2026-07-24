"""Semantic system for managing agent semantic memory and associations."""

from __future__ import annotations

import random
import math
import re
import numpy as np
from config import DIMS
from agents.cognition.semantic_utils import add, sub, scale
from typing import Optional
class SemanticSystem:
    """
    Manages the semantic memory and associations of an agent.
    """
    def __init__(self, owner):
        self.owner = owner

        # Shared semantic container (some mixins/systems expect this to exist).
        if not hasattr(self.owner, "semantic") or not isinstance(getattr(self.owner, "semantic", None), dict):
            self.owner.semantic = {}
        self.owner.semantic.setdefault("concept_tokens", {})
        self.concept_tokens = self.owner.semantic["concept_tokens"]

        self.family_system = self.SemanticFamily(owner)
        # Back-compat: older code expects direct access
        self.families = self.family_system.families
        self.vectors = {}
        self.tokens = {}
        self.links = {}
        self.last_used = {}

        # Flavour channels and latent fields
        self.semantic_flavour = {}
        dim = self._semantic_dim()
        self.flavour_axes = {
            "objectness": np.random.normal(size=dim),
            "processness": np.random.normal(size=dim),
            "relationness": np.random.normal(size=dim),
            "transformness": np.random.normal(size=dim),
            "causativeness": np.random.normal(size=dim),
            "temporalness": np.random.normal(size=dim),
        }
        for k, v in list(self.flavour_axes.items()):
            n = np.linalg.norm(v) + 1e-9
            self.flavour_axes[k] = v / n

        self.flavour_gain = 0.02

        base_dim = dim
        self.latent_attractors = {
            "objectness": self._randvec(scale=1.0),
            "processness": self._randvec(scale=1.0),
            "relationness": self._randvec(scale=1.0),
            "transformness": self._randvec(scale=1.0),
            "causativeness": self._randvec(scale=1.0),
            "temporalness": self._randvec(scale=1.0),
        }

        # Predefine "why" token for reasoning
        self.vectors["why"] = self._randvec()
        self.owner.vocab.add("why")

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------
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
    

    # =====================================================
    # SEMANTIC NEIGHBORS + START TOKEN
    # =====================================================
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

    # -----------------------------------------------------
    # Flavour / latent attractors
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # Co-occurrence observation
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # Graph utilities (degree, linking, gravity, decay)
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # Link-based neighbour sampling
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # Teaching support: export + predict
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # Dictionary / seed integration
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # Similarity utilities
    # -----------------------------------------------------
    def similarity(self, a, b) -> float:
        """Cosine similarity between two semantic tokens (0.0 if missing)."""
        vecs = self.vectors
        if a not in vecs or b not in vecs:
            return 0.0
        va = vecs[a]
        vb = vecs[b]
        if not isinstance(va, (list, tuple)) or not isinstance(vb, (list, tuple)):
            return 0.0
        va = np.array(va, dtype=float)
        vb = np.array(vb, dtype=float)
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    def distance(self, a, b) -> float:
        """Semantic distance = 1 - similarity."""
        return 1.0 - float(self.similarity(a, b))

    def _family_neighbors(self, tok):
        tsem = self.tokens
        fams = self.family_system.families
        meta = tsem.get(tok)
        if not meta:
            return []

        out = set()
        for fid in meta.get("families", {}):
            fam = fams.get(fid)
            if not fam:
                continue
            for member in fam.get("members", {}).keys():
                if member != tok and not self.owner.identity_system._is_identity_like(member):
                    out.add(member)

        return list(out)

    # =====================================================
    # SOCIAL DIALOGUE HELPERS
    # =====================================================
    def _semantic_similarity_to(self, other_id):
        my_tok = getattr(self.owner, "identity_token", None)
        if not isinstance(my_tok, str):
            return 0.0
        other_tok = self.owner.identity_system.get_or_create_nickname_for_id(other_id)
        if other_tok not in self.vectors:
            return 0.0
        return 1.0 - self.owner.semantic_distance(my_tok, other_tok)
    
    # =====================================================
    # TOKENS
    # =====================================================
    class SemanticToken:
        """
        Represents a semantic token with its associated vector.
        """
        def __init__(self, owner):
            dims = DIMS
            self.vector = [random.uniform(-0.25, 0.25) for _ in range(dims)]
            self.usage_count = 0
            self.last_used = 0

        def _ensure_token_semantic(self, tok):
            tstore = self.owner.semantic_system.tokens
            if tok not in tstore:
                tstore[tok] = {
                    "families": {},
                    "usage_count": 0,
                    "birth_generation": getattr(self, "generation_index", 0),
                }
            return tstore[tok]

        
    # =====================================================
    # FAMILIES
    # =====================================================
    class SemanticFamily:
        """
        Represents a family of related semantic tokens.
        """
        def __init__(self, owner):
            self.owner = owner
            self.families = {}
            self.family_counter = 0

            sem = getattr(owner, "semantic", None)
            if isinstance(sem, dict):
                fams = sem.get("families")
                if isinstance(fams, dict):
                    self.families = fams
                fc = sem.get("family_counter")
                if isinstance(fc, int):
                    self.family_counter = fc

        def _create_family(self, centroid=None, parent_ids=None, ftype="emergent"):
            fid = f"F{self.family_counter}"
            self.family_counter += 1
            sem = getattr(self.owner, "semantic", None)
            if isinstance(sem, dict):
                sem["family_counter"] = int(self.family_counter)

            self.families[fid] = {
                "centroid": list(centroid) if centroid is not None else None,
                "members": {},
                "parents": set(parent_ids or []),
                "children": set(),
                "depth": 0,
                "age": 0,
                "type": ftype,
                "concept_tags": set(),
                "confidence": 0.2,
            }

            for p in (parent_ids or []):
                if p in self.families:
                    self.families[p]["children"].add(fid)

            return fid

        def _assign_token_family(self, tok, fid, strength=0.1, confidence=0.1):
            tsem = self.owner._ensure_token_semantic(tok)
            fam = self.families.get(fid)
            if fam is None:
                return
            tsem["families"][fid] = {
                "strength": float(strength),
                "confidence": float(confidence),
                "age": 0,
                "last_reinforced": getattr(self, "generation_index", 0),
            }
            fam["members"][tok] = float(strength)

        def detect_semantic_families(self, radius=0.4, min_members=4, max_families_per_gen=2):
            # sem = self.semantic
            vecs = self.owner.semantic_system.vectors
            tmeta = self.owner.semantic_system.tokens

            tokens = [t for t in vecs.keys()
                    if t not in self.owner.numeric_semantic.keys()
                    and t not in self.owner.semantic_system.concept_tokens.values()
                    and not self.owner._is_identity_like(t)]

            if len(tokens) < min_members:
                return

            visited = set()
            made = 0

            def sqdist_no0(a, b):
                return sum((a[i] - b[i])**2 for i in range(1, len(a)))

            for i, t in enumerate(tokens):
                if t in visited:
                    continue
                center = vecs[t]
                cluster = [t]
                visited.add(t)

                for u in tokens[i+1:]:
                    if u in visited:
                        continue
                    if sqdist_no0(center, vecs[u])**0.5 <= radius:
                        cluster.append(u)
                        visited.add(u)

                if len(cluster) < 4:
                    continue

                dim = len(center)
                cv = [0.0]*dim
                for tok in cluster:
                    v = vecs[tok]
                    for d in range(dim):
                        cv[d] += v[d]
                cv = [x/len(cluster) for x in cv]

                vari = 0.0
                for tok in cluster:
                    v = vecs[tok]
                    vari += sum((v[d]-cv[d])**2 for d in range(1, dim))
                vari /= len(cluster)

                usage_vals = [tmeta[tok]["usage_count"] for tok in cluster if tok in tmeta]
                avg_usage = sum(usage_vals)/len(usage_vals) if usage_vals else 0

                dists = []
                for tok in cluster:
                    d = (sqdist_no0(vecs[tok], cv) ** 0.5)
                    dists.append(d)
                cohesion = 1.0 - (sum(dists)/len(dists))

                if vari > 0.015:
                    continue
                if avg_usage < 20:
                    continue
                if cohesion < 0.65:
                    continue

                size_term = min(1.0, len(cluster)/12)
                freq_term = min(1.0, avg_usage/50)
                var_term  = max(0.0, 1.0 - 8*vari)
                raw = 0.3*size_term + 0.3*cohesion + 0.2*freq_term + 0.2*var_term
                conf = min(0.92, raw)

                fid = self._create_family(centroid=cv)
                self.families[fid]["confidence"] = conf

                for tok in cluster:
                    self._assign_token_family(tok, fid, strength=0.4, confidence=conf)

                made += 1
                if made >= max_families_per_gen:
                    break

        def family_reinforcement_update(self, drift=0.02):
            vecs = self.owner.semantic_system.vectors
            fams = self.families
            tokens = self.owner.semantic_system.tokens

            for fid, fam in fams.items():
                centroid = fam.get("centroid")
                if centroid is None:
                    continue
                for tok, strength in fam["members"].items():
                    if tok not in vecs or tok not in tokens:
                        continue
                    v = vecs[tok]
                    newv = [
                        a + drift * strength * (c - a)
                        for a, c in zip(v, centroid)
                    ]
                    vecs[tok] = newv

        def family_soft_decay(self, decay=0.003, min_strength=0.02):
            fams = self.families
            tokens = self.owner.semantic_system.tokens
            for fid, fam in fams.items():
                members = fam.get("members", {})
                for tok, strength in list(members.items()):
                    new_s = strength - decay
                    if new_s <= min_strength:
                        del members[tok]
                        if tok in tokens and fid in tokens[tok].get("families", {}):
                            del tokens[tok]["families"][fid]
                    else:
                        members[tok] = new_s
                        if tok in tokens and fid in tokens[tok].get("families", {}):
                            tokens[tok]["families"][fid]["strength"] = new_s

        # =====================================================
        # FAMILY GOSSIP
        # =====================================================
        def name_families(self):
            fams = self.families
            for fid, fam in fams.items():
                if fam.get("name") is None:
                    fam["name"] = self.owner._invent_token(prefix="f", concept=True)

        def export_family_snapshot(self, max_families=3):
            fams = self.families
            if not fams:
                return []
            ordered = sorted(
                fams.items(),
                key=lambda kv: kv[1].get("confidence", 0.0),
                reverse=True,
            )
            out = []
            for fid, fam in ordered[:max_families]:
                c = fam.get("centroid")
                if not c:
                    continue
                out.append({
                    "id": fid,
                    "name": fam.get("name"),
                    "centroid": list(c),
                    "confidence": float(fam.get("confidence", 0.2)),
                    "size": len(fam.get("members", {})),
                })
            return out

        def maybe_broadcast_families(self):
            o = self.owner
            api = getattr(o, "api", None)
            if api is None:
                return

            sem = getattr(o, "semantic", None) or {}
            fams = sem.get("families") or self.families
            if not fams:
                return
            ordered = sorted(
                fams.items(),
                key=lambda kv: kv[1].get("confidence", 0.0),
                reverse=True,
            )[:3]

            for fid, fam in ordered:
                conf = fam.get("confidence", 0.2)
                p = min(0.05 + 0.3 * conf, 0.25)
                if random.random() > p:
                    continue
                if fam.get("name") is None:
                    if hasattr(o, "_invent_token"):
                        fam["name"] = o._invent_token(prefix="f", concept=True)
                    else:
                        fam["name"] = f"f{fid.lower()}"
                c = fam.get("centroid")
                if not c:
                    continue
                c_str = ",".join(f"{x:.3f}" for x in c)
                line = (
                    f"A{o.id} teach_family name={fam['name']} "
                    f"conf={conf:.3f} centroid={c_str}\n"
                )
                try:
                    api.append_text("/family_gossip.txt", line, scope="world")
                except Exception:
                    pass

        def _integrate_family_gossip_line(self, line):
            if "teach_family" not in line:
                return
            name = None
            conf = 0.2
            centroid = None
            m = re.search(r"name=([^\s]+)", line)
            if m:
                name = m.group(1)
            m = re.search(r"conf=([0-9]*\.?[0-9]+)", line)
            if m:
                conf = float(m.group(1))
            m = re.search(r"centroid=([\-0-9\.,]+)", line)
            if m:
                vals = []
                for p in m.group(1).split(","):
                    p = p.strip()
                    if p:
                        try:
                            vals.append(float(p))
                        except ValueError:
                            pass
                if vals:
                    centroid = vals
            if centroid is None:
                return
            foreign = {
                "name": name,
                "centroid": centroid,
                "confidence": conf,
            }
            self.integrate_foreign_family(foreign)

        def integrate_foreign_family(self, foreign):
            fams = self.families
            if not fams:
                return
            centroid = foreign.get("centroid")
            if not centroid:
                return
            merge_radius = 0.30
            influence_radius = 0.55
            best_fid = None
            best_d2 = None

            def sqdist(a, b):
                return sum((x - y)**2 for x, y in zip(a, b))

            for fid, fam in fams.items():
                c = fam.get("centroid")
                if not c:
                    continue
                d2 = sqdist(c, centroid)
                if best_d2 is None or d2 < best_d2:
                    best_fid = fid
                    best_d2 = d2

            if best_fid is None:
                return

            dist = best_d2 ** 0.5
            fam = fams[best_fid]
            if dist <= merge_radius:
                new_c = [
                    0.8 * lc + 0.2 * fc
                    for lc, fc in zip(fam["centroid"], centroid)
                ]
                fam["centroid"] = new_c
                fam["confidence"] = min(1.0, fam.get("confidence", 0.3) + 0.1)
                return
            if dist <= influence_radius:
                parents = fam.setdefault("parents", [])
                parents.append({
                    "name": foreign.get("name"),
                    "confidence": foreign.get("confidence", 0.2),
                })
                fam["confidence"] = min(1.0, fam.get("confidence", 0.3) + 0.02)
                return

        def prune_families(self, limit=250):
            fams = self.families
            if len(fams) <= limit:
                return
            ranked = sorted(
                fams.items(),
                key=lambda kv: (
                    kv[1].get("confidence", 0.0),
                    len(kv[1].get("members", {}))
                )
            )
            remove_n = int(len(fams) * 0.20)
            for fid, _ in ranked[:remove_n]:
                del fams[fid]
