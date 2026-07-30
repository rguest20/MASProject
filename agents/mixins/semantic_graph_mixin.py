# agents/mixins/semantic_mixin.py

import random
import numpy as np
from collections import defaultdict

from agents.cognition.semantic_utils import (
    rand_vec, add, sub, scale, cos_sim,
)
from agents.agent_constants import SYLLABLES


class SemanticGraphMixin:
    def semantic_drift_update(self):
        """
        Default drift update for lexical semantics.
        LanguageMixinV2 extends this in its own layer (families, tokens).
        If that mixin is not present, this prevents coordinator crashes.
        """
        # Basic lexical drift is already handled by _gravity_step()
        # so here we simply do nothing.
        pass

    def _ensure_semantic(self):
        """Ensures semantic system exists (mixin safety)."""
        if not hasattr(self, "semantic") or self.semantic is None:
            self._init_semantic_system()

        sem = self.semantic

        # Make sure core fields exist and are correctly typed
        if "links" not in sem or not isinstance(sem["links"], defaultdict):
            sem["links"] = defaultdict(lambda: defaultdict(float))

        if "vecs" not in sem:
            sem["vecs"] = {}

    def _ensure_vec(self, tok):
        if hasattr(self, "semantic_system"):
            self.semantic_system.ensure_vec(tok)
            return

        self._ensure_semantic()
        vecs = self.semantic["vecs"]
        if tok not in vecs:
            vecs[tok] = self._clip_semantic_vec(rand_vec())

    def is_numeric_token(self, tok):
        if hasattr(self, "numeric_system"):
            inv = getattr(self.numeric_system, "inverse", {})
            return tok in inv or tok in getattr(self.numeric_system, "inverse_symbol_map", {})
        if hasattr(self, "token_registry") and self.token_registry:
            return self.token_registry.is_numeric(tok)
        return False

    def _true_degree(self, tok):
        if hasattr(self, "semantic_system"):
            return self.semantic_system._true_degree(tok)

        row = self.semantic["links"].get(tok, {})
        deg = 0
        for entry in row.values():
            if isinstance(entry, dict):
                if abs(entry.get("w", 0.0)) > 1e-6:
                    deg += 1
            else:
                if abs(entry) > 1e-6:
                    deg += 1
        return deg

    def _link(self, a, b, w):
        """
        Bidirectional semantic link with:
        - hard numeric exclusion
        - strict degree cap
        - safe eviction
        - symmetric maintenance
        """

        if hasattr(self, "semantic_system"):
            return self.semantic_system.link(a, b, w)

        # Fallback to legacy in-place logic if semantic_system is missing
        if not a or not b or a == b:
            return
        if self.is_numeric_token(a) or self.is_numeric_token(b):
            return
        self._ensure_semantic()
        self._ensure_vec(a)
        self._ensure_vec(b)
        links = self.semantic["links"]
        row_a = links[a]
        row_b = links[b]
        now = getattr(self, "current_generation", 0)
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

    def _compute_usefulness(self, a, b):
        """
        Compute how 'useful' the link a <-> b is, using only structures this
        project actually maintains:
        - vector similarity
        - novelty vs neighbourhood
        - communication frequency (utterance usage_count)
        - existing link weight
        """

        if hasattr(self, "semantic_system"):
            return self.semantic_system._compute_usefulness(a, b)

        vecs = self.semantic["vecs"]
        links = self.semantic["links"]
        deg = len(self.semantic["links"].get(a, {}))
        deg_penalty = 1.0 / (1.0 + deg)

        # ------------------------------
        # 1. Similarity between vectors
        # ------------------------------
        va = vecs[a]
        vb = vecs[b]

        sim = float(np.dot(va, vb) /
                    (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))

        # ------------------------------
        # 2. Novelty: difference from neighbour mean
        # ------------------------------
        neigh = self.semantic_system._semantic_neighbors(a, k=2)

        # --- Filter out neighbours missing vectors ---
        valid_neigh = [n for n in neigh if n in vecs]

        if len(valid_neigh) < 1:
            novelty = 0.0
        else:
            try:
                mean_vec = np.mean([vecs[n] for n in valid_neigh], axis=0)
                novelty = 1.0 / (1.0 + np.linalg.norm(np.array(vecs[a]) - mean_vec))
            except Exception:
                novelty = 0.0

        # ------------------------------
        # 3. Communication salience
        # ------------------------------
        usage = 1
        if hasattr(self, "utterance_memory"):
            usage = self.utterance_memory.get("usage_count", {}).get(b, 1)

        comm_factor = np.log1p(usage)  # smooth growth

        # ------------------------------
        # 4. Current link weight
        # ------------------------------
        link_entry = links[a].get(b)
        if isinstance(link_entry, dict):
            w = link_entry.get("w", 0.0)
        else:
            w = float(link_entry or 0.0)

        # ------------------------------
        # Combine
        # ------------------------------
        return (
            0.50 * sim +
            0.30 * novelty +
            0.20 * comm_factor
        ) * (1.0 + 0.25 * w) * deg_penalty

    def _gravity_step(self):
        """Pull linked tokens together each tick."""
        self._ensure_semantic()
        g = self.semantic["gravity"]
        if hasattr(self, "semantic_system"):
            return self.semantic_system.gravity_step(g)

        # Fallback legacy behaviour if semantic_system is missing
        vecs = self.semantic["vecs"]
        links = self.semantic["links"]
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
            nbrs = links[a]
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
                new_va = add(va, scale(dv, g * w))
                vecs[a] = new_va
                va = new_va
        self._decay_links()

    def _decompose_token_for_semantics(self, tok):
        """
        Carefully decompose a token into known syllables for semantic learning.

        Design:
          - Never decompose identity-like tokens.
          - If token is itself a base syllable, keep it as-is.
          - Try greedy longest-prefix decomposition using SYLLABLES.
          - If we cannot fully cover the token, fall back to [tok] to avoid
            destroying potential emergent words.
        """
        # Only strings are decomposed
        if not isinstance(tok, str):
            return [tok]

        # Preserve identity tokens as atomic
        if hasattr(self, "is_identity_token") and self.is_identity_token(tok):
            return [tok]

        # We might not want to rely on a global if something is weird
        base_sylls = list(SYLLABLES)
        if not base_sylls:
            return [tok]

        # If it is exactly a syllable, don’t split it
        if tok in base_sylls:
            return [tok]

        # Greedy longest-first decomposition
        base_sylls.sort(key=len, reverse=True)
        remaining = tok
        parts = []

        while remaining:
            matched = False
            for s in base_sylls:
                if remaining.startswith(s):
                    parts.append(s)
                    remaining = remaining[len(s):]
                    matched = True
                    break
            if not matched:
                parts = [tok]
                break

        # If decomposition is trivial (1 part), just keep token as-is
        if len(parts) <= 1:
            return [tok]
        return [tok]

    def observe_utterance(self, utter, gain_scale=1.0):
        tokens = self._decompose_token_for_semantics(utter)
        
        for tok in tokens:
            self.stab_note_token_usage(tok)
        
        gain = self.semantic.get("cooccur_gain", 0.10) * gain_scale
        self._observe_tokens(tokens, gain=gain)

    def _observe_tokens(self, tokens, gain=None):
        if hasattr(self, "semantic_system"):
            return self.semantic_system.observe_tokens(tokens, gain=gain)

        if not tokens:
            return

        # Legacy path retained as fallback (mirrors SemanticSystem.observe_tokens)
        tokens = [
            t for t in tokens
            if not self.is_numeric_token(t)
            and not self.is_identity_token(t)
        ]
        if len(tokens) < 2:
            return
        self._update_token_flavour(tokens, context_gain=gain)
        numeric_sem = getattr(self, "numeric_semantic", {})
        raw = list(tokens)
        filtered = [t for t in raw if isinstance(t, str) and t not in numeric_sem]
        if len(filtered) > 15:
            filtered = random.sample(filtered, 15)
        self._ensure_semantic()
        gain = self.semantic["cooccur_gain"] if gain is None else gain
        for t in filtered:
            self._ensure_vec(t)
            vec = self.semantic["vecs"][t]
            self.semantic["vecs"][t] = self._apply_flavour_drift(t, vec)
            self.recent_tokens.append(t)
        numeric_sem = getattr(self, "numeric_semantic", {})
        for t in filtered:
            if t in numeric_sem:
                continue
            if hasattr(self, "is_identity_token") and self.is_identity_token(t):
                continue
            v = self.semantic["vecs"][t]
            noise = [
                random.uniform(-0.02, 0.02) if i == 0
                else random.uniform(-0.15, 0.15)
                for i, _ in enumerate(v)
            ]
            self.semantic["vecs"][t] = [a + b for a, b in zip(v, noise)]
        for i in range(len(filtered)):
            for j in range(i + 1, len(filtered)):
                self._link(filtered[i], filtered[j], gain)
        if len(self.recent_tokens) > 64:
            self.recent_tokens = self.recent_tokens[-64:]
        if hasattr(self, "identity_system"):
            for t in filtered:
                if t.startswith("a") and t[1:].isdigit():
                    self.identity_system.mark_identity_token(t)

    def _decay_links(self, decay=0.995, prune_threshold=1e-4):
        """
        Reduce old link weights and prune tiny edges.

        Works for both:
            - old format: nbrs[b] = float
            - new format: nbrs[b] = {"w": float, "use": float, "age": int}
        """
        if hasattr(self, "semantic_system"):
            return self.semantic_system.decay_links(decay=decay, prune_threshold=prune_threshold)

        links = self.semantic["links"]
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

    def receive_semantic_seeds(self, seeds):
        """Bridge dictionary-style semantic seeds into SemanticSystem.

        Delegates to SemanticSystem.receive_semantic_seeds when available,
        falling back to the legacy in-mixin implementation otherwise.
        """

        if hasattr(self, "semantic_system"):
            return self.semantic_system.receive_semantic_seeds(seeds)

        # Legacy fallback using local semantic store
        if not seeds or not isinstance(seeds, dict):
            return

        words = seeds.get("words") or []
        syns = seeds.get("synonyms") or []
        ants = seeds.get("antonyms") or []

        for w in words:
            self._ensure_vec(w)

        for a, b in syns:
            self._link(a, b, +0.10)

        for a, b in ants:
            self._link(a, b, -0.10)
