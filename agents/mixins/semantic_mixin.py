# agents/mixins/semantic_mixin.py

import random
import numpy as np
from collections import defaultdict
import math

from agents.semantics import (
    rand_vec, add, sub, scale, norm, cos_sim,
    extract_keywords, sniff_dictionary_keys, DIM
)
from agents.agent_constants import SYLLABLES

DEBUG_SEM = True
MAX_GLOBAL_DEGREE = 20


class SemanticMixin:
    """
    Lexical semantic system:
      - token vectors (lexical layer)
      - co-occurrence graph
      - gravity drift
      - dictionary integration
      - semantic neighbour selection
      - bundle export + prediction
      - optional link decay & cleanup

    NOTE:
      This mixin assumes a shared `self.semantic` dict that may also
      be extended by higher-level systems (e.g. LanguageMixinV2) for
      families/concepts. It *never* overwrites `self.semantic`, only
      fills in lexical fields if missing.
    """

    # ============================================================
    # INITIALISATION
    # ============================================================
    def _init_semantic_system(self):
        """
        Initialise or augment the shared semantic store.
        Does NOT overwrite an existing `self.semantic` created by
        other mixins; instead, it fills in lexical-related keys.
        """
        # Ensure shared container exists
        if not hasattr(self, "semantic") or self.semantic is None:
            self.semantic = {}

        sem = self.semantic

        # Core lexical fields
        if "links" not in sem or not isinstance(sem["links"], defaultdict):
            sem["links"] = defaultdict(lambda: defaultdict(float))

        sem.setdefault("vecs", {})
        sem.setdefault("learning_rate", 0.05)
        sem.setdefault("gravity", 0.008)
        sem.setdefault("cooccur_gain", 0.10)
        sem.setdefault("dictionary_gain", 0.20)

        # --- SOFT SEMANTIC SEEDING: LATENT ATTRACTORS --------------------
        # Gentle fields in meaning-space; not tied to any token or word.
        sem.setdefault("flavour_gain", 0.02)   # how strongly flavours pull vectors

        # Create once, based on whatever semantic dim we're actually using
        if "latent_attractors" not in sem or not sem["latent_attractors"]:
            # Try to infer dimension from existing vecs, else fall back to 32
            if sem["vecs"]:
                some_vec = next(iter(sem["vecs"].values()))
                dim = len(some_vec) if isinstance(some_vec, (list, tuple)) else 32
            else:
                dim = 32

            def _att():
                # smallish random vector so the field is gentle
                return self._rand_vec(dim)

            sem["latent_attractors"] = {
                "objectness":     _att(),
                "processness":    _att(),
                "relationness":   _att(),
                "transformness":  _att(),
                "causativeness":  _att(),
                "temporalness":   _att(),
            }

        # --- SEMANTIC FLAVOUR CHANNELS -------------------------------------
        # token -> flavour vector
        self.semantic_flavour = {}  
        # structure:
        #   t -> {
        #         "objectness": 0.0,
        #         "processness": 0.0,
        #         "relationness": 0.0,
        #         "transformness": 0.0,
        #         "causativeness": 0.0,
        #         "temporalness": 0.0
        #       }

        # identity token registry
        if not hasattr(self, "identity_tokens"):
            self.identity_tokens = set()

        # preferred centroid for identity subspace
        if not hasattr(self, "IDENTITY_ROOT"):
            self.IDENTITY_ROOT = np.array(
                [
                    4.0, -3.0, 2.5, -4.2, 3.1, 1.7, -2.3, 2.8,
                    -3.7, 1.9, -1.4, 0.6, 2.2, -0.8, 1.5, -2.9,
                    *([0.0] * (32 - 16))
                ]
            )
        if not hasattr(self, "IDENTITY_JITTER"):
            self.IDENTITY_JITTER = 0.05

        # dictionary + recency
        if not hasattr(self, "dict_vocab"):
            self.dict_vocab = set()
        if not hasattr(self, "recent_tokens"):
            self.recent_tokens = []

        # dictionary update tracking
        if not hasattr(self, "_last_dict_refresh"):
            self._last_dict_refresh = -9999
        
        # --- Flavour drift axes (stable random directions) ---
        if not hasattr(self, "flavour_axes"):
            dim = 32
            self.flavour_axes = {
                "objectness": np.random.normal(size=dim),
                "processness": np.random.normal(size=dim),
                "relationness": np.random.normal(size=dim),
                "transformness": np.random.normal(size=dim),
                "causativeness": np.random.normal(size=dim),
                "temporalness": np.random.normal(size=dim),
            }
            # normalise each
            for k in self.flavour_axes:
                v = self.flavour_axes[k]
                self.flavour_axes[k] = v / (np.linalg.norm(v) + 1e-9)

        dim = 32  # semantic dimension for local random init

        def _local_randvec():
            import random
            return [random.uniform(-1, 1) for _ in range(dim)]

        # ensure all vecs pre-exist (syllables + dictionary + vocab if present)
        for tok in getattr(self, "vocab", []):
            if tok not in sem["vecs"]:
                sem["vecs"][tok] = _local_randvec()

    def _clip_semantic_vec(self, vec, max_norm=5.0):
        """
        Safety clamp for semantic vectors.
        Keeps norms bounded to avoid runaway explosions.
        """
        if not isinstance(vec, (list, tuple)):
            return vec

        v = np.array(vec, dtype=float)

        # kill NaNs
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

        n = np.linalg.norm(v)
        if n == 0.0:
            return self._rand_vec(len(v))

        if n > max_norm:
            v = v * (max_norm / n)

        return v.tolist()

    def _rand_vec(self, dim=32):
        """Wrapper so all mixins can request random semantic vectors."""
        try:
            from agents.semantics import rand_vec as _rv
            return _rv(dim)
        except Exception:
            # absolute fallback
            import random
            return [random.uniform(-1, 1) for _ in range(dim)]

    # ============================================================
    # TOKEN USAGE TICK (needed by LanguageMixinV2)
    # ============================================================
    def _get_flavour_attractor(self, channel, dim=None):
        """
        Fetch a latent attractor vector for a given flavour channel,
        padded / truncated to match the requested dimension.
        """
        self._ensure_semantic()
        sem = self.semantic
        lat = sem.get("latent_attractors", {})
        base = lat.get(channel)
        if base is None:
            return None

        if not isinstance(base, (list, tuple)):
            return None

        if dim is None:
            return list(base)

        v = list(base)
        if len(v) > dim:
            return v[:dim]
        if len(v) < dim:
            # pad with small noise so we don't create a hard axis
            extra = [random.uniform(-0.01, 0.01) for _ in range(dim - len(v))]
            v.extend(extra)
        return v

    def _apply_flavour_attractors(self):
        """
        Soft semantic seeding:
          Tokens that consistently show a given flavour (process, relation, etc.)
          have their vectors nudged *slightly* toward the corresponding
          latent attractor.

        This does NOT create or rename tokens; it only adjusts vectors
        based on how agents actually use them.
        """
        if not hasattr(self, "semantic_flavour") or not self.semantic_flavour:
            return

        self._ensure_semantic()
        sem = self.semantic
        vecs = sem.get("vecs", {})
        gain = float(sem.get("flavour_gain", 0.02))

        if gain <= 0.0:
            return

        for tok, flavour in list(self.semantic_flavour.items()):
            v = vecs.get(tok)
            if not isinstance(v, (list, tuple)):
                continue

            dim = len(v)
            # Combined pull from all flavours
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

            # Normalise by total_weight to keep step gentle and scale by gain
            step_scale = gain / max(1e-6, total_weight)
            new_v = [
                vi + step_scale * pi
                for vi, pi in zip(v, pull)
            ]
            vecs[tok] = new_v

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

    def _semantic_tick_token(self, tok):
        """
        Lightweight usage tick for a token.
        This is required by LanguageMixinV2 and is intentionally minimal.

        - Ensures token semantic record exists
        - Increments usage_count
        - Does NOT update vectors or links (handled by _observe_tokens)
        """

        self._ensure_semantic()

        # Ensure token semantic metadata exists
        tstore = self.semantic.setdefault("tokens", {})
        if tok not in tstore:
            tstore[tok] = {
                "families": {},
                "usage_count": 0,
                "birth_generation": getattr(self, "generation_index", 0),
            }

        tstore[tok]["usage_count"] += 1

    # ============================================================
    # DRIFT UPDATE (fallback for LanguageMixinV2 compatibility)
    # ============================================================
    def semantic_drift_update(self):
        """
        Default drift update for lexical semantics.
        LanguageMixinV2 extends this in its own layer (families, tokens).
        If that mixin is not present, this prevents coordinator crashes.
        """
        # Basic lexical drift is already handled by _gravity_step()
        # so here we simply do nothing.
        pass

    # ============================================================
    # INTERNAL SAFETY
    # ============================================================
    def _ensure_semantic(self):
        """Ensures semantic system exists (mixin safety)."""
        if not hasattr(self, "semantic") or self.semantic is None:
            self._init_semantic_system()

        sem = self.semantic

        # Make sure core fields exist and are correctly typed
        if "links" not in sem or not isinstance(sem["links"], defaultdict):
            sem["links"] = {defaultdict(lambda: defaultdict(float))}

        if "vecs" not in sem:
            sem["vecs"] = {}

    def _ensure_vec(self, tok):
        self._ensure_semantic()
        vecs = self.semantic["vecs"]
        if tok not in vecs:
            vecs[tok] = self._clip_semantic_vec(rand_vec())

    def is_numeric_token(self, tok):
        return (
            hasattr(self, "token_registry")
            and self.token_registry
            and self.token_registry.is_numeric(tok)
        )

    def register_numeric_token(self, tok):
        if hasattr(self, "token_registry") and self.token_registry:
            self.token_registry.register_numeric(tok)

    # ============================================================
    # SEMANTIC GRAPH (links + gravity)
    # ============================================================
    def _enforce_global_degree(self, tok):
            links = self.semantic["links"][tok]
            if len(links) <= MAX_GLOBAL_DEGREE:
                return

            # sort by usefulness, lowest first
            ordered = sorted(
                links.items(),
                key=lambda kv: kv[1].get("use", 0.0)
            )

            while len(links) > MAX_GLOBAL_DEGREE:
                dead, _ = ordered.pop(0)
                del links[dead]
                if tok in self.semantic["links"].get(dead, {}):
                    del self.semantic["links"][dead][tok]

    def _true_degree(self, tok):
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

        # ---------------------------
        # 0. Sanity guards
        # ---------------------------
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

        # ---------------------------
        # 1. Degree enforcement
        # ---------------------------
        deg = self._true_degree(a)

        if deg >= max_degree:
            # Build eviction candidates (only meaningful links)
            candidates = [
                (k, v) for k, v in row_a.items()
                if isinstance(v, dict) and abs(v.get("w", 0.0)) > 1e-6
            ]

            if not candidates:
                # Nothing worth evicting → reject new link
                return

            new_use = self._compute_usefulness(a, b)

            weakest_key, weakest_data = min(
                candidates,
                key=lambda kv: kv[1].get("use", 0.0)
            )

            weakest_use = weakest_data.get("use", 0.0)

            if new_use <= weakest_use:
                return

            # --- Evict weakest edge symmetrically ---
            row_a.pop(weakest_key, None)
            if weakest_key in links:
                links[weakest_key].pop(a, None)

        # ---------------------------
        # 2. Create / reinforce link
        # ---------------------------
        if b not in row_a:
            row_a[b] = {"w": 0.0, "use": 0.0, "last": now}
        if a not in row_b:
            row_b[a] = {"w": 0.0, "use": 0.0, "last": now}

        row_a[b]["w"] += w
        row_b[a]["w"] += w

        # Update usefulness
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
        neigh = self._semantic_neighbors(a, k=2)

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
        vecs = self.semantic["vecs"]
        links = self.semantic["links"]

        # --------------------------------------------------
        # Compute influence score per token
        # --------------------------------------------------
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

        # --------------------------------------------------
        # Select top-K most influential tokens
        # --------------------------------------------------
        K = 12
        top_tokens = [
            tok for tok, score in sorted(scored, key=lambda x: x[1], reverse=True)[:K]
        ]

        # --------------------------------------------------
        # Apply gravitational pull along weighted semantic links
        # --------------------------------------------------
        for a in top_tokens:
            nbrs = links[a]
            va = vecs.get(a)
            if va is None:
                continue

            for b, entry in nbrs.items():

                # extract weight
                if isinstance(entry, dict):
                    w = entry.get("w", 0.0)
                else:
                    w = float(entry or 0.0)

                if w == 0:
                    continue

                vb = vecs.get(b)
                if vb is None:
                    continue

                # directional pull from vb → va
                dv = sub(vb, va)
                new_va = add(va, scale(dv, g * w))

                vecs[a] = new_va
                va = new_va   # update for iterative effect

        # --------------------------------------------------
        # Optional maintenance step
        # --------------------------------------------------
        self._decay_links()

    # ============================================================
    # TOKEN DECOMPOSITION (CAREFUL)
    # ============================================================
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

    # ============================================================
    # CO-OCCURRENCE LEARNING
    # ============================================================
    # at top of file
    def observe_utterance(self, utter, gain_scale=1.0):
        tokens = self._decompose_token_for_semantics(utter)
        
        for tok in tokens:
            self.stab_note_token_usage(tok)
        
        self._observe_tokens(tokens, gain=self.semantic["cooccur_gain"] * gain_scale)

    def _observe_tokens(self, tokens, gain=None):
        if not tokens:
            return

        # 🔒 HARD NUMERIC ISOLATION
        tokens = [
            t for t in tokens
            if not self.is_numeric_token(t)
            and not self.is_identity_token(t)
        ]

        if len(tokens) < 2:
            return
        # --- SEMANTIC FLAVOUR UPDATE -------------------------------------
        self._update_token_flavour(tokens, context_gain=gain)

        numeric_sem = getattr(self, "numeric_semantic", {})
        raw = list(tokens)

        # Filter numeric anchors
        filtered = [t for t in raw if isinstance(t, str) and t not in numeric_sem]

        if len(filtered) > 15:
            filtered = random.sample(filtered, 15)

        # ---- existing logic resumes from here ----
        self._ensure_semantic()
        gain = self.semantic["cooccur_gain"] if gain is None else gain

        # ensure vecs + recency
        for t in filtered:
            self._ensure_vec(t)
            # Apply flavour drift to semantic vector
            vec = self.semantic["vecs"][t]
            self.semantic["vecs"][t] = self._apply_flavour_drift(t, vec)
            self.recent_tokens.append(t)

        # bloom noise (unchanged)
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

        # pairwise links with edge delta logging
        pairs = 0
        for i in range(len(filtered)):
            for j in range(i + 1, len(filtered)):
                self._link(filtered[i], filtered[j], gain)
                pairs += 1

        # trim recency (unchanged)
        if len(self.recent_tokens) > 64:
            self.recent_tokens = self.recent_tokens[-64:]

        # identity grounding (unchanged)
        if not hasattr(self, "known_agent_tokens"):
            self.known_agent_tokens = set()
        for t in filtered:
            if t.startswith("a") and t[1:].isdigit():
                self.known_agent_tokens.add(t)

    # ============================================================
    # OPTIONAL LINK DECAY / CLEANUP
    # ============================================================
    def _decay_links(self, decay=0.995, prune_threshold=1e-4):
        """
        Reduce old link weights and prune tiny edges.

        Works for both:
            - old format: nbrs[b] = float
            - new format: nbrs[b] = {"w": float, "use": float, "age": int}
        """
        links = self.semantic["links"]

        for a, nbrs in list(links.items()):

            for b, entry in list(nbrs.items()):

                # --- Extract current weight ---
                if isinstance(entry, dict):
                    w = entry.get("w", 0.0)
                else:
                    w = float(entry)

                # --- Apply decay ---
                new_w = w * decay

                # --- Prune if too small ---
                if abs(new_w) < prune_threshold:
                    del nbrs[b]
                    continue

                # --- Store updated weight back ---
                if isinstance(entry, dict):
                    entry["w"] = new_w
                    entry["age"] = entry.get("age", 0) + 1
                    nbrs[b] = entry
                else:
                    nbrs[b] = new_w

            # remove empty rows
            if not nbrs:
                del links[a]

    # ============================================================
    # DICTIONARY INTEGRATION
    # ============================================================
    def receive_semantic_seeds(self, seeds):
        """
        Receives:
        seeds = {
            "words": [...],
            "synonyms": [(a,b), ...],
            "antonyms": [(a,b), ...]
        }
        """

        # Ensure vectors exist for words
        for w in seeds["words"]:
            self._ensure_vec(w)

        # Link synonyms with positive weight
        for a, b in seeds["synonyms"]:
            self._link(a, b, +0.10)

        # Link antonyms with negative weight (inverse pull)
        for a, b in seeds["antonyms"]:
            self._link(a, b, -0.10)

    def populate_dictionary(self, words):
        """
        Populate agent dictionary with a *list of words*.
        Ensures vectors exist. Idempotent and fast.
        """
        if not words:
            return

        for w in words:
            wl = w.lower().strip()
            if not wl or len(wl) < 2:
                continue

            self.dict_vocab.add(wl)

            if hasattr(self, "_ensure_vec"):
                self._ensure_vec(wl)

    # ============================================================
    # SEMANTIC NEIGHBOURS
    # ============================================================
    def pick_semantic_neighbor(self, token):
        """Weighted-random neighbor based on link strength."""
        self._ensure_semantic()

        nbrs = self.semantic["links"].get(token, {})
        if not nbrs:
            return None

        words, weights = zip(*nbrs.items())
        weights = [max(1e-6, w) for w in weights]
        return random.choices(words, weights=weights)[0]

    # ============================================================
    # PUBLIC "TICK" FOR SEMANTIC EVOLUTION
    # ============================================================
    def semantic_update_tick(self):
        """
        Called by language generator or by the coordinator to evolve
        semantic vectors one step.
        """
        # lexical drift via co-occurrence gravity
        self._gravity_step()
        # gentle, usage-driven pull toward latent flavour attractors
        self._apply_flavour_attractors()

        vecs = self.semantic["vecs"]
        for key, vec in list(vecs.items()):
            vecs[key] = self._clip_semantic_vec(vec)

        self.semantic_stabilisation_tick()

    # ============================================================
    # TEACHING SUPPORT: EXPORT + PREDICT
    # ============================================================
    def export_semantic_bundle(self, max_keys=3):
        """Teacher exports a small semantic packet."""
        self._ensure_semantic()

        vecs = self.semantic["vecs"]
        if not vecs:
            return None

        keys = list(vecs.keys())
        random.shuffle(keys)
        chosen = keys[:max_keys]

        return {
            "tokens": chosen,
            "vecs": {t: vecs[t] for t in chosen}
        }

    def attempt_prediction(self, bundle):
        """
        Student synthesises predicted semantic vector from bundle.
        """
        self._ensure_semantic()

        if not bundle:
            return None

        toks = bundle.get("tokens", [])
        if not toks:
            return None

        preds = []

        for t in toks:
            if t not in self.semantic["vecs"]:
                curiosity = self.traits.get("curiosity", 0.5)
                if random.random() < curiosity:
                    self.semantic["vecs"][t] = rand_vec()
                else:
                    continue

            preds.append(self.semantic["vecs"][t])

        if not preds:
            return None

        # average vector
        acc = np.array(preds[0], copy=True)
        for v in preds[1:]:
            acc = add(acc, v)

        return scale(acc, 1.0 / len(preds))

    # ============================================================
    # SIMILARITY UTILITIES
    # ============================================================
    def semantic_similarity(self, a, b):
        self._ensure_semantic()
        vecs = self.semantic["vecs"]
        if a not in vecs or b not in vecs:
            return 0.0
        return cos_sim(vecs[a], vecs[b])

    def semantic_distance(self, a, b):
        return 1.0 - float(self.semantic_similarity(a, b))

    # ============================================================
    # OPTIONAL: INGEST FREE TEXT
    # ============================================================
    def add_semantic_observation(self, text: str):
        tokens = extract_keywords(text.lower())
        if tokens:
            self._observe_tokens(tokens)

    # ============================================================
    # AGENT IDENTITY SEMANTICS
    # ===========================================================
    def _init_agent_identity_semantics(self):
        self.name_token = f"agent_{self.id}"

        if hasattr(self, "vocab"):
            self.vocab.add(self.name_token)

        self._ensure_semantic()
        self._ensure_vec(self.name_token)

        # use the default factory, not a bare {}
        links = self.semantic["links"]
        row = links.get(self.name_token)
        if row is None or not isinstance(row, defaultdict):
            links[self.name_token] = defaultdict(float, row or {})

    def seed_agent_identity(self, agent_id):
        """
        Ensure agent identity token ('A7') has:
        - a stable semantic vector
        - links to trust channels
        """
        self._ensure_semantic()
        token = f"A{agent_id}"

        # 1) Create vector if missing
        vecs = self.semantic["vecs"]
        if token not in vecs:
            vecs[token] = self._rand_vec(dim=32)  # or smaller if you change DIM

        # 2) Bind semantics to trust
        trust = getattr(self, "trust_channels", {}).get(agent_id, {})
        if trust:
            base = vecs[token]
            for ch, v in trust.items():
                # reinforce positive trust as semantic closeness
                factor = 0.01 * v
                base = [x + factor for x in base]
            vecs[token] = base

        # 3) Track identity tokens as stable dictionary entries
        self.dict_vocab.add(token)

    def reinforce_identity(self, partner_id, val):
        """
        val in [-1, 1]. Pushes the internal semantic representation
        of another agent toward positive or negative alignment.
        """
        self._ensure_semantic()
        tok = f"a{partner_id}"
        self._ensure_vec(tok)

        v = self.semantic["vecs"][tok]
        nudge = rand_vec()
        updated = add(v, scale(nudge, 0.15 * val))
        self.semantic["vecs"][tok] = self._clip_semantic_vec(updated)

        # also strengthen link between identities
        if hasattr(self, "_link"):
            self._link(self.name_token, tok, 0.05 * val)

    # ============================================================
    # IDENTITY TOKEN SUPPORT
    # ============================================================
    def _identity_root_vec(self):
        """
        Produce a stable 64-dim identity root vector.
        """
        base = np.array([
            4.0, -3.0, 2.5, -4.2, 3.1, 1.7, -2.3, 2.8,
            -3.7, 1.9, -1.4, 0.6, 2.2, -0.8, 1.5, -2.9,
        ], dtype=float)

        # pad to semantic dim (usually 32)
        dim = 32
        if len(base) < dim:
            base = np.concatenate([base, np.zeros(dim - len(base))])

        return base

    def mark_identity_token(self, tok):
        """
        Ground an identity token in the identity subspace.
        """
        self._ensure_semantic()
        vec = self.IDENTITY_ROOT + np.random.normal(
            scale=self.IDENTITY_JITTER,
            size=self.IDENTITY_ROOT.shape
        )
        vec = self._clip_semantic_vec(vec.tolist())  # 🔒
        self.semantic["vecs"][tok] = vec
        self.identity_tokens.add(tok)

    def is_identity_token(self, tok):
        return tok in getattr(self, "identity_tokens", set())


    # ============================================================
    # SEMANTIC SNAPSHOT EXPORT - COMMUNITY ANALYSIS
    # ===========================================================
    def export_semantic_snapshot(self, k=200):
        """
        Export a limited set of semantic vectors:
        highest-usage tokens get priority.
        """
        sem = self.semantic["vecs"]
        usage = self.semantic.get("tokens", {})

        # Rank tokens by usage_count
        scored = []
        for tok in sem.keys():
            rec = usage.get(tok, {})
            scored.append((tok, rec.get("usage_count", 0)))

        scored.sort(key=lambda x: x[1], reverse=True)

        selected = [tok for tok, _ in scored[:k]]
        return {tok: sem[tok] for tok in selected}

    def community_distance(self, tok, community_map):
        if tok not in self.semantic["vecs"]:
            return None
        if tok not in community_map["vecs"]:
            return None

        import numpy as np
        a = np.array(self.semantic["vecs"][tok])
        b = np.array(community_map["vecs"][tok])
        return float(np.linalg.norm(a - b))

    def _update_token_flavour(self, tokens, context_gain=0.1):
        """
        Update semantic flavour vectors based on contextual cues
        in the token sequence.
        This is soft, incremental, safe.
        """

        # ensure context_gain is numeric
        if context_gain is None:
            context_gain = 1.0

        # detect simple structural contexts
        # (We can expand this as tasks evolve)
        for i, tok in enumerate(tokens):
            entry = self._ensure_flavour_entry(tok)

            # OBJECTNESS: Appears as a standalone or noun-slot-like
            entry["objectness"] += 0.05 * context_gain

            # RELATION: rel-* tokens imply linking
            if tok.startswith("rel"):
                entry["relationness"] += 0.15 * context_gain

            # PROCESS/TRANSFORM: detect sequences with movement/ordering
            # Example heuristics:
            if i > 0:
                prev = tokens[i-1]
                # chain-like patterns = verbish
                if prev.startswith("tol") or tok.startswith("tol"):
                    entry["processness"] += 0.12 * context_gain
                # muktar/muk-/mukka flows often read as transformations
                if tok.startswith("muk"):
                    entry["transformness"] += 0.10 * context_gain

            # CAUSATIVE: presence of "why" or relational grounding
            if tok == "why":
                entry["causativeness"] += 0.15 * context_gain

            # TEMPORAL: repeated chains or sequences
            if tok.endswith("su") or tok.endswith("rin"):
                entry["temporalness"] += 0.05 * context_gain

            # final normalisation pass (keep bounded)
            total = sum(entry.values())
            if total > 0:
                for k in entry:
                    entry[k] /= total

    def _apply_flavour_drift(self, tok, vec, lr=0.01):
        """
        Adjust embedding based on semantic flavour.
        Soft, safe, incremental.
        """
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

        # 🔒 clip before returning
        return self._clip_semantic_vec(newv)