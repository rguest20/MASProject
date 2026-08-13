# agents/mixins/semantic_mixin.py

import random
import numpy as np
from collections import defaultdict

from agents.cognition.semantic_utils import (
    rand_vec, add, sub, scale, cos_sim,
)
from agents.agent_constants import SYLLABLES


class SemanticInitialisationMixin:
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

        # Wire shared storage into cognition layer if available
        if hasattr(self, "semantic_system"):
            # unify vectors
            old_vecs = getattr(self.semantic_system, "vectors", {}) or {}
            sem_vecs = sem.setdefault("vecs", {})
            for k, v in old_vecs.items():
                sem_vecs.setdefault(k, v)
            self.semantic_system.vectors = sem_vecs

            # unify token metadata
            old_tokens = getattr(self.semantic_system, "tokens", {}) or {}
            sem_tokens = sem.setdefault("tokens", {})
            for k, v in old_tokens.items():
                sem_tokens.setdefault(k, v)
            self.semantic_system.tokens = sem_tokens

            # unify concept token mapping (used by family clustering filters)
            sem.setdefault("concept_tokens", {})
            if not isinstance(sem["concept_tokens"], dict):
                sem["concept_tokens"] = {}
            self.semantic_system.concept_tokens = sem["concept_tokens"]

            # unify link graph
            self.semantic_system.links = sem["links"]

            # unify family storage (used by coordinator + task system)
            fams = sem.setdefault("families", {})
            if not isinstance(fams, dict):
                fams = {}
                sem["families"] = fams

            sem.setdefault("family_counter", 0)
            try:
                sem["family_counter"] = int(sem["family_counter"])
            except Exception:
                sem["family_counter"] = 0

            if hasattr(self.semantic_system, "family_system"):
                self.semantic_system.family_system.families = fams
                self.semantic_system.families = fams
                try:
                    self.semantic_system.family_system.family_counter = int(sem["family_counter"])
                except Exception:
                    self.semantic_system.family_system.family_counter = 0

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

        # dictionary + recency
        if not hasattr(self, "dict_vocab"):
            self.dict_vocab = set()

    def detect_semantic_families(self, *args, **kwargs):
        if hasattr(self, "semantic_system") and hasattr(self.semantic_system, "family_system"):
            return self.semantic_system.family_system.detect_semantic_families(*args, **kwargs)
        return None

    def prune_families(self, *args, **kwargs):
        if hasattr(self, "semantic_system") and hasattr(self.semantic_system, "family_system"):
            return self.semantic_system.family_system.prune_families(*args, **kwargs)
        return None

    def family_reinforcement_update(self, *args, **kwargs):
        if hasattr(self, "semantic_system") and hasattr(self.semantic_system, "family_system"):
            return self.semantic_system.family_system.family_reinforcement_update(*args, **kwargs)
        return None

    def family_soft_decay(self, *args, **kwargs):
        if hasattr(self, "semantic_system") and hasattr(self.semantic_system, "family_system"):
            return self.semantic_system.family_system.family_soft_decay(*args, **kwargs)
        return None

    def name_families(self, *args, **kwargs):
        if hasattr(self, "semantic_system") and hasattr(self.semantic_system, "family_system"):
            return self.semantic_system.family_system.name_families(*args, **kwargs)
        return None

    def export_family_snapshot(self, *args, **kwargs):
        if hasattr(self, "semantic_system") and hasattr(self.semantic_system, "family_system"):
            return self.semantic_system.family_system.export_family_snapshot(*args, **kwargs)
        return []

    def maybe_broadcast_families(self, *args, **kwargs):
        if hasattr(self, "semantic_system") and hasattr(self.semantic_system, "family_system"):
            return self.semantic_system.family_system.maybe_broadcast_families(*args, **kwargs)
        return None

    def _integrate_family_gossip_line(self, *args, **kwargs):
        if hasattr(self, "semantic_system") and hasattr(self.semantic_system, "family_system"):
            return self.semantic_system.family_system._integrate_family_gossip_line(*args, **kwargs)
        return None
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
        """Wrapper around SemanticSystem.clip_vector for safety clamps."""
        if hasattr(self, "semantic_system"):
            return self.semantic_system.clip_vector(vec, max_norm=max_norm)

        # Fallback: local implementation if semantic_system is missing
        if not isinstance(vec, (list, tuple)):
            return vec
        v = np.array(vec, dtype=float)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        n = np.linalg.norm(v)
        if n == 0.0:
            return self._rand_vec(len(v))
        if n > max_norm:
            v = v * (max_norm / n)
        return v.tolist()

    def _rand_vec(self, dim=32):
        """Wrapper so all mixins can request random semantic vectors."""
        if hasattr(self, "semantic_system"):
            # SemanticSystem already respects configured semantic_dim
            return self.semantic_system._randvec(scale=1.0)

        try:
            from agents.cognition.semantic_utils import rand_vec as _rv
            return _rv(dim)
        except Exception:
            import random
            return [random.uniform(-1, 1) for _ in range(dim)]

    def _get_flavour_attractor(self, channel, dim=None):
        """Bridge to SemanticSystem flavour attractors."""
        if hasattr(self, "semantic_system"):
            return self.semantic_system._get_flavour_attractor(channel, dim=dim)

        # legacy fallback using self.semantic
        self._ensure_semantic()
        sem = self.semantic
        lat = sem.get("latent_attractors", {})
        base = lat.get(channel)
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

    def _apply_flavour_attractors(self):
        """
        Soft semantic seeding:
          Tokens that consistently show a given flavour (process, relation, etc.)
          have their vectors nudged *slightly* toward the corresponding
          latent attractor.

        This does NOT create or rename tokens; it only adjusts vectors
        based on how agents actually use them.
        """
        if hasattr(self, "semantic_system"):
            return self.semantic_system.apply_flavour_attractors()

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

    def _ensure_flavour_entry(self, tok):
        if hasattr(self, "semantic_system"):
            return self.semantic_system._ensure_flavour_entry(tok)

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

    def _ensure_token_semantic(self, tok):
        """Ensure token metadata exists in the shared semantic store."""
        self._ensure_semantic()
        tstore = self.semantic.setdefault("tokens", {})
        if tok not in tstore:
            tstore[tok] = {
                "families": {},
                "usage_count": 0,
                "birth_generation": getattr(self, "generation_index", 0),
            }
        return tstore[tok]
