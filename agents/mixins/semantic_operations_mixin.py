# agents/mixins/semantic_mixin.py

import random
import numpy as np
from collections import defaultdict

from agents.cognition.semantic_utils import (
    rand_vec, add, sub, scale, cos_sim,
)
from agents.agent_constants import SYLLABLES


class SemanticOperationsMixin:
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

    def export_semantic_bundle(self, max_keys=3):
        """Teacher exports a small semantic packet."""
        if hasattr(self, "semantic_system"):
            return self.semantic_system.export_bundle(max_keys=max_keys)

        self._ensure_semantic()
        vecs = self.semantic["vecs"]
        if not vecs:
            return None
        keys = list(vecs.keys())
        random.shuffle(keys)
        chosen = keys[:max_keys]
        return {"tokens": chosen, "vecs": {t: vecs[t] for t in chosen}}

    def attempt_prediction(self, bundle):
        """
        Student synthesises predicted semantic vector from bundle.
        """
        if hasattr(self, "semantic_system"):
            return self.semantic_system.attempt_prediction(bundle)

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
        acc = np.array(preds[0], copy=True)
        for v in preds[1:]:
            acc = add(acc, v)
        return scale(acc, 1.0 / len(preds))

    def semantic_similarity(self, a, b):
        if hasattr(self, "semantic_system"):
            return self.semantic_system.similarity(a, b)

        self._ensure_semantic()
        vecs = self.semantic["vecs"]
        if a not in vecs or b not in vecs:
            return 0.0
        return cos_sim(vecs[a], vecs[b])

    def semantic_distance(self, a, b):
        return 1.0 - float(self.semantic_similarity(a, b))

    def _init_agent_identity_semantics(self):
        self.name_token = f"agent_{self.id}"
        if hasattr(self, "vocab"):
            self.vocab.add(self.name_token)
        if hasattr(self, "semantic_system"):
            self.semantic_system._ensure_vec(self.name_token)

    def seed_agent_identity(self, agent_id):
        if hasattr(self, "identity_system"):
            self.identity_system.seed_agent_identity(agent_id)

    def reinforce_identity(self, partner_id, val):
        if hasattr(self, "identity_system"):
            self.identity_system.reinforce_identity(partner_id, val)

    def mark_identity_token(self, tok):
        if hasattr(self, "identity_system"):
            self.identity_system.mark_identity_token(tok)

    def is_identity_token(self, tok):
        if hasattr(self, "identity_system"):
            return self.identity_system.is_identity_token(tok)
        return False

    def export_semantic_snapshot(self, k=200):
        """Export a limited set of semantic vectors via EpistemicSystem."""
        if hasattr(self, "epistemic_system"):
            return self.epistemic_system.export_semantic_snapshot(k=k)

        sem = self.semantic.get("vecs", {})
        usage = self.semantic.get("tokens", {})
        scored = []
        for tok in sem.keys():
            rec = usage.get(tok, {})
            scored.append((tok, rec.get("usage_count", 0)))
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [tok for tok, _ in scored[:k]]
        return {tok: sem[tok] for tok in selected}

    def community_distance(self, tok, community_map):
        if hasattr(self, "epistemic_system"):
            return self.epistemic_system.community_distance(tok, community_map)

        if tok not in self.semantic.get("vecs", {}):
            return None
        if not community_map or tok not in community_map.get("vecs", {}):
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

        if hasattr(self, "semantic_system"):
            return self.semantic_system.update_token_flavour(tokens, context_gain=context_gain)

        if context_gain is None:
            context_gain = 1.0
        for i, tok in enumerate(tokens):
            entry = self._ensure_flavour_entry(tok)
            entry["objectness"] += 0.05 * context_gain
            if tok.startswith("rel"):
                entry["relationness"] += 0.15 * context_gain
            if i > 0:
                prev = tokens[i-1]
                if prev.startswith("tol") or tok.startswith("tol"):
                    entry["processness"] += 0.12 * context_gain
                if tok.startswith("muk"):
                    entry["transformness"] += 0.10 * context_gain
            if tok == "why":
                entry["causativeness"] += 0.15 * context_gain
            if tok.endswith("su") or tok.endswith("rin"):
                entry["temporalness"] += 0.05 * context_gain
            total = sum(entry.values())
            if total > 0:
                for k in entry:
                    entry[k] /= total

    def _apply_flavour_drift(self, tok, vec, lr=0.01):
        """
        Adjust embedding based on semantic flavour.
        Soft, safe, incremental.
        """
        if hasattr(self, "semantic_system"):
            return self.semantic_system.apply_flavour_drift(tok, vec, lr=lr)

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
        return self._clip_semantic_vec(newv)
