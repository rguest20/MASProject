"""
agents/cognition/epistemic_system.py
Epistemic system for managing agent beliefs and knowledge.
"""

import numpy as np
import random

class EpistemicSystem:
    """
    Manages the beliefs and knowledge of an agent.
    """
    def __init__(self, owner):
        self.owner = owner
        self.beliefs = {}         # token -> confidence
        self.dissonance = 0.0
        self.semantic_gaps = {
            "misaligned": [],
            "orphans": [],
            "last_gen": -1,
        }
        self.flavour_homeostasis = {}

    def community_distance(self, tok, community_map):
        if not community_map:
            return None
        c_vecs = community_map.get("vecs", {})
        if tok not in c_vecs:
            return None

        vecs = getattr(self.owner.semantic_system, "vectors", {})
        if tok not in vecs:
            return None

        a = np.array(vecs[tok], dtype=float)
        b = np.array(c_vecs[tok], dtype=float)
        return float(np.linalg.norm(a - b))

    # -------------------------------------------------
    # Semantic snapshot export (for community analysis)
    # -------------------------------------------------
    def export_semantic_snapshot(self, k: int = 200):
        vecs = getattr(self.owner.semantic_system, "vectors", {})
        tokens = getattr(self.owner.semantic_system, "tokens", {})
        if not vecs:
            return {}

        scored = []
        for tok in vecs.keys():
            meta = tokens.get(tok, {})
            scored.append((tok, meta.get("usage_count", 0)))

        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [tok for tok, _ in scored[:k]]
        return {tok: vecs[tok] for tok in selected}

    def detect_semantic_gaps(
        self,
        community_map,
        min_usage=5,
        min_comm_count=3,
        dist_thresh=0.6,
        max_gaps=20,
    ):
        if not community_map:
            return self.semantic_gaps
        c_vecs = community_map.get("vecs", {})
        c_counts = community_map.get("counts", {})
        if not c_vecs:
            return self.semantic_gaps
        vecs = self.owner.semantic_system.vectors
        tokens_meta = self.owner.semantic_system.tokens
        misaligned = []
        orphans = []
        for tok, meta in tokens_meta.items():
            usage = meta.get("usage_count", 0)
            if usage < min_usage:
                continue
            if self.owner._is_identity_like(tok):
                continue
            if tok not in vecs:
                continue
            v_local = np.array(vecs[tok], dtype=float)
            comm_count = c_counts.get(tok, 0)
            if tok in c_vecs and comm_count >= min_comm_count:
                v_comm = np.array(c_vecs[tok], dtype=float)
                dist = float(np.linalg.norm(v_local - v_comm))
                if dist >= dist_thresh:
                    misaligned.append({
                        "token": tok,
                        "dist": dist,
                        "usage": int(usage),
                        "community_count": int(comm_count),
                    })
            else:
                orphans.append({
                    "token": tok,
                    "usage": int(usage),
                    "community_count": int(comm_count),
                })
        misaligned.sort(
            key=lambda d: (d["dist"] * max(1, d["usage"])),
            reverse=True,
        )
        orphans.sort(
            key=lambda d: d["usage"],
            reverse=True,
        )
        misaligned = misaligned[:max_gaps]
        orphans = orphans[:max_gaps]
        self.semantic_gaps = {
            "misaligned": misaligned,
            "orphans": orphans,
            "last_gen": getattr(self, "current_generation", -1),
        }
        return self.semantic_gaps
    
    def apply_flavour_homeostasis(
        self,
        community_map,
        soft_strength=0.15,
        dominance_thresh=0.22,
        scarcity_thresh=0.05,
        min_usage=3
    ):
        if not community_map:
            return
        c_vecs = community_map.get("vecs", {})
        c_counts = community_map.get("counts", {})
        if not c_vecs or not c_counts:
            return

        def prefix(tok):
            for p in ["rel", "rin", "bel", "tol", "muk", "zev", "tar", "su", "lo"]:
                if tok.startswith(p):
                    return p
            return "_misc"

        fam_totals = {}
        total = 0
        for tok, count in c_counts.items():
            if count < min_usage:
                continue
            fam = prefix(tok)
            fam_totals[fam] = fam_totals.get(fam, 0) + count
            total += count
        if total == 0:
            return
        fam_freq = {f: fam_totals[f] / total for f in fam_totals}
        dominant = {f for f, p in fam_freq.items() if p >= dominance_thresh}
        scarce   = {f for f, p in fam_freq.items() if p <= scarcity_thresh}
        if not dominant and not scarce:
            return

        vecs = self.owner.semantic_system.vectors
        tokens_meta = self.owner.semantic_system.tokens
        for tok, meta in tokens_meta.items():
            fam = prefix(tok)
            v = vecs.get(tok)
            if v is None:
                continue
            if fam in dominant:
                vecs[tok] = [x * (1 - soft_strength) for x in v]
                meta["generation_bias"] = meta.get("generation_bias", 1.0) * (1 - 0.5 * soft_strength)
            elif fam in scarce:
                vecs[tok] = [x * (1 + soft_strength) for x in v]
                meta["generation_bias"] = meta.get("generation_bias", 1.0) * (1 + 0.4 * soft_strength)

        self.flavour_homeostasis = {
            "dominant": list(dominant),
            "scarce": list(scarce),
            "freqs": fam_freq,
        }

    def maybe_propose_semantic_tasks(self, coordinator, max_tasks=1):
        if not hasattr(self, "semantic_gaps"):
            return
        if self.semantic_gaps.get("last_gen") != getattr(self, "current_generation", -1):
            return
        mis = self.semantic_gaps.get("misaligned", [])
        orp = self.semantic_gaps.get("orphans", [])
        options = []
        for x in mis:
            options.append(("misaligned", x))
        for x in orp:
            options.append(("orphan", x))
        if not options:
            return
        random.shuffle(options)
        todo = options[:max_tasks]
        for kind, item in todo:
            tok = item["token"]
            vec_local = self.owner.semantic_system.vectors.get(tok)
            vec_comm = coordinator.community_semantic["vecs"].get(tok)
            if vec_local is None:
                continue
            task = {
                "task_id": coordinator._generate_task_id(),
                "task_type": "semantic_alignment",
                "token": tok,
                "proposer": self.owner.id,
                "kind": kind,
                "vector_local": list(vec_local),
                "community_vector": list(vec_comm or []),
            }
            coordinator.add_semantic_alignment_task(task)
