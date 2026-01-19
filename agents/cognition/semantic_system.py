"""
agents/cognition/semantic_system.py
Semantic system for managing agent semantic memory and associations.
"""

import random
from config import DIMS
class SemanticSystem:
    """
    Manages the semantic memory and associations of an agent.
    """
    def __init__(self, owner):
        self.owner = owner
        self.family_system = self.SemanticFamily(owner)
        self.vectors = {}
        self.tokens = {}
        self.links = {}
        self.last_used = {}

        # Predefine "why" token for reasoning
        self.vectors["why"] = self._randvec()
        self.owner.vocab.add("why")

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------
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
            if hasattr(self, "vocab") and other not in self.owner.vocab:
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

    def _family_neighbors(self, tok):
        sem = getattr(self, "semantic", None)
        if not sem:
            return []

        tsem = self.tokens
        fams = self.families
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
        if not hasattr(self, "identity_token"):
            return 0.0
        my_tok = self.owner.identity_token
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
            self.family_counter=0

        def _create_family(self, centroid=None, parent_ids=None, ftype="emergent"):
            fid = f"F{self.family_counter}"
            self.family_counter += 1

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
            if not hasattr(self, "api") or self.api is None:
                return
            sem = self.semantic
            fams = sem["families"]
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
                    fam["name"] = self._invent_token(prefix="f", concept=True)
                c = fam.get("centroid")
                if not c:
                    continue
                c_str = ",".join(f"{x:.3f}" for x in c)
                line = (
                    f"A{self.id} teach_family name={fam['name']} "
                    f"conf={conf:.3f} centroid={c_str}\n"
                )
                try:
                    self.api.append_text("/family_gossip.txt", line, scope="world")
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