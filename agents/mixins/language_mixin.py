#!/usr/bin/env python3
# agents/mixins/language_mixin.py
# V2 has been merged into this to create a unified LanguageMixin

import random
import re
import math
import time
from collections import defaultdict

import numpy as np

from agents.agent_constants import SYLLABLES, PUNCT
from agents.language import LanguageOrgan


def PERF(msg):
    print(f"[PERF {time.time():.3f}] {msg}", flush=True)


class LanguageMixin:
    """
    Unified semantic–linguistic system (v2 preferred).

    This version:
      - NEVER creates its own semantic store
      - ALWAYS extends the semantic dict created by SemanticMixin
      - Uses the shared lexical vector space for:
            * syllables
            * invented tokens
            * numeric anchors
            * concept tokens
            * family clusters
      - Provides stable, emergent grammar & justification phenomena
    """

    # =====================================================
    # INITIALISATION
    # =====================================================
    def _init_language_system(self):
        # # 1) Ensure semantic store exists (from SemanticMixin)
        # self._ensure_semantic()
        # sem = self.semantic

        # # 2) Extend semantic store with language-level fields
        # sem.setdefault("concept_tokens", {})
        # sem.setdefault("tokens", {})            # token → family/membership metadata
        # sem.setdefault("families", {})          # family_id → centroid & members
        # sem.setdefault("family_counter", 0)
        # sem.setdefault("rel_family", {})        # rel* meta-tokens (reason talk)
        # sem.setdefault("ref_family", {})        # ref* meta-tokens (talking about tokens)

        # identity cache
        # self.identity_tokens = set()

        # 3) Base vocab = syllables + invented
        self.vocab = set(SYLLABLES)
        self.dict_vocab = set()  # compatibility only

        # Utterance memory
        self.utterance_memory = {
            "associations": {},
            "usage_count": {},
        }

        # --- Semantic gap diagnostics (Phase 2) ---
        # self.semantic_gaps = {
        #     "misaligned": [],
        #     "orphans":   [],
        #     "last_gen":  -1,
        # }

        self.utter_bias = {
            "symbol_preferences": {s: random.uniform(-0.25, 0.25) for s in SYLLABLES},
            "length_bias": random.uniform(0.0, 1.0),
            "repeat_bias": random.uniform(0.0, 1.0),
        }

        self.symbol_drift = {s: 0.0 for s in SYLLABLES}
        self._next_token_id = 0

        # 4) Counting + reasoning organ
        # if not hasattr(self, "counting"):
        #     raise RuntimeError("CountingSystem must be initialised in InitMixin")
        
        self.language = LanguageOrgan(
            owner=self,
            trust_threshold=self.traits.get("trust_threshold", 0.5),
        )

        if not hasattr(self, "recent_tokens"):
            self.recent_tokens = []

        # reasoning subsystem
        if hasattr(self, "_init_reasoning_system"):
            self._init_reasoning_system()

        # 7) Pre-invented tokens (seed diversity)
        for _ in range(100):
            tok = self._invent_token(max_syllables=3)
            self.vocab.add(tok)
            self.semantic_system._ensure_vec(tok)

        # # 8) Identity tokens
        # if not hasattr(self, "identity_token"):
        #     self.identity_token = f"agent_{self.id}"
        #     self.vocab.add(self.identity_token)
        #     self.semantic_system._ensure_vec(self.identity_token)
        #     self.identity_tokens.add(self.identity_token)

        #     dim = self.semantic_system._semantic_dim()
        #     base = self.semantic_system._randvec(scale=0.6)
        #     jitter = [random.uniform(-0.05, 0.05) for _ in range(dim)]
        #     self.semantic["vecs"][self.identity_token] = [
        #         b + j for b, j in zip(base, jitter)
        #     ]

        #     self.identity_vec = list(self.semantic["vecs"][self.identity_token])
        #     self.nicknames_for_others = {}
        #     self.heard_nicknames = defaultdict(set)

    # =====================================================
    # HELPERS
    # =====================================================
    # def _ensure_vec(self, tok, scale=0.25):
    #     vecs = self.semantic.setdefault("vecs", {})
    #     if tok not in vecs:
    #         vecs[tok] = self._randvec(scale=scale)
    #     return vecs[tok]

    # def _l_ensure_vec(self, tok, scale=0.25):
    #     return self.semantic_system._ensure_vec(tok, scale)

    # def _r_ensure_vec(self, tok, scale=0.25):
    #     return self.semantic_system._ensure_vec(tok, scale)

    # def _sanitize_vector_dims(self):
    #     dim = self._semantic_dim()
    #     vecs = self.semantic.get("vecs", {})
    #     for tok, v in list(vecs.items()):
    #         if not isinstance(v, (list, tuple)):
    #             vecs[tok] = self._randvec()
    #             continue
    #         if len(v) != dim:
    #             new_v = list(v[:dim])
    #             if len(new_v) < dim:
    #                 new_v += [random.uniform(-0.1, 0.1) for _ in range(dim - len(new_v))]
    #             vecs[tok] = new_v

    # def _invent_name_token(self, other_id: int) -> str:
    #     tok = f"agent_{other_id}"
    #     self.vocab.add(tok)
    #     self.semantic_system._ensure_vec(tok)
    #     if hasattr(self, "identity_tokens"):
    #         self.identity_tokens.add(tok)
    #     return tok

    def _clamp(self, x, lo, hi):
        return max(lo, min(hi, x))

    def _numeric_collision_score(self):
        mapping = self.numeric_semantic
        tokens = list(mapping.values())
        unique = set(tokens)
        return max(0, len(tokens) - len(unique))

    # def _centroid(self, vecs):
    #     vecs = [v for v in vecs if v]
    #     if not vecs:
    #         dim = self._semantic_dim()
    #         return [0.0] * dim
    #     dim = len(vecs[0])
    #     return [
    #         sum(v[d] for v in vecs) / len(vecs)
    #         for d in range(dim)
    #     ]

    # def _semantic_dim(self):
    #     return getattr(self, "semantic_dim", 32)

    # def _randvec(self, scale=1.0):
    #     dim = self._semantic_dim()
    #     return [random.uniform(-scale, scale) for _ in range(dim)]

    def _is_identity_like(self, tok):
        return tok.startswith("id") and tok[2:].isdigit()

    def _ensure_token_semantic(self, tok):
        tstore = self.semantic["tokens"]
        if tok not in tstore:
            tstore[tok] = {
                "families": {},
                "usage_count": 0,
                "birth_generation": getattr(self, "generation_index", 0),
            }
        return tstore[tok]

    # =====================================================
    # TOKEN CLEANING
    # =====================================================
    def _parse_utterance(self, utterance):
        if not utterance:
            return []
        toks = []
        for t in utterance.split():
            t = re.sub(r"[^\w\-']+$", "", t.lower()).strip()
            if t:
                toks.append(t)
        return toks

    # # =====================================================
    # # SEMANTIC NEIGHBORS + START TOKEN
    # # =====================================================
    # def _semantic_neighbors(self, tok, k=8, max_radius=None):
    #     sem = getattr(self, "semantic", None)
    #     if not sem:
    #         return []

    #     vecs = sem.get("vecs", {})
    #     if tok not in vecs:
    #         return []

    #     v = vecs[tok]
    #     if not isinstance(v, (list, tuple)) or not v:
    #         return []

    #     dim = len(v)
    #     candidates = []

    #     for other, w in vecs.items():
    #         if other == tok:
    #             continue
    #         if self._is_identity_like(other):
    #             continue
    #         if hasattr(self, "vocab") and other not in self.vocab:
    #             continue
    #         if not isinstance(w, (list, tuple)) or not w:
    #             continue

    #         d2 = 0.0
    #         m = min(dim, len(w))
    #         for i in range(1, m):
    #             dv = v[i] - w[i]
    #             d2 += dv * dv

    #         candidates.append((other, d2))

    #     if not candidates:
    #         return []

    #     candidates.sort(key=lambda x: x[1])
    #     out = []
    #     for other, d2 in candidates:
    #         if max_radius is not None and (d2 ** 0.5) > max_radius:
    #             continue
    #         out.append(other)
    #         if len(out) >= k:
    #             break
    #     return out

    # def _family_neighbors(self, tok):
    #     sem = getattr(self, "semantic", None)
    #     if not sem:
    #         return []

    #     tsem = sem.get("tokens", {})
    #     fams = sem.get("families", {})
    #     meta = tsem.get(tok)
    #     if not meta:
    #         return []

    #     out = set()
    #     for fid in meta.get("families", {}):
    #         fam = fams.get(fid)
    #         if not fam:
    #             continue
    #         for member in fam.get("members", {}).keys():
    #             if member != tok and not self._is_identity_like(member):
    #                 out.add(member)

    #     return list(out)

    def _choose_utter_start(self, all_tokens):
        express = float(self.traits.get("expressiveness", 0.5))
        sem = getattr(self, "semantic", None)

        if sem is not None:
            rel_fam = list(sem.get("rel_family", {}).keys())
            if rel_fam and random.random() < (0.25 + 0.20 * express):
                return random.choice(rel_fam)

        if hasattr(self, "_last_tokens") and hasattr(self, "symbol_map"):
            numeric_vals = set(self.symbol_map.values())
            num_cands = [t for t in (self._last_tokens or []) if t in numeric_vals]
            if num_cands and random.random() < 0.4:
                return random.choice(num_cands)

        if getattr(self, "recent_tokens", None):
            recent = [t for t in self.recent_tokens[-20:] if t in all_tokens]
            if recent and random.random() < 0.7:
                return random.choice(recent)

        if all_tokens:
            return random.choice(all_tokens)
        return "na"

    def _sample_token_from_pool(self, pool_tokens, prefs):
        pool_tokens = [t for t in pool_tokens if isinstance(t, str) and t.strip()]
        if not pool_tokens:
            return None

        logits = [prefs.get(t, 0.2) for t in pool_tokens]
        if any(not isinstance(l, (int, float)) or math.isnan(l) for l in logits):
            return random.choice(pool_tokens)

        m = max(logits)
        exps = [math.exp(l - m) for l in logits]
        total = sum(exps)
        if not total or math.isnan(total):
            return random.choice(pool_tokens)

        probs = [e / total for e in exps]
        probs = self.stab_adjust_token_probs(pool_tokens, probs)
        return random.choices(pool_tokens, probs)[0]

    # =====================================================
    # EMOTIONAL MODIFIERS (unchanged)
    # =====================================================
    def _emotion_mod_len(self):
        S = self.state
        return int(
            (S["curiosity"] - 0.5) * 3 +
            (S["loneliness"] - 0.5) * 2 +
            (S["frustration"] - 0.5) * -3 +
            (S["happiness"] - 0.5) * 2
        )

    def _emotion_mod_punct(self):
        S = self.state
        return (
            (S["happiness"] - 0.5) * 0.25 +
            (S["frustration"] - 0.5) * -0.20 +
            (S["confidence"] - 0.5) * 0.15
        )

    # =====================================================
    # SPEAK NUMERIC
    # =====================================================
    def speak_number(self, n):
        try:
            digits = self.counting.interpret(n)
        except Exception:
            return str(n)

        if not hasattr(self, "symbol_map") or self.symbol_map is None:
            self.symbol_map = {}
        else:
            self.symbol_map = {
                k: v for k, v in self.symbol_map.items()
                if isinstance(v, str) and v.strip()
            }

        tokens = []
        for d in digits:
            if d in self.symbol_map:
                tok = self.symbol_map[d]
            else:
                try:
                    tok = self.counting.get_symbol(d)
                except Exception:
                    tok = f"num{d}"
                self.symbol_map[d] = tok

            if tok is None:
                continue
            tok = str(tok).strip()
            if not tok:
                continue

            self.vocab.add(tok)
            self.semantic_system._ensure_vec(tok)
            self._ensure_token_semantic(tok)
            self._semantic_tick_token(tok)
            tokens.append(tok)

        if not tokens:
            tokens = ["na"]

        clean = [t.lower() for t in tokens]
        self._observe_language_tokens(clean)

        utter = " ".join(tokens)
        self.last_written_word = utter
        self._last_tokens = clean
        return utter

    # =====================================================
    # MAIN UTTERANCE GENERATOR (SEMANTIC WALK)
    # =====================================================
    def produce_utterance(self):
        express = float(self.traits.get("expressiveness", 0.5))

        base_len = random.randint(2, 6)
        length = max(2, min(10, base_len + self._emotion_mod_len()))

        punct_prob = max(
            0.01,
            min(0.9, 0.15 + 0.3 * express + self._emotion_mod_punct()),
        )

        syll_pool = list(SYLLABLES)
        invented_pool = [
            t for t in self.vocab
            if t not in SYLLABLES and not self._is_identity_like(t)
        ]
        concept_tokens = set(self.semantic["concept_tokens"].values())
        all_tokens = (set(syll_pool) | set(invented_pool)) - concept_tokens

        if hasattr(self, "symbol_map"):
            all_tokens |= (set(self.symbol_map.values()) - concept_tokens)
        if hasattr(self, "reasoning_tokens"):
            all_tokens |= set(self.reasoning_tokens.values())

        all_tokens = {
            t for t in all_tokens
            if isinstance(t, str) and t.strip() and not self._is_identity_like(t)
        }
        all_tokens.add("why")

        for t in all_tokens:
            self.semantic_system._ensure_vec(t)
        all_tokens = list(all_tokens)

        if not all_tokens:
            utter = "na"
            self._last_tokens = ["na"]
            self.last_written_word = utter
            mem = self.utterance_memory["usage_count"]
            mem[utter] = mem.get(utter, 0) + 1
            return utter

        prefs = self.utter_bias["symbol_preferences"]
        for t in all_tokens:
            prefs.setdefault(t, 0.2)
        for w in invented_pool:
            prefs.setdefault(w, 0.8)
        if hasattr(self, "symbol_map"):
            for tok in self.symbol_map.values():
                prefs.setdefault(tok, 0.6)
        if hasattr(self, "reasoning_tokens"):
            for r in self.reasoning_tokens.values():
                prefs.setdefault(r, 0.05)
        for t in self.recent_tokens[-25:]:
            if not self._is_identity_like(t):
                prefs[t] = min(3.0, prefs.get(t, 0.2) + 0.15)
        if random.random() < express:
            for w in invented_pool:
                prefs[w] = min(3.0, prefs.get(w, 1.2) + 0.15)

        toks = []
        cur = self._choose_utter_start(all_tokens)
        if not isinstance(cur, str) or not cur.strip():
            cur = random.choice(all_tokens)
        toks.append(cur)
        if not self._is_identity_like(cur):
            self.recent_tokens.append(cur)
            prefs[cur] = min(3.0, prefs.get(cur, 0.2) + 0.05)

        for _ in range(length - 1):
            neighbors = self.semantic_system._semantic_neighbors(cur, k=15, max_radius=0.7)
            fam_nbrs = self.semantic_system._family_neighbors(cur)
            if fam_nbrs:
                neighbors = list(set(neighbors) | set(fam_nbrs))
            if neighbors and random.random() < 0.2:
                extra = random.sample(all_tokens, min(5, len(all_tokens)))
                neighbors = list(set(neighbors) | set(extra))
            candidate_pool = neighbors or all_tokens
            tok = self._sample_token_from_pool(candidate_pool, prefs)
            if tok is None:
                tok = random.choice(all_tokens)
            toks.append(tok)
            if not self._is_identity_like(tok):
                self.recent_tokens.append(tok)
                prefs[tok] = min(3.0, prefs.get(tok, 0.2) + 0.05)
            cur = tok

        toks = [t for t in toks if isinstance(t, str) and t.strip()]
        if not toks:
            toks = ["na"]

        utter = " ".join(toks)
        if random.random() < punct_prob:
            utter += random.choice(PUNCT)

        clean = self._parse_utterance(utter)
        if clean:
            self._observe_language_tokens(clean)
            if len(clean) > 1 and hasattr(self, "symbol_map"):
                symvals = set(self.symbol_map.values())
                if all(tok in symvals for tok in clean):
                    self._observe_language_tokens(clean, gain=0.2)

        self._last_tokens = clean
        self.last_written_word = utter
        mem = self.utterance_memory["usage_count"]
        mem[utter] = mem.get(utter, 0) + 1
        return utter

    # =====================================================
    # WORLD INGEST
    # =====================================================
    def language_world_ingest_step(self):
        if not hasattr(self, "api") or self.api is None:
            return

        texts = []
        for fname in ["/notes.txt", "/help_responses.txt", "/family_gossip.txt"]:
            t = self.api.read_text(fname)
            if t:
                texts.append(t)

        if not texts:
            return

        for txt in texts:
            for line in txt.strip().splitlines()[-10:]:
                if "teach_numeric" in line:
                    try:
                        self._integrate_teach_numeric_line(line)
                    except Exception:
                        pass

                toks = []
                for t in line.split():
                    raw = t.strip(",.!?;:\"'")
                    if not raw:
                        continue
                    if self._is_identity_like(raw.lower()):
                        continue

                    tok = raw.lower()
                    if (
                        tok in self.vocab or
                        (hasattr(self, "symbol_map") and tok in self.symbol_map.values())
                    ):
                        toks.append(tok)

                for t in toks:
                    self.vocab.add(t)
                    self.semantic_system._ensure_vec(t)

                if toks:
                    toks = [t for t in toks if t not in self.numeric_semantic]
                    self._observe_language_tokens(
                        toks, gain=self.semantic.get("dictionary_gain", 0.1)
                    )

    # =====================================================
    # FAMILY SYSTEM (detection/reinforcement)
    # =====================================================
    # def _create_family(self, centroid=None, parent_ids=None, ftype="emergent"):
    #     sem = self.semantic
    #     fid = f"F{sem['family_counter']}"
    #     sem["family_counter"] += 1

    #     sem["families"][fid] = {
    #         "centroid": list(centroid) if centroid is not None else None,
    #         "members": {},
    #         "parents": set(parent_ids or []),
    #         "children": set(),
    #         "depth": 0,
    #         "age": 0,
    #         "type": ftype,
    #         "concept_tags": set(),
    #         "confidence": 0.2,
    #     }

    #     for p in (parent_ids or []):
    #         if p in sem["families"]:
    #             sem["families"][p]["children"].add(fid)

    #     return fid

    # def _assign_token_family(self, tok, fid, strength=0.1, confidence=0.1):
    #     sem = self.semantic
    #     tsem = self._ensure_token_semantic(tok)
    #     fam = sem["families"].get(fid)
    #     if fam is None:
    #         return
    #     tsem["families"][fid] = {
    #         "strength": float(strength),
    #         "confidence": float(confidence),
    #         "age": 0,
    #         "last_reinforced": getattr(self, "generation_index", 0),
    #     }
    #     fam["members"][tok] = float(strength)

    # def detect_semantic_families(self, radius=0.4, min_members=4, max_families_per_gen=2):
    #     sem = self.semantic
    #     vecs = sem["vecs"]
    #     tmeta = sem["tokens"]

    #     tokens = [t for t in vecs.keys()
    #             if t not in self.numeric_semantic.keys()
    #             and t not in sem["concept_tokens"].values()
    #             and not self._is_identity_like(t)]

    #     if len(tokens) < min_members:
    #         return

    #     visited = set()
    #     made = 0

    #     def sqdist_no0(a, b):
    #         return sum((a[i] - b[i])**2 for i in range(1, len(a)))

    #     for i, t in enumerate(tokens):
    #         if t in visited:
    #             continue
    #         center = vecs[t]
    #         cluster = [t]
    #         visited.add(t)

    #         for u in tokens[i+1:]:
    #             if u in visited:
    #                 continue
    #             if sqdist_no0(center, vecs[u])**0.5 <= radius:
    #                 cluster.append(u)
    #                 visited.add(u)

    #         if len(cluster) < 4:
    #             continue

    #         dim = len(center)
    #         cv = [0.0]*dim
    #         for tok in cluster:
    #             v = vecs[tok]
    #             for d in range(dim):
    #                 cv[d] += v[d]
    #         cv = [x/len(cluster) for x in cv]

    #         vari = 0.0
    #         for tok in cluster:
    #             v = vecs[tok]
    #             vari += sum((v[d]-cv[d])**2 for d in range(1, dim))
    #         vari /= len(cluster)

    #         usage_vals = [tmeta[tok]["usage_count"] for tok in cluster if tok in tmeta]
    #         avg_usage = sum(usage_vals)/len(usage_vals) if usage_vals else 0

    #         dists = []
    #         for tok in cluster:
    #             d = (sqdist_no0(vecs[tok], cv) ** 0.5)
    #             dists.append(d)
    #         cohesion = 1.0 - (sum(dists)/len(dists))

    #         if vari > 0.015:
    #             continue
    #         if avg_usage < 20:
    #             continue
    #         if cohesion < 0.65:
    #             continue

    #         size_term = min(1.0, len(cluster)/12)
    #         freq_term = min(1.0, avg_usage/50)
    #         var_term  = max(0.0, 1.0 - 8*vari)
    #         raw = 0.3*size_term + 0.3*cohesion + 0.2*freq_term + 0.2*var_term
    #         conf = min(0.92, raw)

    #         fid = self._create_family(centroid=cv)
    #         sem["families"][fid]["confidence"] = conf

    #         for tok in cluster:
    #             self._assign_token_family(tok, fid, strength=0.4, confidence=conf)

    #         made += 1
    #         if made >= max_families_per_gen:
    #             break

    # def family_reinforcement_update(self, drift=0.02):
    #     sem = self.semantic
    #     vecs = sem["vecs"]
    #     fams = sem["families"]
    #     tokens = sem["tokens"]

    #     for fid, fam in fams.items():
    #         centroid = fam.get("centroid")
    #         if centroid is None:
    #             continue
    #         for tok, strength in fam["members"].items():
    #             if tok not in vecs or tok not in tokens:
    #                 continue
    #             v = vecs[tok]
    #             newv = [
    #                 a + drift * strength * (c - a)
    #                 for a, c in zip(v, centroid)
    #             ]
    #             vecs[tok] = newv

    # def family_soft_decay(self, decay=0.003, min_strength=0.02):
    #     sem = self.semantic
    #     fams = sem.get("families", {})
    #     tokens = sem.get("tokens", {})
    #     for fid, fam in fams.items():
    #         members = fam.get("members", {})
    #         for tok, strength in list(members.items()):
    #             new_s = strength - decay
    #             if new_s <= min_strength:
    #                 del members[tok]
    #                 if tok in tokens and fid in tokens[tok].get("families", {}):
    #                     del tokens[tok]["families"][fid]
    #             else:
    #                 members[tok] = new_s
    #                 if tok in tokens and fid in tokens[tok].get("families", {}):
    #                     tokens[tok]["families"][fid]["strength"] = new_s

    # # =====================================================
    # # FAMILY GOSSIP
    # # =====================================================
    # def name_families(self):
    #     fams = self.semantic["families"]
    #     for fid, fam in fams.items():
    #         if fam.get("name") is None:
    #             fam["name"] = self._invent_token(prefix="f", concept=True)

    # def export_family_snapshot(self, max_families=3):
    #     fams = self.semantic["families"]
    #     if not fams:
    #         return []
    #     ordered = sorted(
    #         fams.items(),
    #         key=lambda kv: kv[1].get("confidence", 0.0),
    #         reverse=True,
    #     )
    #     out = []
    #     for fid, fam in ordered[:max_families]:
    #         c = fam.get("centroid")
    #         if not c:
    #             continue
    #         out.append({
    #             "id": fid,
    #             "name": fam.get("name"),
    #             "centroid": list(c),
    #             "confidence": float(fam.get("confidence", 0.2)),
    #             "size": len(fam.get("members", {})),
    #         })
    #     return out

    # def maybe_broadcast_families(self):
    #     if not hasattr(self, "api") or self.api is None:
    #         return
    #     sem = self.semantic
    #     fams = sem["families"]
    #     if not fams:
    #         return
    #     ordered = sorted(
    #         fams.items(),
    #         key=lambda kv: kv[1].get("confidence", 0.0),
    #         reverse=True,
    #     )[:3]

    #     for fid, fam in ordered:
    #         conf = fam.get("confidence", 0.2)
    #         p = min(0.05 + 0.3 * conf, 0.25)
    #         if random.random() > p:
    #             continue
    #         if fam.get("name") is None:
    #             fam["name"] = self._invent_token(prefix="f", concept=True)
    #         c = fam.get("centroid")
    #         if not c:
    #             continue
    #         c_str = ",".join(f"{x:.3f}" for x in c)
    #         line = (
    #             f"A{self.id} teach_family name={fam['name']} "
    #             f"conf={conf:.3f} centroid={c_str}\n"
    #         )
    #         try:
    #             self.api.append_text("/family_gossip.txt", line, scope="world")
    #         except Exception:
    #             pass

    # def _integrate_family_gossip_line(self, line):
    #     if "teach_family" not in line:
    #         return
    #     name = None
    #     conf = 0.2
    #     centroid = None
    #     m = re.search(r"name=([^\s]+)", line)
    #     if m:
    #         name = m.group(1)
    #     m = re.search(r"conf=([0-9]*\.?[0-9]+)", line)
    #     if m:
    #         conf = float(m.group(1))
    #     m = re.search(r"centroid=([\-0-9\.,]+)", line)
    #     if m:
    #         vals = []
    #         for p in m.group(1).split(","):
    #             p = p.strip()
    #             if p:
    #                 try:
    #                     vals.append(float(p))
    #                 except ValueError:
    #                     pass
    #         if vals:
    #             centroid = vals
    #     if centroid is None:
    #         return
    #     foreign = {
    #         "name": name,
    #         "centroid": centroid,
    #         "confidence": conf,
    #     }
    #     self.integrate_foreign_family(foreign)

    # def integrate_foreign_family(self, foreign):
    #     fams = self.semantic["families"]
    #     if not fams:
    #         return
    #     centroid = foreign.get("centroid")
    #     if not centroid:
    #         return
    #     merge_radius = 0.30
    #     influence_radius = 0.55
    #     best_fid = None
    #     best_d2 = None

    #     def sqdist(a, b):
    #         return sum((x - y)**2 for x, y in zip(a, b))

    #     for fid, fam in fams.items():
    #         c = fam.get("centroid")
    #         if not c:
    #             continue
    #         d2 = sqdist(c, centroid)
    #         if best_d2 is None or d2 < best_d2:
    #             best_fid = fid
    #             best_d2 = d2

    #     if best_fid is None:
    #         return

    #     dist = best_d2 ** 0.5
    #     fam = fams[best_fid]
    #     if dist <= merge_radius:
    #         new_c = [
    #             0.8 * lc + 0.2 * fc
    #             for lc, fc in zip(fam["centroid"], centroid)
    #         ]
    #         fam["centroid"] = new_c
    #         fam["confidence"] = min(1.0, fam.get("confidence", 0.3) + 0.1)
    #         return
    #     if dist <= influence_radius:
    #         parents = fam.setdefault("parents", [])
    #         parents.append({
    #             "name": foreign.get("name"),
    #             "confidence": foreign.get("confidence", 0.2),
    #         })
    #         fam["confidence"] = min(1.0, fam.get("confidence", 0.3) + 0.02)
    #         return

    # def prune_families(self, limit=250):
    #     fams = self.semantic["families"]
    #     if len(fams) <= limit:
    #         return
    #     ranked = sorted(
    #         fams.items(),
    #         key=lambda kv: (
    #             kv[1].get("confidence", 0.0),
    #             len(kv[1].get("members", {}))
    #         )
    #     )
    #     remove_n = int(len(fams) * 0.20)
    #     for fid, _ in ranked[:remove_n]:
    #         del fams[fid]

    # =====================================================
    # EXPECTATION
    # =====================================================
    def get_overall_expectation(self):
        expectation = float(getattr(self, "numeric_bias", 0.0))
        assoc = self.utterance_memory.get("associations", {})
        if assoc:
            expectation = 0.7*expectation + 0.3*(sum(assoc.values())/len(assoc))
        return max(-1.0, min(1.0, expectation))

    # =====================================================
    # FEEDBACK
    # =====================================================
    def get_utterance_expectation(self, utterance):
        return float(self.utterance_memory["associations"].get(utterance, 0.0))

    def learn_from_feedback(self, utterance, reward, lr=0.1):
        if not utterance:
            return
        r = max(-1.0, min(+1.0, reward))
        mem = self.utterance_memory["associations"]
        old = mem.get(utterance, 0.0)
        mem[utterance] = max(-1.0, min(1.0, old + lr*r))
        for k in list(mem.keys()):
            mem[k] *= 0.999

    # =====================================================
    # SEMANTIC GAP DETECTION (vs community map)
    # =====================================================
    # def detect_semantic_gaps(
    #     self,
    #     community_map,
    #     min_usage=5,
    #     min_comm_count=3,
    #     dist_thresh=0.6,
    #     max_gaps=20,
    # ):
    #     if not community_map:
    #         return self.semantic_gaps
    #     c_vecs = community_map.get("vecs", {})
    #     c_counts = community_map.get("counts", {})
    #     if not c_vecs:
    #         return self.semantic_gaps
    #     vecs = self.semantic.get("vecs", {})
    #     tokens_meta = self.semantic.get("tokens", {})
    #     misaligned = []
    #     orphans = []
    #     for tok, meta in tokens_meta.items():
    #         usage = meta.get("usage_count", 0)
    #         if usage < min_usage:
    #             continue
    #         if self._is_identity_like(tok):
    #             continue
    #         if tok not in vecs:
    #             continue
    #         v_local = np.array(vecs[tok], dtype=float)
    #         comm_count = c_counts.get(tok, 0)
    #         if tok in c_vecs and comm_count >= min_comm_count:
    #             v_comm = np.array(c_vecs[tok], dtype=float)
    #             dist = float(np.linalg.norm(v_local - v_comm))
    #             if dist >= dist_thresh:
    #                 misaligned.append({
    #                     "token": tok,
    #                     "dist": dist,
    #                     "usage": int(usage),
    #                     "community_count": int(comm_count),
    #                 })
    #         else:
    #             orphans.append({
    #                 "token": tok,
    #                 "usage": int(usage),
    #                 "community_count": int(comm_count),
    #             })
    #     misaligned.sort(
    #         key=lambda d: (d["dist"] * max(1, d["usage"])),
    #         reverse=True,
    #     )
    #     orphans.sort(
    #         key=lambda d: d["usage"],
    #         reverse=True,
    #     )
    #     misaligned = misaligned[:max_gaps]
    #     orphans = orphans[:max_gaps]
    #     self.semantic_gaps = {
    #         "misaligned": misaligned,
    #         "orphans": orphans,
    #         "last_gen": getattr(self, "current_generation", -1),
    #     }
    #     return self.semantic_gaps

    # =====================================================
    # FLAVOUR HOMEOSTASIS v2 (Soft Prefix/Family Control)
    # =====================================================
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

        vecs = self.semantic.get("vecs", {})
        tokens_meta = self.semantic.get("tokens", {})
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

        self.semantic.setdefault("flavour_homeostasis", {})
        self.semantic["flavour_homeostasis"] = {
            "dominant": list(dominant),
            "scarce": list(scarce),
            "freqs": fam_freq,
        }

    # =====================================================
    # PHASE C: Agents propose semantic-alignment tasks
    # =====================================================
    # def maybe_propose_semantic_tasks(self, coordinator, max_tasks=1):
    #     if not hasattr(self, "semantic_gaps"):
    #         return
    #     if self.semantic_gaps.get("last_gen") != getattr(self, "current_generation", -1):
    #         return
    #     mis = self.semantic_gaps.get("misaligned", [])
    #     orp = self.semantic_gaps.get("orphans", [])
    #     options = []
    #     for x in mis:
    #         options.append(("misaligned", x))
    #     for x in orp:
    #         options.append(("orphan", x))
    #     if not options:
    #         return
    #     random.shuffle(options)
    #     todo = options[:max_tasks]
    #     for kind, item in todo:
    #         tok = item["token"]
    #         sem = self.semantic
    #         vec_local = sem["vecs"].get(tok)
    #         vec_comm = coordinator.community_semantic["vecs"].get(tok)
    #         if vec_local is None:
    #             continue
    #         task = {
    #             "task_id": coordinator._generate_task_id(),
    #             "task_type": "semantic_alignment",
    #             "token": tok,
    #             "proposer": self.id,
    #             "kind": kind,
    #             "vector_local": list(vec_local),
    #             "community_vector": list(vec_comm or []),
    #         }
    #         coordinator.add_semantic_alignment_task(task)

    # def _maybe_learn_numeric_from(self, teacher):
    #     if not hasattr(self, "numeric_semantic"):
    #         return
    #     if not hasattr(teacher, "numeric_semantic"):
    #         return
    #     my_score = self._numeric_collision_score()
    #     their_score = teacher._numeric_collision_score()
    #     if their_score < my_score:
    #         for n, tok in teacher.numeric_semantic.items():
    #             self.numeric_semantic[n] = tok
    #         if hasattr(self, "trust_channels"):
    #             self.trust_channels[teacher.id] = \
    #                 min(1.0, self.trust_channels.get(teacher.id, 0.0) + 0.05)
    #         if hasattr(self, "motivations"):
    #             self.motivations["esteem"] = min(
    #                 1.0, self.motivations.get("esteem", 0.5) + 0.03
    #             )
    #         if teacher.traits.get("teaching_drive", 0) > 0.4:
    #             teacher.motivations["esteem"] = min(
    #                 1.0, teacher.motivations.get("esteem", 0.5) + 0.02
    #             )

    # =====================================================
    # SOCIAL DIALOGUE BEHAVIOUR
    # =====================================================
    def _get_trust_to(self, other_id):
        tc = getattr(self, "trust_channels", None)
        if not isinstance(tc, dict):
            return 0.0
        prof = tc.get(other_id)
        if not isinstance(prof, dict):
            return float(prof) if isinstance(prof, (int, float)) else 0.0
        vals = [v for v in prof.values() if isinstance(v, (int, float))]
        if not vals:
            return 0.0
        trust_scalar = sum(vals) / len(vals)
        return max(-1.0, min(1.0, trust_scalar))

    # def _semantic_similarity_to(self, other_id):
    #     if not hasattr(self, "identity_token"):
    #         return 0.0
    #     my_tok = self.identity_token
    #     other_tok = self.get_or_create_nickname_for_id(other_id)
    #     if other_tok not in self.semantic["vecs"]:
    #         return 0.0
    #     return 1.0 - self.semantic_distance(my_tok, other_tok)

    def _base_talk_probability(self):
        chattiness = self.traits.get("chattiness", 0.5)
        curiosity = self.traits.get("curiosity", 0.5)
        p = 0.05 + 0.25 * chattiness + 0.20 * curiosity
        mot = getattr(self, "motivation", None)
        if isinstance(mot, dict) and "social" in mot:
            social = mot.get("social", 0.5)
            p *= (0.5 + social)
        return self._clamp(p, 0.02, 0.60)

    def _social_affinity(self, other):
        trust = self._get_trust_to(other.id)
        sem = self.semantic_system._semantic_similarity_to(other.id)
        if hasattr(self, "identity_token"):
            ident = self._semantic_similarity_to(other.id)
        else:
            ident = 0.0
        curiosity = self.traits.get("curiosity", 0.5)
        noise = random.uniform(-0.1, 0.1)
        score = (
            0.45 * trust +
            0.35 * sem +
            0.20 * ident +
            noise
        )
        return score

    def choose_conversation_partner(self, agents):
        if random.random() > self._base_talk_probability():
            return None
        best_id = None
        best_score = None
        for other in agents:
            if other is self:
                continue
            s = self._social_affinity(other)
            if best_score is None or s > best_score:
                best_score = s
                best_id = other.id
        if best_id is None or best_score is None or best_score < 0.05:
            return None
        return best_id

    def receive_message(self, from_id, utterance):
        if not utterance:
            return
        toks = [t for t in utterance.split() if t.strip()]
        if not toks:
            return
        try:
            self._observe_nickname_use(from_id, utterance)
        except Exception:
            pass
        try:
            self._observe_language_tokens(toks, gain=0.05)
        except Exception:
            pass
        tc = getattr(self, "trust_channels", None)
        if isinstance(tc, dict):
            tc[from_id] = self._clamp(tc.get(from_id, 0.0) + 0.01, -1.0, 1.0)
        mot = getattr(self, "motivation", None)
        if isinstance(mot, dict) and "social" in mot:
            mot["social"] = self._clamp(mot.get("social", 0.5) + 0.02, 0.0, 1.0)
        if not hasattr(self, "dialogue_stats"):
            self.dialogue_stats = {"turns_seen": 0, "turns_spoken": 0}
        self.dialogue_stats["turns_seen"] += 1
        if isinstance(tc, dict):
            prof = tc.get(from_id)
            if isinstance(prof, dict):
                trust_val = sum(prof.values()) / len(prof)
                drift = 0.02 * trust_val
                teacher = next((a for a in self.coordinator.agents if a.id == from_id), None)
                if teacher:
                    self._identity_social_drift(teacher, drift)

    def mark_spoken_turn(self):
        if not hasattr(self, "dialogue_stats"):
            self.dialogue_stats = {"turns_seen": 0, "turns_spoken": 0}
        self.dialogue_stats["turns_spoken"] += 1

    def _identity_social_drift(self, other, amount=0.01):
        if not hasattr(self, "identity_token") or not hasattr(other, "identity_token"):
            return
        vecs = self.semantic["vecs"]
        a = vecs.get(self.identity_token)
        b = vecs.get(other.identity_token)
        if a is None or b is None:
            return
        new = []
        for x, y in zip(a, b):
            new.append(x + amount * (y - x))
        vecs[self.identity_token] = new
        self.identity_vec = new

    def get_or_create_nickname_for_id(self, other_id: int) -> str:
        tok = f"agent_{other_id}"
        self.nicknames_for_others[other_id] = tok
        self.vocab.add(tok)
        self.semantic_system._ensure_vec(tok)
        id_tok = f"agent_{other_id}"
        base_vec = self.semantic["vecs"].get(id_tok)
        if base_vec is not None:
            v = self.semantic["vecs"][tok]
            self.semantic["vecs"][tok] = [
                a + 0.3 * (b - a) for a, b in zip(v, base_vec)
            ]
        return tok

    def get_or_create_nickname_for_agent(self, other):
        if isinstance(other, int):
            oid = other
        else:
            oid = other.id
        return self.get_or_create_nickname_for_id(oid)

    def _observe_nickname_use(self, from_id: int, utterance: str):
        if not utterance:
            return
        tokens = utterance.split()
        if not tokens:
            return
        first = tokens[0]
        if first.startswith("agent_") and first[6:].isdigit():
            self.heard_nicknames[from_id].add(first)

    def produce_addressed_utterance(self, listener) -> str:
        base = self.produce_utterance() or ""
        if not listener or random.random() < 0.4:
            return base
        try:
            nick = self.get_or_create_nickname_for_agent(listener)
        except Exception:
            return base
        if not nick:
            return base
        if base:
            return f"{nick} {base}"
        else:
            return nick


# Backwards-compat: allow old imports to work if any remain
LanguageMixinV2 = LanguageMixin