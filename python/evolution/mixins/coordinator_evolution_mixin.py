# file: evolution/coordinator.py
import math
import random
import re
import json
import secrets
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from agents.agent import Agent
import config as project_config
import evolution.coordinator_settings as coordinator_settings
from evolution.challenge import ChallengeSystem
from agents.cognition.numeric_system import NumericSystem
from evolution.logging import compute_generation_summary, append_generation_to_csv, write_generation_report, _get_log_filenames
from evolution.programs import run_program, safe
from evolution.mixins.global_registry import GlobalTokenRegistry
from evolution.community_lexicon import CommunityLexicon
from evolution.human_dictionary import HumanDictionary
from evolution.community_reading import CommunityReadingRoom
from evolution.coordinator_settings import (
    POP_SIZE,
    ENABLE_DICTIONARY_INJECTION,
    UTTER_EFFECT,
    MATE_POOL_SIZE,
    COMPAT_WEIGHT,
    FITNESS_WEIGHT,
    MEMORY_WEIGHT,
    DIVERSITY_WEIGHT,
    OUTCOME_SCALE,
    MAX_COOP_BONUS,
    MAX_NOVELTY_BONUS,
    ENERGY_MAX,
    IDLE_TAX,
    MIN_PARTICIPATION,
)
from evolution.mixins.coordinator_task_mixin import CoordinatorTaskMixin
from evolution.mixins.coordinator_language_mixin import CoordinatorLanguageMixin
from evolution.community_conversation import CommunityConversation

from evolution.behaviours.orchestration import orchestrate_action


# If your phase3/__init__.py re-exports these, this import works:
from phase3 import SandboxSpec, build_sandbox
from phase3.agent_api import AgentAPI


class CoordinatorEvolutionMixin:
    def _clamp(self, v, lo=0.0, hi=1.0):
        return max(lo, min(hi, v))

    def agent_by_id(self, agent_id):
        for ag in self.agents:
            if ag.id == agent_id:
                return ag
        return None

    def damp_trait(self, value, strength=0.05):
        # pull traits gently toward neutral to avoid absorbing extremes
        return value + (0.5 - value) * strength

    def _trait_vec(self, a):
        return (
            a.traits["cooperation_weight"],
            a.traits["novelty_weight"],
            a.traits["stability_weight"],
            a.traits["mutation_rate"],
            a.traits["trust_threshold"],
        )

    def _euclid(self, u, v):
        return sum((x - y) ** 2 for x, y in zip(u, v)) ** 0.5

    def _cosine(self, u, v):
        """Safe cosine similarity in [-1,1]."""
        if not u or not v:
            return 0.0
        m = min(len(u), len(v))
        if m == 0:
            return 0.0
        u = u[:m]
        v = v[:m]
        num = sum(a * b for a, b in zip(u, v))
        du = math.sqrt(sum(a * a for a in u))
        dv = math.sqrt(sum(b * b for b in v))
        if du == 0.0 or dv == 0.0:
            return 0.0
        val = num / (du * dv)
        # clamp just in case of numerical noise
        return max(-1.0, min(1.0, val))

    def _compatibility(self, a, b):
        # higher is better → use negative distance
        return -self._euclid(self._trait_vec(a), self._trait_vec(b))

    def _blend_assoc(self, *parents, noise=0.02, decay=0.95):
        """
        Merge utterance associations across parents.
        Returns a dict(utterance -> value in [-1,1]).
        """
        keys = set()
        for p in parents:
            keys |= set(p.utterance_memory["associations"].keys())
        out = {}
        for k in keys:
            vals = []
            for p in parents:
                vals.append(float(p.utterance_memory["associations"].get(k, 0.0)))
            mean = sum(vals) / max(1, len(vals))
            mean = mean * decay + random.uniform(-noise, noise)
            out[k] = max(-1.0, min(1.0, mean))
        return out

    def evaluate_agents(self):
        for a in self.agents:
            # ✅ Skip callable programs + empty genomes
            if callable(a.program) or not a.program:
                raw = 0.0
            else:
                raw = run_program(a.program)

            if raw != raw:  # NaN protection
                raw = 0.0
            raw = max(-1e6, min(raw, 1e6))

            # sanitize
            if raw != raw:  # NaN
                raw = 0.0
            raw = max(-1e6, min(raw, 1e6))

            behaviour = math.tanh(raw / 5000.0)  # [-1,1]
            prog_score = behaviour * 10.0        # cap at +/-10

            own = prog_score
            own = max(-50.0, min(own, 50.0))     # safety clamp

            prev = a.memory.get("last_fitness", 0.0)
            a.memory["last_fitness_change"] = own - prev
            a.memory["last_fitness"] = own
            a.own_fitness = own

    def apply_cooperation(self):
        if not self.agents:
            return
        mean_f = sum(a.own_fitness for a in self.agents) / len(self.agents)
        for a in self.agents:
            coop = a.traits["cooperation_weight"]
            raw = coop * (a.own_fitness - mean_f)
            bonus = 0.05 * raw
            bonus = max(-MAX_COOP_BONUS, min(bonus, MAX_NOVELTY_BONUS))
            a.cooperation_bonus = bonus

            if coop > 0.95:
                a.cooperation_bonus *= 0.5

            if bonus > 0:
                a.memory["cooperation_success"] += bonus

    def apply_novelty(self):
        fitnesses = [a.own_fitness for a in self.agents]
        if not fitnesses:
            return
        mean_f = sum(fitnesses) / len(fitnesses)
        distances = [abs(f - mean_f) for f in fitnesses]
        cutoff = sorted(distances)[int(0.7 * len(self.agents))] if self.agents else 0.0

        for a in self.agents:
            raw = abs(a.own_fitness - mean_f)
            raw = min(raw, cutoff)
            compressed = math.tanh(raw * 0.005)  # 0..1
            nov = compressed * a.traits["novelty_weight"] * 0.5
            excess = max(0.0, a.traits["novelty_weight"] - 0.7)
            nov -= excess * 0.4
            nov = max(0.0, min(nov, MAX_NOVELTY_BONUS))
            a.novelty_bonus = nov

            if nov > 0:
                a.memory["novelty_success"] += nov

    def compute_total_fitness(self):
        for a in self.agents:
            own  = safe(a.own_fitness,       lo=0,   hi=300)
            coop = safe(getattr(a, "cooperation_bonus", 0.0), lo=-5, hi=5)
            nov  = safe(getattr(a, "novelty_bonus", 0.0),     lo=0,  hi=5)

            total = own + coop + nov

            # ------------------------------------------
            # NEW: Trust-openness + social-degree bonus
            # ------------------------------------------
            try:
                # 0..1: low threshold = more open
                tt = float(a.traits.get("trust_threshold", 0.5))
                tt = max(0.0, min(1.0, tt))
                openness = 1.0 - tt

                coop_trait = float(a.traits.get("cooperation_weight", 0.5))
                coop_trait = max(0.0, min(1.0, coop_trait))

                partners = len(getattr(a, "social_memory", {}) or {})
                social_factor = math.log1p(partners) if partners > 0 else 0.0
                social_factor = max(0.0, min(1.5, social_factor))

                trust_bonus = openness * coop_trait * social_factor * 3.0
                total += safe(trust_bonus, lo=-2, hi=5)
            except Exception:
                pass

            a.total_fitness = safe(total, lo=0, hi=300)

    def _social_bias(self, chooser, candidate):
        rec = chooser.social_memory.get(candidate.id)
        if rec is None:
            return 0.0
        base = (rec["mean_outcome"] + 0.05 * rec["trust_delta"] + 0.5 * rec["offspring_success"])
        return chooser.memory_influence * base

    def _reputation_bias(self, chooser, candidate):
        acc = 0.0
        count = 0
        for other in self.agents:
            if other.id in (chooser.id, candidate.id):
                continue
            rec_sender = chooser.social_memory.get(other.id)
            sender_weight = 1.0
            if rec_sender is not None:
                sender_weight += min(2.0, max(-0.5, rec_sender["trust_delta"] * 0.005))
            rec = other.social_memory.get(candidate.id)
            if rec is None:
                continue
            rep = 0.5 * rec["mean_outcome"] + 0.3 * rec["trust_delta"] * 0.01 + 0.2 * rec["offspring_success"]
            acc += sender_weight * rep
            count += 1
        return (acc / count) if count else 0.0

    def select_parents(self, elites):
        """
        Tri-parent selection with biases.
        Slightly bias chooser toward the fittest elite when many deaths occurred.
        """
        chooser_pool = elites if elites else self.agents
        if not chooser_pool:
            chooser = random.choice(self.agents)
        else:
            # 60% chance pick from top 20% of elites to favour best performers
            k = max(1, int(0.2 * len(chooser_pool)))
            if random.random() < 0.6:
                chooser = random.choice(chooser_pool[:k])
            else:
                chooser = random.choice(chooser_pool)

        if len(self.agents) >= 3:
            pool = random.sample(self.agents, min(MATE_POOL_SIZE, len(self.agents)))
        else:
            return random.sample(self.agents, min(3, len(self.agents)))
        pool = [a for a in pool if a.id != chooser.id]
        if len(pool) < 2:
            return random.sample(self.agents, 3)

        fit_map = {a.id: a.total_fitness for a in self.agents}
        scored = []

        for cand in pool:
            try:
                s_fit = FITNESS_WEIGHT * fit_map.get(cand.id, 0.0)
                s_comp = COMPAT_WEIGHT * self._compatibility(chooser, cand)
                s_mem = MEMORY_WEIGHT * (self._social_bias(chooser, cand) + self._social_bias(cand, chooser))

                # Diversity bias: prefer candidates different from chooser and pop mean
                trait_means = {
                    k: sum(a.traits[k] for a in self.agents) / len(self.agents)
                    for k in ["cooperation_weight", "novelty_weight", "stability_weight",
                              "mutation_rate", "trust_threshold"]
                }
                pop_dist = sum(abs(cand.traits[k] - trait_means[k]) for k in trait_means)
                s_div = DIVERSITY_WEIGHT * (pop_dist - self._euclid(self._trait_vec(chooser), self._trait_vec(cand)))

                # Language bias
                s_utt = 0.0
                if cand.id in self.last_utterances:
                    utt = self.last_utterances[cand.id]
                    exp = chooser.get_utterance_expectation(utt)
                    s_utt = UTTER_EFFECT * exp

                total = s_fit + s_comp + s_mem + self._reputation_bias(chooser, cand) * 0.5 + s_div + s_utt
            except Exception:
                total = -1e9

            scored.append((cand, total))

        if not scored:
            return random.sample(self.agents, 3)

        scored.sort(key=lambda x: x[1], reverse=True)
        p2 = scored[0][0]
        p3 = scored[1][0] if len(scored) > 1 else random.choice(self.agents)

        # Social memory update
        try:
            norm = max(1.0, abs(chooser.total_fitness) + abs(p2.total_fitness) + abs(p3.total_fitness))
            chooser.remember_interaction(p2.id, outcome=(p2.total_fitness / norm), gen_index=self.generation_index)
            chooser.remember_interaction(p3.id, outcome=(p3.total_fitness / norm), gen_index=self.generation_index)
            p2.remember_interaction(chooser.id, outcome=(chooser.total_fitness / norm), gen_index=self.generation_index)
            p3.remember_interaction(chooser.id, outcome=(chooser.total_fitness / norm), gen_index=self.generation_index)
        except Exception:
            pass

        return [chooser, p2, p3]

    def make_child(self, p1, p2, p3, new_id):
        """
        Tri-parent child creation with full semantic, linguistic, numeric,
        and trait inheritance — robust against weird link row types.

        Child inherits:
            - traits (blended)
            - vocab (union)
            - semantic vecs (averaged with noise)
            - co-occurrence links (merged safely)
            - family definitions + counters
            - utterance associations (blended)
            - symbol maps + counting system
            - recent tokens
            - dictionary vocab

        Child starts with:
            - 75% energy
            - empty social memory
            - identity token 'a{new_id}' correctly grounded
        """

        child = Agent(id = new_id, token_registry=self.token_registry)
        child.community_lexicon = self.community_lexicon
        child.human_dictionary = self.human_dictionary

        # ---------------------------------------------------
        # 1) TRAITS
        # ---------------------------------------------------
        for key in child.traits:
            inherited = random.choice([
                p1.traits[key], p2.traits[key], p3.traits[key]
            ])
            inherited = self.damp_trait(inherited, strength=0.08)
            inherited += random.uniform(-0.01, 0.01)
            child.traits[key] = self._clamp(inherited, 0.0, 1.0)

        child.memory_influence = random.choice(
            [p1.memory_influence, p2.memory_influence, p3.memory_influence]
        )
        child.memory_decay_rate = self._clamp(
            random.choice([
                p1.memory_decay_rate,
                p2.memory_decay_rate,
                p3.memory_decay_rate
            ]) + random.uniform(-0.02, 0.02),
            0.0, 1.0
        )

        # ---------------------------------------------------
        # 2) VOCAB INHERITANCE
        # ---------------------------------------------------
        inherited_vocab = set().union(p1.vocab, p2.vocab, p3.vocab)
        public_vocab = set(self.community_lexicon.numeric_conventions().values())
        public_vocab.update(self.community_lexicon.referential_conventions().values())

        def vocabulary_score(token):
            score = 8.0 if token in public_vocab else 0.0
            for parent in (p1, p2, p3):
                score = max(
                    score,
                    float(getattr(parent, "utter_bias", {}).get("symbol_preferences", {}).get(token, 0.0)),
                )
                if token in getattr(parent, "recent_tokens", [])[-40:]:
                    score += 1.0
            return score

        # Inherit a compact active repertoire rather than the unbounded union
        # of every historical token.  Public conventions are always retained.
        child.vocab = set(sorted(
            inherited_vocab,
            key=lambda token: (-vocabulary_score(token), str(token)),
        )[:80])
        child.vocab.update(public_vocab)
        child.dict_vocab = set().union(
            p1.dict_vocab, p2.dict_vocab, p3.dict_vocab
        )

        # ---------------------------------------------------
        # 3) SEMANTIC SYSTEM INITIALISATION
        # ---------------------------------------------------
        child._init_semantic_system()
        csem = child.semantic
        vsem = csem["vecs"]
        lsem = csem["links"]

        parent_vecs = [p1.semantic["vecs"], p2.semantic["vecs"], p3.semantic["vecs"]]
        parent_links = [p1.semantic["links"], p2.semantic["links"], p3.semantic["links"]]

        # ----------------------------
        # 3A. Vectors
        # ----------------------------
        child.vocab = {t for t in child.vocab if isinstance(t, str) and t.strip()}
        for tok in child.vocab:
            vecs = []
            for pv in parent_vecs:
                if tok in pv:
                    vecs.append(pv[tok])

            if vecs:
                base = np.array(vecs[0], dtype=float)
                for v in vecs[1:]:
                    base += np.array(v, dtype=float)
                base /= len(vecs)
            else:
                base = np.array(child._rand_vec(32), dtype=float)

            base += np.random.normal(scale=0.02, size=base.shape)

            # --- NEW: nudge toward community centroid for high-confidence tokens ---
            try:
                com = getattr(self, "community_semantic", None)
                if com is not None:
                    c_vecs = com.get("vecs", {})
                    c_conf = com.get("confidence", {})
                    if tok in c_vecs:
                        conf = float(c_conf.get(tok, 0.0))
                        if conf > 0.6:  # only when community strongly agrees
                            gamma_low = 0.05
                            gamma_high = 0.25
                            gamma = gamma_low + (gamma_high - gamma_low) * ((conf - 0.6) / 0.4)
                            gamma = max(gamma_low, min(gamma_high, gamma))
                            cvec = np.array(c_vecs[tok], dtype=float)
                            base = (1.0 - gamma) * base + gamma * cvec
            except Exception:
                pass

            vsem[tok] = base.tolist()

        if None in vsem:
            del vsem[None]
        if None in lsem:
            del lsem[None]

        # ----------------------------
        # 3B. Links (safe merge)
        # ----------------------------

        # First: normalise any weird existing rows in child's links
        for a, nbrs in list(lsem.items()):
            if not isinstance(nbrs, defaultdict):
                lsem[a] = defaultdict(lambda: 0.0, nbrs)

        def _extract_weight(entry):
            """Handle both float and dict-style link entries."""
            if isinstance(entry, dict):
                return float(entry.get("w", 0.0))
            return float(entry)

        def _merge_link_entry(existing, incoming_weight, incoming_entry=None):
            """
            Merge an existing link value with an incoming one.

            existing: float or dict or None
            incoming_weight: float (already extracted)
            incoming_entry: dict or None (original incoming link)
            """
            base_w = 0.0
            if existing is not None:
                if isinstance(existing, dict):
                    base_w = float(existing.get("w", 0.0))
                else:
                    base_w = float(existing)

            new_w = base_w + incoming_weight * (1.0 / 3.0)

            # If either side is dict-style, return a dict-style entry
            if isinstance(existing, dict) or isinstance(incoming_entry, dict):
                out = existing.copy() if isinstance(existing, dict) else {}
                # merge usefulness if present
                uses = []
                if isinstance(existing, dict) and "use" in existing:
                    uses.append(existing["use"])
                if isinstance(incoming_entry, dict) and "use" in incoming_entry:
                    uses.append(incoming_entry["use"])
                if uses:
                    out["use"] = sum(uses) / len(uses)

                # age: child starts "young" – we can reset or lightly blend
                if isinstance(existing, dict) and "age" in existing:
                    out["age"] = max(0, int(existing["age"]))
                else:
                    out["age"] = 0

                out["w"] = new_w
                return out

            # plain float mode
            return new_w

        # Then: merge parent link graphs
        for plinks in parent_links:
            for a, nbrs in plinks.items():
                # Ensure row is a defaultdict
                row = lsem.get(a)
                if row is None or not isinstance(row, defaultdict):
                    row = lsem[a] = defaultdict(lambda: 0.0, row or {})

                for b, entry in nbrs.items():
                    # extract numeric weight from parent entry
                    incoming_w = _extract_weight(entry)
                    existing = row.get(b)
                    row[b] = _merge_link_entry(existing, incoming_w, entry)

        # prune tiny edges (supports float + dict)
        for a, nbrs in list(lsem.items()):
            for b, entry in list(nbrs.items()):
                if isinstance(entry, dict):
                    w = float(entry.get("w", 0.0))
                else:
                    w = float(entry)

                if abs(w) < 1e-6:
                    del nbrs[b]

            if not nbrs:
                del lsem[a]

        # ---------------------------------------------------
        # 4) FAMILY / CONCEPT INHERITANCE
        # ---------------------------------------------------
        f1 = p1.semantic.get("families", {})
        f2 = p2.semantic.get("families", {})
        f3 = p3.semantic.get("families", {})

        merged_families = {}
        for fams in (f1, f2, f3):
            for fid, spec in fams.items():
                merged_families[fid] = spec

        fam_ids = []
        for fid in merged_families.keys():
            try:
                fam_ids.append(int(fid))
            except Exception:
                pass

        child.semantic.setdefault("families", {})
        if isinstance(child.semantic["families"], dict):
            child.semantic["families"].clear()
            child.semantic["families"].update(merged_families)
        else:
            child.semantic["families"] = dict(merged_families)

        child.semantic["family_counter"] = (max(fam_ids) + 1) if fam_ids else 1
        if hasattr(child, "semantic_system") and hasattr(child.semantic_system, "family_system"):
            child.semantic_system.family_system.families = child.semantic["families"]
            child.semantic_system.families = child.semantic["families"]
            try:
                child.semantic_system.family_system.family_counter = int(child.semantic["family_counter"])
            except Exception:
                pass

        # ---------------------------------------------------
        # 5) LINGUISTIC MEMORY / ASSOCIATIONS
        # ---------------------------------------------------
        child.utterance_memory["associations"] = self._blend_assoc(
            p1, p2, p3,
            noise=0.02,
            decay=0.90
        )
        child.utterance_memory["usage_count"].clear()
        child.referent_lexicon = {}
        child.action_lexicon = {}
        for parent in (p1, p2, p3):
            child.referent_lexicon.update(getattr(parent, "referent_lexicon", {}))
            child.action_lexicon.update(getattr(parent, "action_lexicon", {}))

        # ---------------------------------------------------
        # 6) NUMERIC-SYMBOL + COUNTING SYSTEM
        # ---------------------------------------------------
        child.symbol_map = {}
        for m in (p1.symbol_map, p2.symbol_map, p3.symbol_map):
            for k, v in m.items():
                child.symbol_map[k] = v

        cs = NumericSystem(owner=child)
        cs.merge_from(p1.counting, p2.counting, p3.counting)
        child.numeric_system = cs
        child.counting = cs
        child.counting.set_symbol_map(child.symbol_map)

        # ---------------------------------------------------
        # 7) RECENT TOKENS
        # ---------------------------------------------------
        merged_recent = (
            p1.recent_tokens[-8:] +
            p2.recent_tokens[-8:] +
            p3.recent_tokens[-8:]
        )
        child.recent_tokens = merged_recent[-16:]

        # ---------------------------------------------------
        # 8) SOCIAL MEMORY RESET
        # ---------------------------------------------------
        child.social_memory.clear()

        # ---------------------------------------------------
        # 9) ENERGY
        # ---------------------------------------------------
        child.energy = ENERGY_MAX * 0.75

        # ---------------------------------------------------
        # 10) IDENTITY — FINAL STEP
        # ---------------------------------------------------
        child.name_token = f"a{new_id}"
        if hasattr(child, "identity_system"):
            child.identity_system.mark_identity_token(child.name_token)
        else:
            child.identity_tokens.add(child.name_token)
            child.mark_identity_token(child.name_token)

        return child

    def apply_memory_decay(self):
        for a in self.agents:
            decay = 1.0 - a.memory_decay_rate
            for key, val in a.memory.items():
                if isinstance(val, (int, float)):
                    a.memory[key] = val * decay
