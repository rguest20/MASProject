# file: evolution/coordinator.py
import math
import random
import re
import json
import secrets
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from agents.agent import Agent
import config as project_config
import evolution.coordinator_settings as coordinator_settings
from evolution.challenge import ChallengeSystem
from agents.cognition.numeric_system import NumericSystem
from evolution.logging import compute_generation_summary, append_generation_to_csv, write_generation_report, _get_log_filenames
from evolution.programs import run_program, safe, mutate_program
from evolution.mixins.global_registry import GlobalTokenRegistry
from evolution.community_lexicon import CommunityLexicon
from evolution.coordinator_settings import (
    POP_SIZE,
    ELITE_RATIO,
    ENABLE_DICTIONARY_INJECTION,
    UTTER_CHANCE,
    UTTER_EFFECT,
    REWARD_EPS,
    REWARD_TEMP,
    PHASE2_LR,
    TASK_SEMANTIC_ALIGNMENT,
    MATE_POOL_SIZE,
    COMPAT_WEIGHT,
    FITNESS_WEIGHT,
    MEMORY_WEIGHT,
    DIVERSITY_WEIGHT,
    GOSSIP_PAIRS_PER_GEN,
    TEACH_PROB,
    TEACH_TOP_FRACTION,
    TEACH_PAIRS_PER_PASS,
    TEACH_ROUNDS_PER_GEN,
    TEACH_ACC_TEMP,
    IMPROVE_ONLY_TEACH,
    OUTCOME_SCALE,
    REFERENTIAL_BONUS,
    MAX_COOP_BONUS,
    MAX_NOVELTY_BONUS,
    MAX_FIT,
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

import time

def PERF(msg):
    print(f"[PERF {time.time():.3f}] {msg}", flush=True)


class Coordinator(CoordinatorTaskMixin, CoordinatorLanguageMixin):
    def __init__(self, run_dir=None, seed=None, conversation_path=None):
        """Create an isolated, reproducible simulation run.

        Each coordinator owns its reports and sandbox state under a unique
        directory so a new experiment cannot append to or overwrite an older
        one.  ``run_dir`` is available for callers that need to choose the
        destination explicitly; it must not already exist.
        """
        self.random_seed = int(seed) if seed is not None else secrets.randbits(64)
        self.run_dir = self._create_run_dir(run_dir)
        self.report_path = self.run_dir / "generation_report.txt"
        self.csv_path = self.run_dir / "cultural_log.csv"
        self.dialogue_log_path = self.run_dir / "dialogue_log.txt"
        self._write_run_metadata()

        random.seed(self.random_seed)
        np.random.seed(self.random_seed % (2 ** 32))

        self.semantic_alignment_tasks = []
        self.token_registry = GlobalTokenRegistry()
        self.community_lexicon = CommunityLexicon()

        self.cached_dictionary_words = None
        self.semantic_seeds = {"words": [], "synonyms": [], "antonyms": []}
        self.semantic_seeds = self.build_semantic_seeds()
        self.cached_dictionary_words = self.extract_dictionary_words(limit=300)

        # Core state
        self.generation_index = 0
        self.agents = [Agent(id=i, token_registry=self.token_registry) for i in range(POP_SIZE)]
        for agent in self.agents:
            agent.community_lexicon = self.community_lexicon
        self.last_utterances = {}   # agent_id -> utterance
        self.challenge = ChallengeSystem()
        self.action_queue = []

        self.next_task_id = 1
        self.active_tasks = []
        self.referential_memory = []
        self.numeric_memory = []
        self.action_memory = []
        self.human_token_memory = Counter()
        # Reset and filled by run_dialogues each generation.  Keeping this
        # separate from the long dialogue archive makes the current social
        # language pressure visible in the CSV/report.
        self.dialogue_metrics = {}
        # A human-facing transcript lives outside an individual run so it can
        # remain available while a watch-mode simulation continues.  The
        # bridge is inert until a completed ``Ryan: ...`` line appears.
        self.conversation_path = Path(conversation_path or "converse.txt")
        self.community_conversation = CommunityConversation(self.conversation_path)

        self.community_semantic = {
            "vecs": {},          # token -> centroid vector
            "counts": {},        # token -> number of contributors
            "confidence": {},  # token -> 0..1 confidence
            "last_update_gen": 0
        }

        # Build sandbox world + per-agent private FS and APIs
        spec = SandboxSpec(
            root=str(self.run_dir / "sandbox"),
            world_w=64,
            world_h=64,
            seed=self.random_seed % (2 ** 32),
        )
        world, homes, apis, bus, ledger = build_sandbox(self.agents, spec)

        self.world = world
        self.homes = homes
        self.agent_apis = apis
        self.bus = bus
        self.ledger = ledger

        # Attach API, energy, and apply initial semantic seeds to all agents
        for a in self.agents:
            if a.id in self.agent_apis:
                a.attach_api(self.agent_apis[a.id])
            a.energy = ENERGY_MAX

            # Apply global semantic seeds once at initialisation
            try:
                if hasattr(a, "semantic_system") and self.semantic_seeds:
                    a.semantic_system.receive_semantic_seeds(self.semantic_seeds)
            except Exception:
                pass

        # identity grounding for all agents
        for a in self.agents:
            tok = f"A{a.id}"
            a.vocab.add(tok)               # ensure language sees it
            if hasattr(a, "identity_system"):
                a.identity_system.mark_identity_token(tok)
            else:
                a.mark_identity_token(tok)     # fallback bridge

        # Homeostatic control state
        self.archetype_stats = {
            "explorer": 0.0,
            "cooperator": 0.0,
            "habit": 0.0,   # habit learner
            "loner": 0.0,
        }

        # Task weights (will be renormalised by homeostasis)
        self.task_weights = {
            "compare_numbers": 1.0,
            "reconcile_counts": 1.0,
            "agreement_dialogue": 1.0,
        }

        # thresholds for “too low” / “too high”
        self.homeostasis_cfg = {
            "low_frac": 0.10,   # below this → boost
            "high_frac": 0.80,  # above this → damp
            "min_frac_coop": 0.25,
            "max_frac_coop": 0.75,
        }

        self.task_scorers = {
            # already / obvious
            "compare_numbers": self.score_compare_numbers,
            "reconcile_counts": self.score_reconcile_counts,
            "pref_align": self.score_pref_align,

            # coordination pressure
            "agreement_dialogue": self.score_agreement_dialogue,
            "similarity_debate": self.score_similarity_debate,
            "definition_swap": self.score_definition_swap,
            "misunderstanding_detection": self.score_misunderstanding_detection,
        }

        self.completed_tasks = []

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _config_snapshot(module):
        """Return serialisable public constants from a configuration module."""
        return {
            name: value
            for name, value in vars(module).items()
            if name.isupper() and isinstance(value, (str, int, float, bool, type(None)))
        }

    def _create_run_dir(self, requested_dir):
        if requested_dir is not None:
            destination = Path(requested_dir).expanduser().resolve()
            try:
                destination.mkdir(parents=True, exist_ok=False)
            except FileExistsError as exc:
                raise ValueError(
                    f"Run directory already exists: {destination}. "
                    "Choose a new directory to avoid mixing experiment outputs."
                ) from exc
            return destination

        runs_root = Path("runs").resolve()
        runs_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        prefix = f"{timestamp}_seed-{self.random_seed:016x}"

        for suffix in range(1000):
            name = prefix if suffix == 0 else f"{prefix}_{suffix}"
            destination = runs_root / name
            try:
                destination.mkdir()
                return destination
            except FileExistsError:
                continue

        raise RuntimeError("Could not allocate a unique directory for this run.")

    def _write_run_metadata(self):
        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": self.random_seed,
            "run_directory": str(self.run_dir),
            "artifacts": {
                "generation_report": str(self.report_path),
                "cultural_log": str(self.csv_path),
                "dialogue_log": str(self.dialogue_log_path),
                "sandbox": str(self.run_dir / "sandbox"),
            },
            "config": self._config_snapshot(project_config),
            "coordinator_settings": self._config_snapshot(coordinator_settings),
        }
        with (self.run_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)
            file.write("\n")

    def compact_agent_vocabularies(self, max_private_tokens=80):
        """Retain public and recently useful language, not every old token."""
        public = set(self.community_lexicon.numeric_conventions().values())
        public.update(self.community_lexicon.referential_conventions().values())
        public.update(self.community_lexicon.action_conventions().values())
        public.update(getattr(self, "human_token_memory", {}).keys())

        for agent in self.agents:
            protected = set(public)
            protected.update(getattr(agent, "symbol_map", {}).values())
            protected.update(getattr(agent, "referent_lexicon", {}).values())
            protected.update(getattr(agent, "action_lexicon", {}).values())
            protected.update(getattr(agent, "recent_tokens", [])[-40:])
            protected.update(getattr(agent, "reasoning_tokens", {}).values())

            preferences = getattr(agent, "utter_bias", {}).get("symbol_preferences", {})
            candidates = [
                token for token in getattr(agent, "vocab", set())
                if isinstance(token, str) and token.strip() and token not in protected
            ]
            candidates.sort(key=lambda token: (-float(preferences.get(token, 0.0)), token))
            agent.vocab = {
                token for token in protected
                if isinstance(token, str) and token.strip()
            }
            agent.vocab.update(candidates[:max_private_tokens])

    def enqueue_action(self, agent, action):
        self.action_queue.append((agent, action))

    def build_semantic_seeds(self):
        """
        Build a compact semantic seed structure from dictionary.json:
        - word list
        - synonym pairs
        - antonym pairs
        """

        try:
            raw = self.world.read_json("/dictionary.json")
            if not raw or not isinstance(raw, dict):
                return {
                    "words": [],
                    "synonyms": [],
                    "antonyms": []
                }

            words = []
            synonyms = []
            antonyms = []

            for word, entry in raw.items():
                w = word.lower()

                if 2 <= len(w) <= 14 and w.isalpha():
                    words.append(w)

                # SYNONYMS
                for syn in entry.get("SYNONYMS", []):
                    syn = syn.lower()
                    if syn.isalpha() and len(syn) > 1:
                        synonyms.append((w, syn))

                # ANTONYMS
                for ant in entry.get("ANTONYMS", []):
                    ant = ant.lower()
                    if ant.isalpha() and len(ant) > 1:
                        antonyms.append((w, ant))

            # Deduplicate & limit size
            random.shuffle(words)
            words = words[:400]

            return {
                "words": words,
                "synonyms": synonyms,
                "antonyms": antonyms
            }

        except Exception:
            return {
                "words": [],
                "synonyms": [],
                "antonyms": []
            }
        print ("Semantic seeds built:" , len(words), "words;", len(synonyms), "synonyms;", len(antonyms), "antonyms")

    def extract_dictionary_words(self, limit=250):
        if self.cached_dictionary_words is not None:
            return self.cached_dictionary_words[:limit]

        try:
            raw = self.world.read_json("/dictionary.json")
            if not raw or not isinstance(raw, dict):
                self.cached_dictionary_words = []
                return []

            words = []

            for key, entry in raw.items():
                key = key.lower()
                if 2 <= len(key) <= 16:
                    words.append(key)

                for syn in entry.get("SYNONYMS", []):
                    syn = re.sub(r"[^a-z0-9]", "", syn.lower())
                    if 2 <= len(syn) <= 16:
                        words.append(syn)

            random.shuffle(words)
            self.cached_dictionary_words = words
            return words[:limit]

        except Exception:
            self.cached_dictionary_words = []
            return []

    def _clamp(self, v, lo=0.0, hi=1.0):
        return max(lo, min(hi, v))

    def agent_by_id(self, agent_id):
        for ag in self.agents:
            if ag.id == agent_id:
                return ag
        return None

    def sample_random_utterance(self):
        if not hasattr(self, "dialogue_log") or not self.dialogue_log:
            return None
        entry = random.choice(self.dialogue_log)
        if not entry:
            return None
        turns = entry.get("turns") or []
        if not turns:
            return None
        turn = random.choice(turns)
        return turn.get("utterance")

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

    # ---------- cultural inheritance helpers ----------
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

    def _blend_symbol_drift(self, *parents, noise=0.02, decay=0.98):
        # assume all parents have same vocab keys
        if not parents:
            return {}
        keys = list(parents[0].symbol_drift.keys())
        out = {}
        for k in keys:
            vals = [p.symbol_drift.get(k, 0.0) for p in parents]
            mean = sum(vals) / max(1, len(vals))
            mean = mean * decay + random.uniform(-noise, noise)
            out[k] = self._clamp(mean, -1.0, 1.0)
        return out


    # -----------------------------
    # Fitness pipeline
    # -----------------------------
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

    # -----------------------------
    # Parent selection (tri-parent)
    # -----------------------------
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

    # -----------------------------
    # Generation loop
    # -----------------------------
    def run_generation(self):

        # -------------------------------------------------------
        # 0. SET GENERATION + PREPARE AGENT CONTEXT (CRITICAL)
        # -------------------------------------------------------
        self.generation_index += 1

        for a in self.agents:
            a.current_generation = self.generation_index
            a.population = self.agents

            if not hasattr(a, "tasks_attempted_this_gen"):
                a.tasks_attempted_this_gen = set()
            if not hasattr(a, "last_task_gen"):
                a.last_task_gen = -1

            # keep numeric map consistent
            if hasattr(a, "counting") and hasattr(a, "symbol_map") and a.symbol_map:
                try:
                    a.counting.set_symbol_map(a.symbol_map)
                except Exception:
                    pass

            # reset per-generation semantic engagement marks
            if hasattr(a, "touched_semantic_tokens"):
                a.touched_semantic_tokens.clear()
            else:
                a.touched_semantic_tokens = set()

        # -------------------------------------------------------
        # CONTINUE NORMAL GENERATION
        # -------------------------------------------------------
        # process queued actions
        for agent, action in self.action_queue:
            orchestrate_action(agent, action, self)
        self.action_queue.clear()

        # challenge
        self.challenge.new_challenge()

        for agent in self.agents:
            agent.challenge_guess = self.challenge.expectation_to_guess(
                agent.get_overall_expectation()
            )
            agent.last_utterance = agent.numeric_system.speak_number(agent.challenge_guess)

        self.ledger.reset_gen()

        # language phase
        self.run_language_phase()

        # -------------------------------------------------------
        # ARCHETYPE CLASSIFICATION (needed for homeostasis)
        # -------------------------------------------------------
        self.assign_archetypes(self.agents, self.classify_agent_archetype)

        # -------------------------------------------------------
        # TASK GENERATION (homeostatic mix)
        # -------------------------------------------------------

        # Build a mixed batch of tasks for this generation
        # For example: 4 tasks per generation
        task_fitness_baseline = {a.id: a.own_fitness for a in self.agents}
        self.active_tasks = self._build_task_batch(
            num_tasks = 4,
            gen = self.generation_index
        )

        # Agents attempt tasks
        for ag in self.agents:
            if hasattr(ag, "try_solve_tasks"):
                ag.try_solve_tasks(self.active_tasks, self.generation_index)

        # -------------------------------------------------------
        # POST-TASK EVALUATION
        # -------------------------------------------------------

        for task in self.active_tasks:
            self.score_task(task)

        self.compact_numeric_overlays()

        # Preserve task rewards across the later behavioural-fitness reset.
        for a in self.agents:
            delta = a.own_fitness - task_fitness_baseline[a.id]
            a.task_fitness = max(-2.0, min(5.0, delta))

        # then archive / trim
        self.completed_tasks.extend(self.active_tasks)
        # Keep the whole current-generation batch available to telemetry and
        # reports.  The next generation replaces ``active_tasks`` outright;
        # only the historical archive needs a cap.
        self.completed_tasks = self.completed_tasks[-500:]

        # motivated action
        for agent in self.agents:
            action = agent.decision_system.decide_action()
            success = orchestrate_action(agent, action, self)
            agent.last_action = action
            agent.last_action_success = success

        # sandbox
        for a in self.agents:
            if a.id in self.agent_apis:
                a.attach_api(self.agent_apis[a.id])
            a.run_sandbox_step()

        self.reward_penalty_phase()

        # challenge reward (phase 2)
        for agent in self.agents:
            agent.own_fitness += self.challenge.evaluate_guess(
                agent, self.generation_index
            )

        # recovery dynamics
        for a in self.agents:
            # A modest baseline recovery lets long-lived agents accumulate
            # shared language and numeric conventions.
            a.energy = min(100.0, a.energy + 1.0)
            peer = random.choice(self.agents)
            if peer.id != a.id:
                a.adjust_trust(peer.id, +0.01, channel=1)

            if getattr(a, "teaching_attempted", False):
                c = getattr(a, "curiosity", 0.5)
                a.energy = min(100.0, a.energy + 0.05 * c)
                a.own_fitness += 0.02 * c
                a.teaching_attempted = False

        # debug snapshot
        if self.agents:
            sm = random.choice(self.agents)
            print(f"[Gen {self.generation_index}] A{sm.id}: energy={sm.energy:.2f}")

        for agent in self.agents:
            # Per-generation semantic drift (gravity + flavour attractors)
            if hasattr(agent, "semantic_update_tick"):
                agent.semantic_update_tick()

        # -------------------------------------------------------
        # DIALOGUE & FEEDBACK PHASE
        # -------------------------------------------------------
        self.run_dialogues()

        # =======================================================
        # PHASE 1 FITNESS (program behaviour)
        # =======================================================
        self.evaluate_agents()

        # evaluate_agents() recomputes own_fitness, so carry task success
        # forward as an independent selection signal.
        for a in self.agents:
            a.own_fitness += getattr(a, "task_fitness", 0.0)

        # ✅ ADD OPERATOR-BASED REWARD (selection pressure)
        for a in self.agents:
            ev = getattr(a, "last_relation_event", None)
            if not ev:
                continue

            if not hasattr(a, "reasoning_tokens"):
                continue

            truth = ev.get("truth")
            used_ops = ev.get("used_ops", []) or []

            correct_tok = a.reasoning_tokens.get(truth)
            if not correct_tok:
                continue

            reward = 0.0
            if correct_tok in used_ops:
                # small positive bump for matching operator
                reward += 0.5

            # optionally penalise clearly wrong operators
            wrong = [t for t in used_ops if t != correct_tok]
            if wrong:
                reward -= 0.2 * len(wrong)

            a.own_fitness += reward

        # Add challenge score (phase-2 contribution) again for behaviour alignment
        for a in self.agents:
            a.own_fitness += self.challenge.evaluate_guess(a)

        # PHASE D: semantic alignment bonuses (small, safe)
        self.evaluate_semantic_alignment_tasks()

        self.apply_cooperation()
        self.apply_novelty()
        self.compute_total_fitness()

        # participation
        participants = set()
        for a in self.agents:
            st = self.ledger.stats.get(a.id, {"actions": 0})
            if st["actions"] == 0:
                a.energy = max(0.0, a.energy - IDLE_TAX)
            if st["actions"] >= MIN_PARTICIPATION:
                participants.add(a.id)

        spoke = set(self.last_utterances.keys())
        final_participants = participants.union(spoke)

        # === NUMERIC DISTINCTION REWARD ===
        for agent in self.agents:
            mapping = agent.numeric_system.numeric_semantic
            collisions = agent.numeric_system._numeric_collision_score()

            # reward clarity
            if collisions == 0:
                agent.total_fitness += 0.05
                agent.esteem = min(1.0, agent.esteem + 0.01)

            # penalise collapsed number-lines
            else:
                agent.total_fitness -= 0.02 * collisions
                agent.energy = max(0.0, agent.energy - 0.05 * collisions)

        self.language_feedback()

        # motivational decay
        for a in self.agents:
            a.update_needs()
            a.decay_states()
            a.population = self.agents
            # --- rel-family drift ---
            rel_fam = a.semantic.get("rel_family", {})
            if rel_fam:
                rel_centroid = a.semantic_system._centroid([i["vec"] for i in rel_fam.values()])
                if rel_centroid is not None:
                    for tok, info in rel_fam.items():
                        info["age"] += 1
                        info["vec"] = [
                            v + 0.01 * (c - v)
                            for v, c in zip(info["vec"], rel_centroid)
                        ]
                        a.semantic["vecs"][tok] = info["vec"]

            # --- NEW: ref-family drift ---
            ref_fam = a.semantic.get("ref_family", {})
            if ref_fam:
                ref_centroid = a.semantic_system._centroid([i["vec"] for i in ref_fam.values()])
                if ref_centroid is not None:
                    # slightly weaker pull than rel, to keep “about” tokens a bit looser
                    for tok, info in ref_fam.items():
                        info["age"] += 1
                        info["vec"] = [
                            v + 0.007 * (c - v)
                            for v, c in zip(info["vec"], ref_centroid)
                        ]
                        a.semantic["vecs"][tok] = info["vec"]
        
        # === NEW: update community semantic map (Mode C) ===
        self.update_community_semantic()

        # -------------------------------------------------------
        # CULL & REPRODUCE  (Gentle Evolution Mode)
        # -------------------------------------------------------

        survivors = [a for a in self.agents if a.energy > 0]
        dead = [a for a in self.agents if a.energy <= 0]

        # -------------------------------------------------------
        # 1. Performance Cull (GENTLE MODE)
        # -------------------------------------------------------

        # Kill **exactly 1** worst performer per generation.
        # (Unless energy-deaths already removed some.)
        survivors_sorted = sorted(survivors, key=lambda a: a.total_fitness)

        num_cull = 1
        culled_for_fitness = survivors_sorted[:num_cull]

        survivors = [a for a in survivors if a not in culled_for_fitness]

        culled_ids = [a.id for a in culled_for_fitness]

        # -------------------------------------------------------
        # 2. Reproduce until population full
        # -------------------------------------------------------
        needed = POP_SIZE - len(survivors)
        children = []

        # reproduction pool = top 40%
        BREED_RATIO = 0.40
        repro_pool = sorted(survivors, key=lambda a: a.total_fitness, reverse=True)
        repro_pool = repro_pool[:max(3, int(len(repro_pool) * BREED_RATIO))]

        while len(children) < needed:
            p1, p2, p3 = self.select_parents(repro_pool)

            # reuse culled ID if available
            if culled_ids:
                new_id = culled_ids.pop(0)
            else:
                new_id = len(survivors) + len(children)

            child = self.make_child(p1, p2, p3, new_id=new_id)
            children.append(child)

        self.agents = survivors + children
        self.compact_agent_vocabularies()

        # ✅ DO *NOT* FLUSH HELP FILES HERE ANY MORE
        # (they’re read via offsets; flushing would kill late readers)

        if ENABLE_DICTIONARY_INJECTION:
            # dictionary injection
            dict_words = self.extract_dictionary_words(300)
            for a in self.agents:
                a.populate_dictionary(dict_words)
                a.vocab.update(dict_words)
                a.dict_vocab.update(dict_words)

        # reattach APIs
        for a in children:
            api = self.agent_apis.get(a.id)
            if api:
                api.agent = a     # rebind agent
                a.attach_api(api)
            else:
                home = self.homes[a.id]
                api = AgentAPI(a, self.world, home, self.bus, self.ledger)
                self.agent_apis[a.id] = api
                a.attach_api(api)

        # memory decay
        self.apply_memory_decay()

        try:
            best = max(self.agents, key=lambda x: x.total_fitness)
            for a in self.agents:
                if a.social_memory:
                    a.lineage_score = 0.95 * a.lineage_score + \
                                      0.05 * (best.total_fitness * OUTCOME_SCALE)
        except Exception:
            pass

        for a in self.agents:
            a.decay_social_memory(
                decay_rate=a.memory_decay_rate,
                gen_index=self.generation_index
            )

        # --- TRUST HOMEOSTASIS ---
        for a in self.agents:
            tt = float(a.traits.get("trust_threshold", 0.5))

            # Pull toward a *trust-friendly* set-point (~0.35)
            tt += (0.35 - tt) * 0.08

            # Clamp to a narrower, cooperative range
            tt = max(0.10, min(0.80, tt))
            a.traits["trust_threshold"] = tt

        # Agents with extremely sparse interactions:
        # gently decay their social impressions toward neutral.
        for a in self.agents:
            # count total recorded interactions, not just distinct partners
            total_interactions = sum(
                rec.get("interactions", 0)
                for rec in getattr(a, "social_memory", {}).values()
            )
            if total_interactions < 3:  # mostly isolated agents
                for rec in getattr(a, "social_memory", {}).values():
                    rec["mean_outcome"] *= 0.95
                    rec["trust_delta"]   *= 0.95
                    rec["offspring_success"] *= 0.95

        # A human prompt is deliberately allowed to sit for a couple of
        # generations before the community answers.  This gives agents time
        # to keep practising and makes the external interaction observable in
        # the same per-generation artifacts as everything else.
        try:
            self.community_conversation.poll(self)
        except Exception as error:
            print(f"[CONVERSATION ERROR] {error}")

        txt_file, csv_file = _get_log_filenames(self)
        summary = compute_generation_summary(self)

        append_generation_to_csv(summary, csv_file)
        write_generation_report(self, txt_file, self.generation_index)

        self.cached_dictionary_words = None

        # family updates and semantic drift
        for a in self.agents:
            # every 5 generations
            if self.generation_index % 5 == 0:
                if hasattr(a, "semantic_system") and hasattr(a.semantic_system, "family_system"):
                    a.semantic_system.family_system.detect_semantic_families()
                    a.semantic_system.family_system.prune_families()
                else:
                    if hasattr(a, "detect_semantic_families"):
                        a.detect_semantic_families()
                    a.prune_families()

            if self.generation_index % 50 == 0:
                a.debug_dump_semantics()

            # each generation
            a.semantic_drift_update()
            a.semantic_system._sanitize_vector_dims()
            if hasattr(a, "semantic_system") and hasattr(a.semantic_system, "family_system"):
                a.semantic_system.family_system.family_reinforcement_update()
                a.semantic_system.family_system.family_soft_decay()
            else:
                if hasattr(a, "family_reinforcement_update"):
                    a.family_reinforcement_update()
                a.family_soft_decay()
            # --- Update community semantic centroid ---

        # --- Community Semantics ---
        self.print_community_semantic_stats()

        for a in self.agents:
            a.epistemic_system.detect_semantic_gaps(self.community_semantic)
            a.epistemic_system.apply_flavour_homeostasis(self.community_semantic)

        self.print_semantic_gap_stats()
        self.semantic_alignment_tasks = []

        if hasattr(self, "summarize_dialogues"):
            print(self.summarize_dialogues(last_n=50))

        print(f"Generation {self.generation_index} running...")

    # =======================================================
    # COMMUNITY SEMANTIC MAP
    # ======================================================
    def print_community_semantic_stats(self):
        size = len(self.community_semantic["vecs"])
        print(f"[Community Semantic] Tokens={size}")

    def print_semantic_gap_stats(self, top_k=5):
        total_misaligned = 0
        total_orphans = 0
        samples = []

        for a in self.agents:
            gaps = getattr(a, "semantic_gaps", None)
            if not gaps:
                continue

            mis = gaps.get("misaligned", [])
            orp = gaps.get("orphans", [])

            total_misaligned += len(mis)
            total_orphans += len(orp)

            if mis:
                samples.append((a.id, mis[0]["token"], mis[0]["dist"]))

        print(f"[Semantic Gaps] misaligned={total_misaligned} orphans={total_orphans}")
        if samples:
            samples = sorted(samples, key=lambda x: x[2], reverse=True)[:top_k]
            for aid, tok, dist in samples:
                print(f"  A{aid}: '{tok}' dist={dist:.3f}")

    def community_distance(self, tok, community_map):
        if tok not in community_map["vecs"]:
            return None
        if tok not in self.semantic["vecs"]:
            return None

        a = np.array(self.semantic["vecs"][tok])
        b = np.array(community_map["vecs"][tok])
        return float(np.linalg.norm(a - b))

    # =====================================================
    # Receive gap-driven tasks from agents
    # =====================================================
    def add_semantic_alignment_task(self, task):
        """
        task: {
            "task_id": ...,
            "task_type": "semantic_alignment",
            "token": "...",
            "proposer": agent_id,
            "vector_local": [...],      # proposer's vector
            "community_vector": [...]    # community vector
        }
        """
        self.semantic_alignment_tasks.append(task)

    # =====================================================
    # PHASE D: Fitness shaping for semantic alignment
    # =====================================================
    def evaluate_semantic_alignment_tasks(self):
        """
        Reward agents who:
          - engaged with semantic-alignment tasks, and
          - whose local token semantics are well-aligned with the
            community centroid and the proposer.

        We:
          - reward alignment, not conformity (no hard penalties for
            disagreement, just less bonus)
          - reward proposers if others align with the token they flagged
        """
        if not hasattr(self, "semantic_alignment_tasks"):
            return
        if not getattr(self, "community_semantic", None):
            return

        comm_vecs = self.community_semantic.get("vecs", {})
        proposer_credit = defaultdict(float)

        for task in self.semantic_alignment_tasks:
            tok = task.get("token")
            proposer_id = task.get("proposer")
            if tok is None or proposer_id is None:
                continue

            # community view for this token (may be missing)
            comm_vec = comm_vecs.get(tok)

            # current proposer view (live, not frozen)
            proposer = None
            for ag in self.agents:
                if ag.id == proposer_id:
                    proposer = ag
                    break

            prop_vec = None
            if proposer is not None:
                prop_vec = proposer.semantic["vecs"].get(tok, None)

            # evaluate each agent who actually worked on this token this gen
            for ag in self.agents:
                touched = getattr(ag, "touched_semantic_tokens", set())
                if tok not in touched:
                    continue

                local_vec = ag.semantic["vecs"].get(tok)
                if local_vec is None:
                    continue

                sim_comm = self._cosine(local_vec, comm_vec) if comm_vec is not None else 0.0
                sim_prop = self._cosine(local_vec, prop_vec) if prop_vec is not None else 0.0

                # blend: lean slightly toward community, but still respect proposer
                score = 0.6 * sim_comm + 0.4 * sim_prop

                # exploration-safe: we never *punish* disagreement here,
                # we just give less or no bonus for low/negative scores.
                if score <= 0.0:
                    continue

                # soft cap on contribution per agent per token
                # score is in (0,1]; we scale into ~[0, 1.5]
                raw_reward = min(1.5, score * 1.5)

                # apply to own_fitness and energy gently
                ag.own_fitness += raw_reward
                ag.energy = min(100.0, ag.energy + 0.1 * raw_reward)

                # track semantic alignment success for later analysis
                ag.memory.setdefault("semantic_alignment_success", 0.0)
                ag.memory["semantic_alignment_success"] += raw_reward

                # build proposer credit (if not self-proposed)
                if proposer is not None and proposer.id != ag.id:
                    proposer_credit[proposer.id] += 0.5 * raw_reward

                    # tiny social-memory record to feed into parent selection
                    try:
                        outcome = raw_reward * OUTCOME_SCALE * 200.0
                        ag.remember_interaction(
                            proposer.id,
                            outcome=outcome,
                            gen_index=self.generation_index,
                        )
                        proposer.remember_interaction(
                            ag.id,
                            outcome=outcome * 0.6,
                            gen_index=self.generation_index,
                        )
                    except Exception:
                        pass

        # Apply proposer bonuses (reward "good questions")
        for pid, val in proposer_credit.items():
            for ag in self.agents:
                if ag.id == pid:
                    # proposers benefit, but at lower gain
                    bonus = min(1.0, val)
                    ag.own_fitness += bonus
                    ag.energy = min(100.0, ag.energy + 0.05 * bonus)
                    ag.memory.setdefault("semantic_alignment_proposals", 0.0)
                    ag.memory["semantic_alignment_proposals"] += bonus
                    break

    # =======================================================
    # COMMUNITY SEMANTIC MAP (MIXED STRENGTH, MODE C)
    # ======================================================
    def _compute_token_stats(self, token):
        """
        For a given token, gather all agent vectors and compute:
          - mean vector
          - mean distance to the mean
          - coverage (fraction of agents that know it)
          - mean local usage_count across agents that know it
        """
        import numpy as np

        vecs = []
        usage_vals = []
        num_agents = max(1, len(self.agents))

        for ag in self.agents:
            sem = getattr(ag, "semantic", None)
            if not sem:
                continue

            v = sem.get("vecs", {}).get(token)
            if v is None:
                continue

            vecs.append(np.array(v, dtype=float))

            meta = sem.get("tokens", {}).get(token, {})
            usage_vals.append(float(meta.get("usage_count", 0.0)))

        if not vecs:
            return None, 0.0, 0.0, 0.0

        arr = np.stack(vecs, axis=0)
        mean_vec = arr.mean(axis=0)

        dists = np.linalg.norm(arr - mean_vec, axis=1)
        mean_dist = float(dists.mean()) if len(dists) > 0 else 0.0

        coverage = len(vecs) / num_agents
        mean_usage = float(sum(usage_vals) / max(1, len(usage_vals)))

        return mean_vec, mean_dist, coverage, mean_usage

    def _community_token_confidence(self, mean_dist, coverage):
        """
        Confidence in [0,1]:
          - higher when many agents agree (coverage high)
          - higher when they are close together (mean_dist small)
        """
        # distance factor: ~1 when dist=0, decays with distance
        dist_factor = 1.0 / (1.0 + 0.5 * mean_dist)
        dist_factor = max(0.0, min(1.0, dist_factor))

        cov_factor = max(0.0, min(1.0, coverage))

        # combine, slightly emphasise agreement
        conf = (0.6 * cov_factor + 0.4 * dist_factor)
        return max(0.0, min(1.0, conf))

    def update_community_semantic(
        self,
        max_tokens=1024,
        min_coverage=0.15,
        max_mean_dist=2.0,
        min_usage=5,
        prune_min_age=10,
        prune_conf=0.25,
        prune_target_size=512,
    ):
        """
        Mixed-strength EM update (bounded Mode C):

          • Only consider tokens that:
              - are used enough locally (min_usage)
              - appear in enough agents (min_coverage)
              - are not wildly inconsistent (mean_dist <= max_mean_dist)

          • High-confidence tokens update faster (strong anchor),
            low-confidence ones drift slowly.

          • Old, persistently low-confidence tokens are pruned.

          • Global vocab size is capped at prune_target_size by
            dropping the lowest-confidence, least-updated tokens.
        """

        if not self.agents:
            return

        com = self.community_semantic
        c_vecs = com["vecs"]
        c_counts = com["counts"]
        c_conf = com["confidence"]

        # ----------------------------------------------
        # 1) Collect candidate tokens with usage gating
        # ----------------------------------------------
        token_set = set()

        for ag in self.agents:
            sem = getattr(ag, "semantic", None)
            if not sem:
                continue

            tokens_meta = sem.get("tokens", {})
            for tok, meta in tokens_meta.items():
                # Skip obviously identity-like tokens (a23, A7, etc.)
                if isinstance(tok, str) and tok.lower().startswith("a") and tok[1:].isdigit():
                    continue

                if meta.get("usage_count", 0) >= min_usage:
                    token_set.add(tok)

        if not token_set:
            return

        tokens = list(token_set)
        random.shuffle(tokens)
        tokens = tokens[:max_tokens]

        # ----------------------------------------------
        # 2) EM-style centroid update with confidence
        # ----------------------------------------------
        for tok in tokens:
            stats = self._compute_token_stats(tok)
            if stats is None:
                continue
            mean_vec, mean_dist, coverage, mean_usage = stats

            if mean_vec is None:
                continue

            # Hard gates: only “community-worthy” tokens enter
            if coverage < min_coverage:
                continue
            if mean_dist > max_mean_dist:
                continue

            conf = self._community_token_confidence(mean_dist, coverage)

            # Mixed-strength learning rate:
            #   base for all tokens, extra boost for high-confidence ones
            alpha_low = 0.03   # everyone at least gets this
            alpha_high = 0.20  # upper cap for very high confidence
            alpha = alpha_low + (alpha_high - alpha_low) * (conf ** 2)
            alpha = max(alpha_low, min(alpha_high, alpha))

            old = np.array(c_vecs.get(tok, mean_vec), dtype=float)
            new = (1.0 - alpha) * old + alpha * mean_vec

            c_vecs[tok] = new.tolist()
            c_counts[tok] = c_counts.get(tok, 0) + 1
            c_conf[tok] = conf

        # ----------------------------------------------
        # 3) Prune old, low-confidence tokens
        # ----------------------------------------------
        to_delete = []

        for tok, old_conf in list(c_conf.items()):
            age = c_counts.get(tok, 0)

            # Only prune tokens that have had time to stabilise
            if age >= prune_min_age and old_conf < prune_conf:
                to_delete.append(tok)

        for tok in to_delete:
            c_vecs.pop(tok, None)
            c_counts.pop(tok, None)
            c_conf.pop(tok, None)

        # ----------------------------------------------
        # 4) Enforce global vocab cap
        # ----------------------------------------------
        if len(c_vecs) > prune_target_size:
            # Sort by (confidence asc, age asc) → drop worst first
            sorted_tokens = sorted(
                list(c_vecs.keys()),
                key=lambda t: (c_conf.get(t, 0.0), c_counts.get(t, 0))
            )
            excess = len(c_vecs) - prune_target_size
            for tok in sorted_tokens[:excess]:
                c_vecs.pop(tok, None)
                c_counts.pop(tok, None)
                c_conf.pop(tok, None)

        com["last_update_gen"] = self.generation_index
