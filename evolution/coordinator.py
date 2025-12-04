# file: evolution/coordinator.py
import math
import random
import re
from collections import defaultdict
import numpy as np

from agents.agent import Agent
from evolution.challenge import ChallengeSystem
from evolution.counting import CountingSystem
from evolution.logging import compute_generation_summary, append_generation_to_csv, write_generation_report, _get_log_filenames
from evolution.programs import run_program, safe
from evolution.behaviours.orchestration import orchestrate_action


# If your phase3/__init__.py re-exports these, this import works:
from phase3 import SandboxSpec, build_sandbox
from phase3.agent_api import AgentAPI

import time

def PERF(msg):
    print(f"[PERF {time.time():.3f}] {msg}", flush=True)


# ===========================
# Tunables / Knobs
# ===========================
POP_SIZE               = 30
ELITE_RATIO            = 0.20

ENABLE_DICTIONARY_INJECTION = False

# --- Language
UTTER_CHANCE           = 1.0
UTTER_EFFECT           = 0.15
REWARD_EPS             = 1e-9
REWARD_TEMP            = 1.0
PHASE2_LR              = 0.10
TASK_SEMANTIC_ALIGNMENT = "semantic_alignment"

# --- Social / cultural
MATE_POOL_SIZE         = 12
COMPAT_WEIGHT          = 0.5
FITNESS_WEIGHT         = 1.0
MEMORY_WEIGHT          = 0.8
DIVERSITY_WEIGHT       = 0.2

GOSSIP_PAIRS_PER_GEN   = 20
TEACH_PROB             = 0.15
TEACH_TOP_FRACTION     = 0.20
TEACH_PAIRS_PER_PASS   = 16     # ~half pop attempts each pass
TEACH_ROUNDS_PER_GEN   = 2      # we will call semantic_teaching_phase() twice per gen
TEACH_ACC_TEMP         = 1.0    # keep reward linear in accuracy
IMPROVE_ONLY_TEACH     = True
OUTCOME_SCALE          = 0.001
REFERENTIAL_BONUS = 0.35  # 0..1 (try 0.2–0.5)

# --- Safety caps
MAX_COOP_BONUS         = 1000.0
MAX_NOVELTY_BONUS      = 1000.0
MAX_FIT                = 1e6

# --- Phase-3 energy / participation
ENERGY_MAX             = 100.0
IDLE_TAX               = 1.0
MIN_PARTICIPATION      = 1


class Coordinator:
    def __init__(self):
        self.active_tasks = self.load_tasks()
        self.semantic_alignment_tasks = []

        self.cached_dictionary_words = None
        self.semantic_seeds = {"words": [], "synonyms": [], "antonyms": []}
        self.semantic_seeds = self.build_semantic_seeds()
        self.cached_dictionary_words = self.extract_dictionary_words(limit=300)

        # Core state
        self.generation_index = 0
        self.agents = [Agent(i) for i in range(POP_SIZE)]
        self.last_utterances = {}   # agent_id -> utterance
        self.challenge = ChallengeSystem()
        self.action_queue = []

        self.next_task_id = 1
        self.active_tasks = []

        self.community_semantic = {
            "vecs": {},          # token -> centroid vector
            "counts": {},        # token -> number of contributors
            "confidence": {},  # token -> 0..1 confidence
            "last_update_gen": 0
        }

        # Build sandbox world + per-agent private FS and APIs
        spec = SandboxSpec(root="sandbox_root", world_w=64, world_h=64, seed=42)
        world, homes, apis, bus, ledger = build_sandbox(self.agents, spec)

        self.world = world
        self.homes = homes
        self.agent_apis = apis
        self.bus = bus
        self.ledger = ledger

        # Attach API & energy to all agents
        for a in self.agents:
            if a.id in self.agent_apis:
                a.attach_api(self.agent_apis[a.id])
            a.energy = ENERGY_MAX

        # identity grounding for all agents
        for a in self.agents:
            tok = f"A{a.id}"
            a.vocab.add(tok)               # ensure language sees it
            a.mark_identity_token(tok)     # push semantic meaning into identity subspace

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

    # -----------------------------
    # Helpers
    # -----------------------------
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

    # ---------------------------------------------------------
    #  Archetype stats
    # ---------------------------------------------------------
    def _update_archetype_stats(self):
        """
        Compute current distribution of archetypes over the population.
        Expects each agent to expose `emergent_archetype` with one of:
            "explorer", "cooperator", "habit", "loner"
        Gracefully degrades if missing.
        """
        counts = {
            "explorer": 0,
            "cooperator": 0,
            "habit": 0,
            "loner": 0,
        }

        for ag in self.agents:
            label = getattr(ag, "emergent_archetype", None)
            if label is None:
                # if you use a different attribute name, change this line
                label = getattr(ag, "archetype_label", None)

            if label in counts:
                counts[label] += 1

        total = sum(counts.values())
        if total == 0:
            # No archetype info yet; leave stats unchanged
            return

        self.archetype_stats = {k: counts[k] / total for k in counts}
        
    def classify_agent_archetype(self, ag):
        """
        Classify an agent into one of:
            "explorer", "cooperator", "habit", "loner"

        Based on existing traits:
        - novelty_weight
        - cooperation_weight
        - stability_weight (inverse of exploration)
        - trust_threshold (higher = more isolated)
        """

        # pull traits safely
        nov = float(ag.traits.get("novelty_weight", 0.5))
        coop = float(ag.traits.get("cooperation_weight", 0.5))
        stab = float(ag.traits.get("stability_weight", 0.5))
        trust = float(ag.traits.get("trust_threshold", 0.5))

        # ---- EXPLORER ----
        # seeks novelty, low stability, flexible trust
        if nov > 0.60 and stab < 0.40:
            return "explorer"

        # ---- COOPERATOR ----
        # high cooperation, not too isolated
        if coop > 0.60 and trust < 0.65:
            return "cooperator"

        # ---- HABIT LEARNER ----
        # high stability → pattern seeker / routine builder
        if stab > 0.60:
            return "habit"

        # ---- LONER ----
        # high trust threshold (strict), low cooperation
        if trust > 0.70 and coop < 0.40:
            return "loner"

        # default fallback → classify as mild cooperator
        return "cooperator"

    # ---------------------------------------------------------
    #  Homeostatic task weighting
    # ---------------------------------------------------------
    def _update_task_weights_homeostasis(self, verbose=False):
        """
        Adjust self.task_weights based on archetype distribution.
        Implements a simple negative feedback loop:
          - if an archetype falls below low_frac → boost tasks that reward it
          - if an archetype exceeds high_frac → damp those tasks
        """
        # refresh archetype fractions
        self._update_archetype_stats()
        dist = self.archetype_stats

        cfg = self.homeostasis_cfg
        low = cfg["low_frac"]
        high = cfg["high_frac"]
        min_coop = cfg["min_frac_coop"]
        max_coop = cfg["max_frac_coop"]

        # start from neutral base
        w = {
            "compare_numbers": 1.0,
            "reconcile_counts": 1.0,
            "agreement_dialogue": 1.0,
        }

        # ---- Explorers ↔ compare_numbers (and some dialogue to spread novelty) ----
        frac_exp = dist.get("explorer", 0.0)

        if frac_exp < low:
            # too few explorers → encourage exploration / distinction-making
            w["compare_numbers"] *= 2.0
            w["agreement_dialogue"] *= 1.3  # propagates new relations
        elif frac_exp > high:
            # too many explorers → slow them a bit
            w["compare_numbers"] *= 0.4

        # ---- Habit learners ↔ reconcile_counts (compression / pattern reinforcement) ----
        frac_habit = dist.get("habit", 0.0)

        if frac_habit < low:
            # too few pattern-compressors → push them
            w["reconcile_counts"] *= 2.0
        elif frac_habit > high:
            # too many “rote” minds → ease up
            w["reconcile_counts"] *= 0.4

        # ---- Cooperators ↔ agreement_dialogue (alignment / trust-building) ----
        frac_coop = dist.get("cooperator", 0.0)

        if frac_coop < min_coop:
            w["agreement_dialogue"] *= 2.0
        elif frac_coop > max_coop:
            # society over-synchronised → reduce alignment comfort
            w["agreement_dialogue"] *= 0.6

        # Optionally: give loners a little nudge towards social tasks
        frac_lon = dist.get("loner", 0.0)
        if frac_lon > 0.15:
            # more loners → shift probability away from solo-ish tasks
            w["compare_numbers"] *= 0.8
            w["agreement_dialogue"] *= 1.2

        # ---- normalise ----
        total_w = sum(w.values())
        if total_w <= 0:
            # fallback to uniform
            n = len(w)
            self.task_weights = {k: 1.0 / n for k in w}
        else:
            self.task_weights = {k: v / total_w for k, v in w.items()}

        if verbose:
            print("[HOMEOSTASIS] archetypes:", dist)
            print("[HOMEOSTASIS] task_weights:", self.task_weights)

    # ---------------------------------------------------------
    #  Build tasks for a generation
    # ---------------------------------------------------------
    def _build_task_batch(self, num_tasks: int, gen: int):
        """
        Build a batch of mixed tasks for this generation.
        Homeostasis nudges the mix based on archetype diversity.
        """
        # update task weights based on current archetype ecology
        self._update_task_weights_homeostasis(
            verbose=(gen % 20 == 0)   # e.g. log every 20 gens
        )

        batch = []
        for _ in range(num_tasks):
            ttype = self._sample_task_type()

            if ttype == "compare_numbers":
                task = self.generate_numeric_compare_task()
            elif ttype == "reconcile_counts":
                task = self.generate_reconcile_counts_task()
            elif ttype == "agreement_dialogue":
                task = self.generate_agreement_dialogue_task()
            else:
                # fallback (shouldn't happen)
                task = self.generate_numeric_compare_task()

            batch.append(task)

        return batch

    def assign_archetypes(self, population, classifier_fn):
        """
        classifier_fn(agent) -> one of {"explorer", "cooperator", "habit", "loner"}
        """
        for ag in population:
            ag.emergent_archetype = classifier_fn(ag)

    # ---------------------------------------------------------
    #  Task type sampling
    # ---------------------------------------------------------
    def _sample_task_type(self) -> str:
        """
        Sample a task type according to self.task_weights.
        """
        items = list(self.task_weights.items())
        types, weights = zip(*items)
        r = random.random()
        cum = 0.0
        for t, w in zip(types, weights):
            cum += w
            if r <= cum:
                return t
        return types[-1]  # numerical safety

    # -----------------------------
    # Language & Social
    # -----------------------------
    def gossip_exchange(self):
        """
        Trust-safe scalar gossip:
        - Sender shares a small reputation list (id -> scalar)
        - Receiver:
            (a) nudges trust toward the *sender* (channel=2)
            (b) nudges indirect trust toward named third parties (channel=3)
        """
        agents = self.agents
        if len(agents) < 2:
            return

        teacher, student = random.sample(agents, 2)
        if random.random() < 0.3:  # not every gossip is a teaching moment
            if teacher.semantic["vecs"]:  # avoid empty vocab crash
                word = random.choice(list(teacher.semantic["vecs"].keys()))
                teacher.teach_student(student, word)

        for sender in agents:
            # sender exports compact public view: List[(pid, rep)]
            rep_list = sender.export_reputation(top_k=5) or []
            conf = float(getattr(sender, "reputation_strength", 0.012))

            # pick 1–3 receivers
            num_receivers = random.randint(1, 3)
            receivers = random.sample(agents, min(num_receivers, len(agents)))

            for recv in receivers:
                if recv.id == sender.id:
                    continue

                # (a) trust toward sender for social-info credibility
                recv.adjust_trust(
                    target_id=sender.id,
                    amount=min(conf, 0.02),   # <-- capped
                    channel=2
                )

                # (b) indirect third-party nudges
                for pid, rep in rep_list:
                    if pid == recv.id or pid == sender.id:
                        continue
                    influence = 0.05 * conf * float(rep)
                    if influence == 0.0:
                        continue
                    recv.adjust_trust(
                        target_id=pid,
                        amount=min(influence, 0.015),
                        channel=3
                    )
                
                # (c) semantic-sharing (Option B — low gain)
                if sender.semantic["vecs"]:
                    # 1. choose a word from sender's public vocabulary
                    shared_word = random.choice(list(sender.semantic["vecs"].keys()))

                    # 2. student observes it with low semantic impact
                    recv.observe_utterance(
                        shared_word,
                        gain_scale=0.25   # <--- safe, gentle, will not homogenise vocab
                    )

    def semantic_teaching_phase(self):
        """
        Teacher shares a tiny semantic bundle; student predicts and we
        reward accuracy via cosine similarity. Trust channels updated both ways.
        """
        if not self.agents:
            return

        pairs = min(TEACH_PAIRS_PER_PASS, len(self.agents))
        for _ in range(pairs):
            teacher, student = random.sample(self.agents, 2)

            # Teacher selectivity: willingness threshold dampens spam teaching
            if teacher.teaching_willingness(student.id) < 0.1 and random.random() < 0.85:
                continue

            bundle = teacher.export_semantic_bundle(max_keys=3)
            if not bundle:
                continue

            # Student predicts
            pred = student.attempt_prediction(bundle)
            if pred is None:
                continue

            # Ground truth: mean teacher vector of the same tokens
            toks = bundle.get("tokens", []) or []
            vecs = teacher.semantic["vecs"]
            gt = None
            try:
                acc = None
                if toks:
                    # average known vectors; if missing, skip
                    cols = [vecs[t] for t in toks if t in vecs]
                    if not cols:
                        continue
                    accv = cols[0][:]
                    for v in cols[1:]:
                        accv = add(accv, v)
                    gt = scale(accv, 1.0 / len(cols))
                else:
                    continue
            except Exception:
                continue

            # Accuracy via cosine similarity in [-1,1] → map to [-1,1] reward
            try:
                acc = cos_sim(pred, gt)  # already in [-1,1]
            except Exception:
                acc = 0.0

            if TEACH_ACC_TEMP != 1.0 and acc is not None:
                # optional shaping; we keep linear by default
                acc = math.copysign(abs(acc) ** TEACH_ACC_TEMP, acc)

            reward = max(-1.0, min(1.0, float(acc)))
            if reward > 0:
                k = 0.10 * reward   # increase
            else:
                k = 0.02 * reward   # soften penalty

            # Apply rewards
            teacher.apply_teaching_reward(student.id, reward)
            student.apply_learning_reward(teacher.id, reward)

            # Update trust channels directly (competence/reliability/collaboration)
            # Teacher judged competent & reliable if reward > 0; else slight penalty
            k = 0.06 * reward
            teacher.update_trust_channels(student.id, reward)   # multi-channel small drift
            student.update_trust_channels(teacher.id, reward)

            # Targeted nudges:
            student.adjust_trust(
                teacher.id,
                amount=max(-0.03, min(0.03, k)),
                channel=4
            )
            teacher.adjust_trust(
                student.id,
                amount=max(-0.02, min(0.02, k * 0.5)),
                channel=3
            )

    def language_feedback(self):
        """
        Base fitness → [-1,1] reward (as before), plus a referential bonus:
        If a listener hears an utterance whose tokens overlap the speaker's
        last action tokens, the listener learns more strongly from it.
        """
        tots = [a.total_fitness for a in self.agents]
        mmin, mmax = min(tots), max(tots)
        span = (mmax - mmin) if (mmax - mmin) > REWARD_EPS else 1.0

        base_r = {}
        for a in self.agents:
            x01 = (a.total_fitness - mmin) / span
            if REWARD_TEMP != 1.0:
                x01 = x01 ** REWARD_TEMP
            base_r[a.id] = (x01 * 2.0) - 1.0  # [-1,1]

        def jaccard(a_tokens, b_tokens):
            A, B = set(a_tokens), set(b_tokens)
            if not A or not B: return 0.0
            return len(A & B) / len(A | B)

        for listener in self.agents:
            for speaker_id, utt in self.last_utterances.items():
                if listener.id == speaker_id:
                    continue

                # baseline reward from fitness
                r = base_r.get(speaker_id, 0.0)

                # referential bonus: overlap(utter_tokens, speaker_last_tokens)
                speaker = next((x for x in self.agents if x.id == speaker_id), None)
                if speaker is not None:
                    utt_tokens = utt.split()
                    act_tokens = getattr(speaker, "_last_tokens", []) or []
                    r += REFERENTIAL_BONUS * jaccard(utt_tokens, act_tokens)

                # clamp to [-1,1]
                r = max(-1.0, min(1.0, r))

                speaker = next((x for x in self.agents if x.id == speaker_id), None)
                if speaker is not None:
                    listener._maybe_learn_numeric_from(speaker)

                listener.learn_from_feedback(utt, reward=r, lr=PHASE2_LR)

        # entropy anti-collapse
        for a in self.agents:
            assoc = a.utterance_memory["associations"]
            if len(assoc) >= 2:
                spread = max(assoc.values()) - min(assoc.values())
                if spread < 0.2:
                    for s in a.symbol_drift:
                        a.symbol_drift[s] *= 0.97

        # If utterances collapsed to 1–2 types, nudge exploration by decaying LMs
        uniq_utts = len({u for u in self.last_utterances.values()})
        if uniq_utts <= 2:
            for a in self.agents:
                try:
                    a.lm.decay(rate=0.99)
                except Exception:
                    pass
    
    def reward_penalty_phase(self):
        """
        Apply simple reinforcement dynamics based on teaching outcomes and trust.
        Rewards improve fitness and energy; penalties reduce trust.
        """
        for agent in self.agents:
            # Baseline from past interactions
            net_outcome = sum(v for (_, v) in agent.interaction_memory[-10:]) if hasattr(agent, "interaction_memory") else 0.0

            # Convert to reward signal
            reward_signal = max(-1.0, min(1.0, net_outcome * 0.1))

            # Energy and trust adjustments
            if reward_signal > 0:
                agent.energy += 0.05 * reward_signal
                agent.trust_bias = getattr(agent, "trust_bias", 0.0) + 0.01 * reward_signal
                agent.own_fitness += reward_signal
            elif reward_signal < 0:
                agent.energy -= 0.05 * abs(reward_signal)
                agent.trust_bias = getattr(agent, "trust_bias", 0.0) - 0.01 * abs(reward_signal)
                # Apply a small penalty to overall trust network
                for pid in agent.social_memory.keys():
                    agent.adjust_trust(pid, amount=-0.005 * abs(reward_signal), channel=4)
        
        for agent in self.agents:
            agent.energy *= random.uniform(0.96, 0.99)
            # prevent runaway accumulation
            agent.energy = min(agent.energy, 100.0)

    def communicate(self, speaker, listener, utterance):
        """
        Simple communication feedback loop.
        Success if listener has seen the utterance before (shared memory).
        Adjusts energy and trust thresholds to reward comprehension.
        """
        shared = utterance in listener.utterance_memory["usage_count"]
        coop_avg = (speaker.traits["cooperation_weight"] + listener.traits["cooperation_weight"]) / 2
        success = shared and random.random() < coop_avg

        if success:
            # Reward energy and trust relaxation
            delta_e = 0.5
            speaker.energy = min(100.0, speaker.energy + delta_e)
            listener.energy = min(100.0, listener.energy + delta_e)

            # communication success → lower trust threshold a bit
            speaker.traits["trust_threshold"] = max(
                0.0, speaker.traits["trust_threshold"] - 0.03
            )
            listener.traits["trust_threshold"] = max(
                0.0, listener.traits["trust_threshold"] - 0.03
            )

            # and strengthen mutual trust (reliability / collaboration)
            speaker.adjust_trust(listener.id, amount=+0.02, channel=2)
            listener.adjust_trust(speaker.id, amount=+0.02, channel=2)
            speaker.adjust_trust(listener.id, amount=+0.01, channel=3)
            listener.adjust_trust(speaker.id, amount=+0.01, channel=3)
        else:
            # Penalise slight energy loss and raise trust threshold
            delta_e = 0.3
            speaker.energy = max(0.0, speaker.energy - delta_e)
            listener.energy = max(0.0, listener.energy - delta_e)
            speaker.traits["trust_threshold"] = min(
                1.0, speaker.traits["trust_threshold"] + 0.005
            )
            listener.traits["trust_threshold"] = min(
                1.0, listener.traits["trust_threshold"] + 0.005
            )

            # small trust penalty on both sides (competence channel)
            speaker.adjust_trust(listener.id, amount=-0.01, channel=4)
            listener.adjust_trust(speaker.id, amount=-0.01, channel=4)

            # decay failed association
            old = speaker.utterance_memory["associations"].get(utterance, 0.0)
            speaker.utterance_memory["associations"][utterance] = old * 0.9

    # -----------------------------
    # Teaching (optional)
    # -----------------------------
    def crossover_program(prog_a, prog_b):
        try:
            len_a, len_b = len(prog_a), len(prog_b)
            if len_a == 0 or len_b == 0:
                return prog_a[:] if len_a >= len_b else prog_b[:]
            cut_a = random.randrange(len_a)
            cut_b = random.randrange(len_b)
            child = prog_a[:cut_a] + prog_b[cut_b:]
            if len(child) == 0:
                child = (prog_a if random.random() < 0.5 else prog_b)[:]
            return child
        except Exception:
            return prog_a[:]

    def teaching_phase(self):
        """
        Hybrid teaching phase (trust-selective):
        1. High-fitness agents act as teachers.
        2. Each teacher selects students based on trust & willingness.
        3. Program + semantic learning both occur.
        4. Rewards and energy adjustments propagate through both sides.
        """
        ranked = sorted(self.agents, key=lambda a: a.total_fitness, reverse=True)
        top_n = max(1, int(len(self.agents) * TEACH_TOP_FRACTION))
        teachers = ranked[:top_n]

        for teacher in teachers:
            # --- only some teachers act each gen ---
            if random.random() > TEACH_PROB:
                continue

            # choose student candidates weighted by teacher's trust
            candidates = [a for a in self.agents if a.id != teacher.id]
            if not candidates:
                continue

            # compute willingness scores
            weights = []
            for c in candidates:
                will = max(0.0, teacher.teaching_willingness(c.id))
                # low-trust partners still possible but rarer
                weights.append(0.05 + will)

            student = random.choices(candidates, weights=weights)[0]

            # just before or after a teacher tries to teach
            student.teaching_attempted = True
            teacher.teaching_attempted = True

            # --- skip if trust threshold not met ---
            if teacher.teaching_willingness(student.id) < teacher.traits.get("trust_threshold", 0.3):
                continue

            # ------------------------------
            # 1. Program inheritance (existing logic)
            # ------------------------------
            new_prog = Coordinator.crossover_program(student.program, teacher.program)
            from evolution.programs import mutate_program
            new_prog = mutate_program(new_prog)

            if IMPROVE_ONLY_TEACH:
                old_fit = student.own_fitness
                try:
                    tmp_fit = run_program(new_prog)
                except Exception:
                    tmp_fit = -1e9
                if tmp_fit > old_fit:
                    student.program = new_prog
                    outcome = (tmp_fit - old_fit) * OUTCOME_SCALE
                    student.remember_interaction(teacher.id, outcome=outcome, gen_index=self.generation_index)
                    teacher.remember_interaction(student.id, outcome=outcome * 0.5, gen_index=self.generation_index)
            else:
                student.program = new_prog

            # ------------------------------
            # 2. Semantic teaching (new layer)
            # ------------------------------
            bundle = teacher.export_semantic_bundle(max_keys=1)
            if not bundle:
                continue

            word = bundle["tokens"][0]
            expected_vec = bundle["vecs"][word]

            # teacher performs a teaching attempt
            teach_reward, teach_penalty, sim = teacher.teach_student(
                student, word, current_gen=self.generation_index
            )

            # student evaluates understanding
            eval_reward = student.evaluate_teaching(
                teacher.id, word, expected_vec
            )

            # teacher receives reinforcement from student's evaluation
            teacher.apply_teaching_reward(student.id, eval_reward)

            # --- world log (diagnostic) ---
            try:
                self.world.append_text(
                    "/notes.txt",
                    f"[TeachPhase] A{teacher.id}->{student.id} "
                    f"trust={teacher.teaching_willingness(student.id):.2f} "
                    f"word={word} sim={sim:.2f} "
                    f"teachR={teach_reward:.2f} evalR={eval_reward:.2f}\n"
                )
            except Exception:
                pass

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

    # -----------------------------
    # Child creation (with learning inheritance)
    # -----------------------------
    # def make_child(self, p1, p2, p3, new_id):
    #     """
    #     Tri-parent child with:
    #     - damped + noisy trait inheritance
    #     - program mutation based on a random parent's program
    #     - cultural inheritance of utterance associations + symbol drift
    #     - fresh social memory (to avoid runaway cliques), but you can
    #       carry a tiny seed if desired (kept empty here for robustness)
    #     """
    #     child = Agent(new_id)

    #     # 1) Traits (pull toward neutral + noise)
    #     for key in child.traits:
    #         inherited = random.choice([p1.traits[key], p2.traits[key], p3.traits[key]])
    #         inherited = self.damp_trait(inherited, strength=0.08)
    #         inherited += random.uniform(-0.02, 0.02)
    #         child.traits[key] = self._clamp(inherited, 0.0, 1.0)

    #     # memory traits
    #     child.memory_influence = random.choice([p1.memory_influence, p2.memory_influence, p3.memory_influence])
    #     child.memory_decay_rate = self._clamp(
    #         random.choice([p1.memory_decay_rate, p2.memory_decay_rate, p3.memory_decay_rate]) + random.uniform(-0.02, 0.02),
    #         0.0, 1.0
    #     )

    #     # 2) Program + counting system inheritance 🌱
    #     # Pick one parent for genetic & cultural inheritance
    #     chosen_parent = random.choice([p1, p2, p3])
    #     parent_prog = chosen_parent.program
    #     child.mutate(parent_prog, parent_agent=chosen_parent)

    #     # 3) Cultural inheritance: language learning
    #     #    - associations: blended & decayed
    #     #    - symbol drift: blended & decayed
    #     child.utterance_memory["associations"] = self._blend_assoc(p1, p2, p3, noise=0.03, decay=0.90)
    #     # usage_count starts fresh (prevents ancient dominance)
    #     child.utterance_memory["usage_count"].clear()

    #     child.symbol_drift = self._blend_symbol_drift(p1, p2, p3, noise=0.03, decay=0.95)

    #     # 4) Social memory starts clean (keeps dynamics healthy)
    #     child.social_memory.clear()
    #     child.lineage_score = 0.0

    #     # scalar memory channels reset
    #     child.memory["last_fitness"] = 0.0
    #     child.memory["last_fitness_change"] = 0.0
    #     child.memory["cooperation_success"] = 0.0
    #     child.memory["novelty_success"] = 0.0

    #     # Energy: newborn starts slightly below max
    #     child.energy = ENERGY_MAX * 0.75

    #     # 5) Numeric-symbol & counting system inheritance 🌱
    #     parent_maps = [getattr(p, "symbol_map", {}) for p in (p1, p2, p3) if hasattr(p, "symbol_map")]
    #     child.symbol_map = {}

    #     # Ensure counting system sees the same mapping as the agent
    #     if hasattr(child, "counting") and hasattr(child, "symbol_map"):
    #         try:
    #             child.counting.set_symbol_map(child.symbol_map)
    #         except Exception:
    #             pass

    #     # Merge all parent mappings (last parent wins if conflict)
    #     for m in parent_maps:
    #         if not m:
    #             continue
    #         for k, v in m.items():
    #             child.symbol_map[k] = v

    #     # If none of the parents had a map, initialise a fresh one
    #     if not child.symbol_map:
    #         cs = CountingSystem()
    #         child.symbol_map = {i: cs.get_symbol(i) for i in range(cs.base)}
    #         child.counting = cs
    #     else:
    #         # If at least one parent had a system, merge their counting bases
    #         parent_systems = [getattr(p, "counting", None) for p in (p1, p2, p3) if hasattr(p, "counting")]
    #         if parent_systems:
    #             cs = CountingSystem()
    #             cs.merge_from(*[s for s in parent_systems if s])
    #             child.counting = cs
    #         else:
    #             # fallback if maps exist but no counting obj
    #             child.counting = CountingSystem()

    #     # Ensure numeric symbol map initialized cleanly
    #     if not hasattr(child, "symbol_map"):
    #         child.symbol_map = {}

    #     # Always initialise history
    #     child.symbol_map_history = {}

    #     return child

    from collections import defaultdict

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

        child = Agent(new_id)

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
        child.vocab = set().union(p1.vocab, p2.vocab, p3.vocab)
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
                base = np.array(self._rand_vec(32), dtype=float)

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
                lsem[a] = defaultdict(float, nbrs)

        # Then: merge parent link graphs
        for plinks in parent_links:
            for a, nbrs in plinks.items():
                # Ensure row is a defaultdict(float)
                row = lsem.get(a)
                if row is None or not isinstance(row, defaultdict):
                    row = lsem[a] = defaultdict(float, row or {})
                # Accumulate weights safely (no KeyError even if row is plain dict)
                for b, w in nbrs.items():
                    row[b] = row.get(b, 0.0) + (w * (1.0 / 3.0))

        # prune tiny edges
        for a, nbrs in list(lsem.items()):
            for b, w in list(nbrs.items()):
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

        child.semantic["families"] = merged_families
        child.semantic["family_counter"] = (max(fam_ids) + 1) if fam_ids else 1

        # ---------------------------------------------------
        # 5) LINGUISTIC MEMORY / ASSOCIATIONS
        # ---------------------------------------------------
        child.utterance_memory["associations"] = self._blend_assoc(
            p1, p2, p3,
            noise=0.02,
            decay=0.90
        )
        child.utterance_memory["usage_count"].clear()

        # ---------------------------------------------------
        # 6) NUMERIC-SYMBOL + COUNTING SYSTEM
        # ---------------------------------------------------
        child.symbol_map = {}
        for m in (p1.symbol_map, p2.symbol_map, p3.symbol_map):
            for k, v in m.items():
                child.symbol_map[k] = v

        cs = CountingSystem()
        cs.merge_from(p1.counting, p2.counting, p3.counting)
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
    # Language Phase (Challenge semantics integrated)
    # -----------------------------
    def run_language_phase(self):
        self.last_utterances = {}

        seeds = self.semantic_seeds
        words = self.cached_dictionary_words

        # 0) New challenge setup (as you had)

        for a in self.agents:
            a.challenge_guess = random.randint(0, 4)
            a.challenge_system = self.challenge

        self.challenge.new_challenge()
        self.challenge.assign_liars(self.agents)

        # 1) Each agent emits one utterance we can actually *see* this gen.
        #    (We also push it to shared notes via each agent's own API.)
        for a in self.agents:
            try:
                utt = a.produce_utterance()
                self.last_utterances[a.id] = utt
                if a.id in self.agent_apis:
                    self.agent_apis[a.id].append_text("/notes.txt", f"A{a.id}: {utt}\n", scope="world")
            except Exception:
                pass

        # --- Communication feedback loop ---
        ids = list(self.last_utterances.keys())
        for _ in range(len(ids)):
            if len(ids) < 2:
                break
            speaker_id, listener_id = random.sample(ids, 2)
            speaker = next(a for a in self.agents if a.id == speaker_id)
            listener = next(a for a in self.agents if a.id == listener_id)
            utt = self.last_utterances[speaker_id]
            self.communicate(speaker, listener, utt)

        # 2) Light world-ingest (as you had; safe-guarded)
        for a in self.agents:
            try:
                a.language_world_ingest_step()
            except Exception:
                pass

        # 3) Build dictionary-derived vocabulary and inject

        # 4) FIRST semantic teaching pass
        self.semantic_teaching_phase()

        # 5) Gossip
        self.gossip_exchange()

        # 6) SECOND semantic teaching pass (consolidation)
        self.semantic_teaching_phase()

        # 7) Listening & semantic reward (keep your existing logic, but ensure
        #    it uses self.last_utterances, which we now populate above)
        def jaccard(a_tokens, b_tokens):
            A, B = set(a_tokens), set(b_tokens)
            if not A or not B: return 0.0
            return len(A & B) / len(A | B)

        # Base reward from final fitness is applied later in language_feedback().
        # Here we can optionally do a small referential nudge, or keep it simple.
        # (We’ll keep it simple here and let language_feedback() do the heavy lift.)

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
            agent.last_utterance = agent.speak_number(agent.challenge_guess)

        self.ledger.reset_gen()

        # language phase
        self.run_language_phase()

        # -------------------------------------------------------
        # HELP ANSWERING (ONCE EARLY IN GEN)
        # -------------------------------------------------------
        for a in self.agents:
            if hasattr(a, "maybe_answer_numeric_help"):
                a.maybe_answer_numeric_help()

        # -------------------------------------------------------
        # TEACHING INGESTION
        # -------------------------------------------------------
        for a in self.agents:
            if hasattr(a, "process_numeric_teaching"):
                a.process_numeric_teaching()

        # -------------------------------------------------------
        # ARCHETYPE CLASSIFICATION (needed for homeostasis)
        # -------------------------------------------------------
        self.assign_archetypes(self.agents, self.classify_agent_archetype)

        # -------------------------------------------------------
        # TASK GENERATION (homeostatic mix)
        # -------------------------------------------------------

        # Build a mixed batch of tasks for this generation
        # For example: 4 tasks per generation
        self.active_tasks = self._build_task_batch(
            num_tasks = 4,
            gen = self.generation_index
        )

        # Agents attempt tasks
        for ag in self.agents:
            ag.try_solve_tasks(self.active_tasks, self.generation_index)

        # -------------------------------------------------------
        # TASK SOLVING
        # -------------------------------------------------------
        # if hasattr(self, "active_tasks") and self.active_tasks:
        #     for a in self.agents:
        #         if hasattr(a, "try_solve_tasks"):
        #             a.try_solve_tasks(self.active_tasks, self.generation_index)

        # === POST-TASK EVALUATION HOOKS ===
        for task in self.active_tasks:
            if task.get("task_type") == "pref_align":
                # gather responses from agents
                responses = task.get("responses", [])

                # we expect exactly 2 entries
                if len(responses) == 2:
                    r1, r2 = responses[0], responses[1]

                    # ensure they picked something
                    if "choice" in r1 and "choice" in r2:
                        # check agreement
                        if r1["choice"] == r2["choice"]:
                            # agreed → fitness and trust reward
                            a_id = int(r1["agent_id"][1:])
                            b_id = int(r2["agent_id"][1:])

                            a = self.agent_by_id(a_id)
                            b = self.agent_by_id(b_id)

                            if a and b:
                                a.own_fitness += 0.2
                                b.own_fitness += 0.2

                                if hasattr(a, "adjust_trust"):
                                    a.adjust_trust(b_id, +0.05, channel=2)
                                    b.adjust_trust(a_id, +0.05, channel=2)

        # keep short task list
        if hasattr(self, "active_tasks"):
            self.active_tasks = self.active_tasks[-3:]

        # motivated action
        for agent in self.agents:
            action = agent.decide_action()
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
            a.energy = min(100.0, a.energy + 0.2)
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
            mapping = agent.numeric_semantic
            collisions = agent._numeric_collision_score()

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
                rel_centroid = a._centroid([i["vec"] for i in rel_fam.values()])
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
                ref_centroid = a._centroid([i["vec"] for i in ref_fam.values()])
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


        txt_file, csv_file = _get_log_filenames(self)
        summary = compute_generation_summary(self)

        append_generation_to_csv(summary, csv_file)
        write_generation_report(self, txt_file, self.generation_index)

        self.cached_dictionary_words = None

        # family updates and semantic drift
        for a in self.agents:
            # every 5 generations
            if self.generation_index % 5 == 0:
                if hasattr(a, "detect_semantic_families"):
                    a.detect_semantic_families()

            if self.generation_index % 100 == 0:
                a.debug_dump_semantics()

            # each generation
            a.semantic_drift_update()
            a._sanitize_vector_dims()
            if hasattr(a, "family_reinforcement_update"):
                a.family_reinforcement_update()
            # --- Update community semantic centroid ---

        # --- Community Semantics ---
        self.print_community_semantic_stats()
        for a in self.agents:
            if hasattr(a, "detect_semantic_gaps"):
                a.detect_semantic_gaps(self.community_semantic)
        self.print_semantic_gap_stats()
        self.semantic_alignment_tasks = []

        print(self.summarize_dialogues(last_n=50))

        print(f"Generation {self.generation_index} running...")

    # =======================================================
    # TASKS & CHALLENGE UPDATES
    # ======================================================

    def _generate_task_id(self):
        tid = f"T{self.next_task_id:04d}"
        self.next_task_id += 1
        return tid

    def load_tasks(self, filename="/tasks.json"):
        """Load tasks from a shared JSON file inside the sandbox."""
        try:
            raw = self.world.read_json(filename)
            if not raw:
                return []
            if isinstance(raw, dict):
                return [raw]
            if isinstance(raw, list):
                return raw
            return []
        except Exception as e:
            print("TASK LOAD ERROR:", e)
            return []

    def generate_numeric_compare_task(self):
        """
        Produces tasks where A and B may be:
            - single-digit tokens (existing behaviour)
            - OR multi-token base-16 numbers, e.g. "qa hu" for 0x10.

        Agents must use their existing CountingSystem and
        token maps to interpret multi-token sequences.
        """

        # choose a *reference* agent with stable symbol_map
        ref = random.choice(self.agents)
        smap = getattr(ref, "symbol_map", None)
        if not smap:
            return None

        base = getattr(ref.counting, "base", 16)
        max_n = base ** 2 + random.randint(0, base * 4)   # let them see 2-digit numbers

        # two random integers
        A_val = random.randint(0, max_n)
        B_val = random.randint(0, max_n)

        # convert each to multi-token base-16 numeral
        A_tokens = ref.speak_number(A_val).split()
        B_tokens = ref.speak_number(B_val).split()

        # flatten back into space-separated tokens
        A_str = " ".join(A_tokens)
        B_str = " ".join(B_tokens)

        return {
            "task_id": self._generate_task_id(),
            "task_type": "compare_numbers",
            "instruction": {
                "base": base,
                "format": "compare",
                "description": "Which number is larger?"
            },
            "data": {
                "A": A_str,
                "B": B_str
            },
            "responses": []
        }

    def generate_cooperative_compare_task(self):
        """
        Create a cooperative numeric comparison task:
          - pick a reference agent to define the 'ground truth'
          - pick two distinct participants to solve it cooperatively
          - encode A / B using the ref agent's number system
        """
        if not self.agents or len(self.agents) < 2:
            return None

        ref = random.choice(self.agents)
        # simple: sample two random numbers in [0, base^2)
        base = getattr(ref.counting, "base", 8)
        max_val = base * base

        a_val = random.randint(0, max_val - 1)
        b_val = random.randint(0, max_val - 1)
        if a_val == b_val:
            # nudge to avoid constant equality
            b_val = (b_val + 1) % max_val

        A_tokens = ref.counting.interpret(a_val)   # assuming you have this helper
        B_tokens = ref.counting.interpret(b_val)

        A_phrase = " ".join(ref.counting.get_symbol(d) for d in A_tokens)
        B_phrase = " ".join(ref.counting.get_symbol(d) for d in B_tokens)

        # choose two participants
        participants = random.sample(self.agents, 2)
        assigned = [participants[0].id, participants[1].id]

        task = {
            "task_id": self._generate_task_id(),
            "task_type": "cooperative_compare_numbers",
            "data": {
                "A": A_phrase,
                "B": B_phrase,
                "a_val": a_val,       # optional, for evaluation
                "b_val": b_val,
                "ref_agent": ref.id,
            },
            "assigned_agents": assigned,
        }

        # optional: log for debugging
        try:
            line = (
                f"COOP_TASK {task['task_id']} ref=A{ref.id} "
                f"agents={assigned} A='{A_phrase}' B='{B_phrase}' "
                f"a_val={a_val} b_val={b_val}\n"
            )
            self.world.append_text("/coop_tasks.txt", line)
        except Exception:
            pass

        return task

    def generate_agreement_dialogue_task(self):
        # pick 2 distinct agents
        if len(self.agents) < 2:
            return None

        a, b = random.sample(self.agents, 2)
        tid = self._generate_task_id()

        # pick two random utterances or numeric tokens as the “topic”
        # This should be *ambiguous* to encourage negotiation
        try:
            u1 = a.produce_utterance()
            u2 = b.produce_utterance()
        except Exception:
            u1, u2 = "su", "tol"   # fallback

        return {
            "task_id": tid,
            "task_type": "agreement_dialogue",
            "topic": f"{u1} || {u2}",
            "assigned_agents": [a.id, b.id],
        }

    def generate_explain_partner_tasks(self, rounds=10):
        tasks = []
        for _ in range(rounds):
            a, b = random.sample(self.agents, 2)

            # choose a base task to explain (compare_numbers)
            base = self.generate_numeric_compare_task()
            b_answer = b.solve_task(base)

            tasks.append({
                "task_type": "explain_partner",
                "task_id": self.new_task_id(),
                "assigned_agents": [a.id, b.id],
                "partner_answer": b_answer.get("answer", ""),
            })
        return tasks

    def generate_token_compression_tasks(self, rounds=10):
        tasks = []
        for _ in range(rounds):
            ag = random.choice(self.agents)
            # choose random utterance from logs or fabricate
            utt = self.sample_random_utterance() or ag.produce_utterance()

            tasks.append({
                "task_type": "token_compress",
                "task_id": self.new_task_id(),
                "assigned_agents": [ag.id],
                "utterance": utt,
            })
        return tasks

    def generate_preference_alignment_tasks(self, rounds=10):
        tasks = []
        roots = ["tar", "rin", "muk", "vak", "tol", "bel", "zev", "ka", "lo", "su"]

        for _ in range(rounds):
            A, B, P = random.sample(roots, 3)
            a, b = random.sample(self.agents, 2)

            tasks.append({
                "task_type": "pref_align",
                "task_id": self.new_task_id(),
                "assigned_agents": [a.id, b.id],
                "A": A,
                "B": B,
                "pivot": P,
            })
        return tasks

    def generate_reconcile_counts_task(self):
        """
        Habit-learner / pattern-compression task.

        Two agents receive *different descriptions* of a number.
        Their goal is to output the SAME normalized value after decoding.

        Pressure:
        - habit learners: pattern→normal form
        - cooperators: converge on a shared numeric interpretation
        - explorers: less rewarded (stabilising force)
        """
        if len(self.agents) < 2:
            return None

        # choose two different participants
        a, b = random.sample(self.agents, 2)

        # choose reference agent to generate canonical numeric form
        ref = random.choice(self.agents)
        base = getattr(ref.counting, "base", 8)

        # draw a value
        val = random.randint(0, base**2 - 1)

        # each participant gets *its own* encoding of the same number
        def encode(agent, value):
            try:
                toks = agent.speak_number(value).split()
            except Exception:
                toks = ref.speak_number(value).split()
            return " ".join(toks)

        A_view = encode(a, val)
        B_view = encode(b, val)

        tid = self._generate_task_id()

        return {
            "task_id": tid,
            "task_type": "reconcile_counts",
            "value": val,
            "views": {
                a.id: A_view,
                b.id: B_view,
            },
            "assigned_agents": [a.id, b.id],
            "responses": []
        }
    
    def evaluate_cooperative_task(self, task, responses):
        # task was created with key "assigned_agents"
        a_id, b_id = task["assigned_agents"]

        if a_id not in responses or b_id not in responses:
            return

        ra = responses[a_id]
        rb = responses[b_id]

        # Agreement score
        score = 0

        # 1. same relational category
        if ra["relation"] == rb["relation"]:
            score += 1.0

        # 2. same chosen answer token(s)
        if ra["answer"] == rb["answer"]:
            score += 1.0

        # 3. semantic match of shared tokens
        if ra.get("shared_token") and rb.get("shared_token"):
            if ra["shared_token"] == rb["shared_token"]:
                score += 1.0

        # 4. correct numeric comparison
        # Use the phrases we stored under "A" and "B"
        A_full = task["data"]["A"]
        B_full = task["data"]["B"]
        valA = self.agents[a_id]._decode_number_phrase(A_full)
        valB = self.agents[b_id]._decode_number_phrase(B_full)

        correct_rel = (
            REL_GT if valA > valB else
            REL_LT if valB > valA else
            REL_EQ
        )

        if ra["relation"] == correct_rel:
            score += 0.5
        if rb["relation"] == correct_rel:
            score += 0.5

        # 5. reward trust between partners
        self.agents[a_id].adjust_trust(b_id, +0.05 * score, channel=3)
        self.agents[b_id].adjust_trust(a_id, +0.05 * score, channel=3)

        # 6. reward cooperation fitness
        self.agents[a_id].cooperation_bonus += score
        self.agents[b_id].cooperation_bonus += score

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

    def update_community_semantic(self, sample_k=200, smooth=0.2):
        """
        Build/refresh the community semantic centroid.
        Called once per generation.
        """
        import numpy as np

        aggregate = {}
        counts = {}

        # -----------------------------
        # Collect snapshots from agents
        # -----------------------------
        for ag in self.agents:
            snap = ag.export_semantic_snapshot(k=sample_k)
            for tok, vec in snap.items():

                # Skip identity tokens (A23 etc)
                if isinstance(tok, str) and tok.startswith("a") and tok[1:].isdigit():
                    continue

                # Initialise sum
                if tok not in aggregate:
                    aggregate[tok] = np.array(vec, dtype=float)
                    counts[tok] = 1
                else:
                    aggregate[tok] += np.array(vec, dtype=float)
                    counts[tok] += 1

        # -----------------------------
        # Compute averaged centroids
        # -----------------------------
        for tok, v in aggregate.items():
            newv = (v / counts[tok]).tolist()

            if tok in self.community_semantic["vecs"]:
                oldv = self.community_semantic["vecs"][tok]
                # smoothing = slow drift toward consensus
                blended = [
                    (1 - smooth) * o + smooth * n
                    for o, n in zip(oldv, newv)
                ]
                self.community_semantic["vecs"][tok] = blended
            else:
                self.community_semantic["vecs"][tok] = newv

        self.community_semantic["counts"] = counts
        self.community_semantic["last_update_gen"] = self.generation_index

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
        from collections import defaultdict
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

    # =====================================================
    # COMMUNITY SEMANTIC MAP (MIXED STRENGTH, MODE C)
    # =====================================================
    def _compute_token_stats(self, token):
        """
        For a given token, gather all agent vectors and compute:
          - mean vector
          - mean distance to the mean
          - coverage (fraction of agents that know it)
        """
        vecs = []
        for ag in self.agents:
            sem = getattr(ag, "semantic", None)
            if not sem:
                continue
            v = sem.get("vecs", {}).get(token)
            if v is None:
                continue
            vecs.append((ag.id, np.array(v, dtype=float)))

        if not vecs:
            return None, 0.0, 0.0

        arr = np.stack([v for (_, v) in vecs], axis=0)
        mean_vec = arr.mean(axis=0)

        dists = np.linalg.norm(arr - mean_vec, axis=1)
        mean_dist = float(dists.mean()) if len(dists) > 0 else 0.0

        coverage = len(vecs) / max(1, len(self.agents))

        return mean_vec, mean_dist, coverage

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

    def update_community_semantic(self, max_tokens=5000):
        """
        Mixed-strength EM update (Mode C):
          - all tokens get *some* community centroid
          - high-confidence tokens update fast (strong anchor)
          - low-confidence tokens update slowly (preserve exploration)
        """
        if not self.agents:
            return

        com = self.community_semantic
        c_vecs = com["vecs"]
        c_counts = com["counts"]
        c_conf = com["confidence"]

        # Collect candidate tokens from population
        token_set = set()
        for ag in self.agents:
            sem = getattr(ag, "semantic", None)
            if not sem:
                continue
            token_set |= set(sem.get("vecs", {}).keys())

        if not token_set:
            return

        # Limit for sanity
        tokens = list(token_set)
        random.shuffle(tokens)
        tokens = tokens[:max_tokens]

        for tok in tokens:
            mean_vec, mean_dist, coverage = self._compute_token_stats(tok)
            if mean_vec is None:
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

    # =====================================================
    # EMERGENT DIALOGUE ARENA
    # =====================================================

    def run_dialogues(self, max_pairs_per_gen=30, max_turns_per_pair=2):
        """
        Let agents initiate pairwise dialogues based on their own choices.

        - Agents decide *if* they want to talk and *whom* to talk to.
        - Chatrooms are keyed by unordered pairs (a,b).
        - Each chatroom gets a few alternating turns.
        """
      # PERF(f"== Begin dialogues gen {getattr(self, 'generation_index', '?')} ==")
        if not hasattr(self, "dialogue_log"):
            self.dialogue_log = []  # persistent over generations if you like

      # PERF("Collecting partner proposals…")
        # 1) Agents propose partners
        proposed_pairs = set()
        for agent in self.agents:
            partner_id = agent.choose_conversation_partner(self.agents)
            if partner_id is None:
                continue

            key = tuple(sorted((agent.id, partner_id)))
            proposed_pairs.add(key)

        if not proposed_pairs:
            return

        # Limit total chatrooms per generation to avoid explosion
        proposed_list = list(proposed_pairs)
        random.shuffle(proposed_list)
        chosen_pairs = proposed_list[:max_pairs_per_gen]

        # index agents by id for quick lookup
        id_to_agent = {a.id: a for a in self.agents}

      # PERF(f"Total proposed_pairs={len(proposed_pairs)}")

        # 2) Run dialogues
        for a_id, b_id in chosen_pairs:
          # PERF(f"Starting room {a_id}-{b_id}")
            a = id_to_agent.get(a_id)
            b = id_to_agent.get(b_id)
            if a is None or b is None:
                continue

            room_turns = []
            last_utter = None
            last_speaker_id = None

            # Alternate speaker turns
            speaker_order = [a, b] * max_turns_per_pair

            for speaker in speaker_order:
                listener = b if speaker is a else a

                # Listener receives last message (if from the other side)
                if last_utter is not None and last_speaker_id != listener.id:
                    try:
                        listener.receive_message(last_speaker_id, last_utter)
                    except Exception:
                        pass

                # Speaker produces a new utterance (with possible nickname address)
                try:
                    if hasattr(speaker, "produce_addressed_utterance"):
                        utter = speaker.produce_addressed_utterance(listener)
                    else:
                        utter = speaker.produce_utterance()
                except Exception:
                    utter = None

                if not utter:
                    # still record a "quiet" turn if you like
                    utter = ""

                speaker.mark_spoken_turn()
                last_utter = utter
                last_speaker_id = speaker.id

                room_turns.append({
                    "speaker_id": speaker.id,
                    "listener_id": listener.id,
                    "utterance": utter,
                })

            # 3) Log the conversation for diagnostics
            self.dialogue_log.append({
                "generation": getattr(self, "generation_index", None),
                "pair": (a_id, b_id),
                "turns": room_turns[-10:],  # keep last few
            })

        # Optional: keep dialogue log from growing forever
        if len(self.dialogue_log) > 5000:
            self.dialogue_log = self.dialogue_log[-5000:]

        with open("dialogue_log.txt", "a") as f:
            for d in self.dialogue_log[-5:]:
                f.write(f"Gen {d['generation']} Pair {d['pair']}\n")
                for t in d["turns"]:
                    f.write(f"  A{t['speaker_id']} → A{t['listener_id']}: {t['utterance']}\n")
                f.write("\n")

    def summarize_dialogues(self, last_n=100):
        if not hasattr(self, "dialogue_log") or not self.dialogue_log:
            return "No dialogues yet."

        recent = self.dialogue_log[-last_n:]
        pairs = {}
        for rec in recent:
            key = tuple(sorted(rec["pair"]))
            pairs.setdefault(key, 0)
            pairs[key] += len(rec["turns"])

        lines = [f"Recent dialogue pairs (last {last_n} records):"]
        for (a, b), count in sorted(pairs.items(), key=lambda x: -x[1])[:15]:
            lines.append(f"  A{a}–A{b}: {count} turns")

        return "\n".join(lines)