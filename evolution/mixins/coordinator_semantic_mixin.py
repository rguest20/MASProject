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


class CoordinatorSemanticMixin:
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
