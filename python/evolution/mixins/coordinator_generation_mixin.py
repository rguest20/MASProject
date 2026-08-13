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


class CoordinatorGenerationMixin:
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

        # In the absence of a waiting human prompt, a short reading passage
        # gives the community fresh contextual English before it begins this
        # generation's ordinary language and task work.
        self.run_quiet_reading_cycle()

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

        # Repeated wrong attempts and long novelty droughts create a bounded
        # exploratory drive.  It alters later action choice this generation;
        # it is not deducted from fitness or energy.
        self.update_community_discomfort()

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
