"""
agents/cognition/teaching_system.py
Teaching system for semantic alignment and social learning.
"""

from __future__ import annotations

import random
import numpy as np

from agents.cognition.semantic_utils import cos_sim


class TeachingSystem:
    """Owns teaching mechanics; mixins should delegate here."""

    def __init__(self, owner):
        self.owner = owner

    def init(self):
        o = self.owner
        o.last_taught = 0
        o.teach_cooldown = 3
        o._last_teaching_sim = 0.0
        o._last_teaching_reward = 0.0
        o._last_learning_reward = 0.0

    def apply_teaching_reward(self, student_id, reward):
        o = self.owner
        try:
            reward = float(reward)
        except Exception:
            reward = 0.0

        o._last_teaching_reward = reward
        if hasattr(o, "energy"):
            o.energy = max(0.0, min(100.0, float(o.energy) + 0.05 * reward))

        if hasattr(o, "state_event"):
            o.state_event("teaching_success" if reward > 0 else "teaching_failure")

    def apply_learning_reward(self, teacher_id, reward):
        o = self.owner
        try:
            reward = float(reward)
        except Exception:
            reward = 0.0

        o._last_learning_reward = reward
        if hasattr(o, "energy"):
            o.energy = max(0.0, min(100.0, float(o.energy) + 0.05 * reward))

        if hasattr(o, "state_event"):
            o.state_event("learning_success" if reward > 0 else "learning_failure")

    def teaching_willingness(self, partner_id):
        o = self.owner
        ch = getattr(o, "trust_channels", {}).get(partner_id)
        if not ch:
            return 0.0
        return (
            0.40 * ch["competence"]
            + 0.30 * ch["reliability"]
            + 0.20 * ch["affinity"]
            + 0.10 * ch["collaboration"]
        )

    def teach_student(self, student, word, current_gen=0):
        o = self.owner

        if not hasattr(o, "semantic"):
            return 0.0, 0.0, 0.0
        if word not in o.semantic["vecs"]:
            return 0.0, 0.0, 0.0

        teacher_vec = o.semantic["vecs"][word]

        drive = o.traits.get("teaching_drive", 0.5)
        if random.random() > drive:
            return 0.0, 0.0, 0.0

        chattiness = o.traits.get("chattiness", 0.5)
        base_cd = max(1, int(5 - 4 * chattiness))

        if current_gen - o.last_taught < o.teach_cooldown:
            return 0.0, 0.0, 0.0

        o.last_taught = current_gen
        o.teach_cooldown = max(1, int(np.random.normal(base_cd, 0.4)))

        patience = o.traits.get("patience", 0.5)
        energy_cost = 0.03 + 0.07 * patience
        o.energy = max(0.0, o.energy - energy_cost)

        student._ensure_vec(word)
        student_vec = student.semantic["vecs"][word]

        sim = cos_sim(teacher_vec, student_vec)

        if sim > 0.75:
            reward, penalty = 0.06, 0.0
        elif sim > 0.45:
            reward, penalty = 0.03, 0.0
        else:
            reward, penalty = 0.0, 0.06

        trust_delta = (reward - penalty)
        o.update_trust_channels(student.id, trust_delta)
        student.update_trust_channels(o.id, trust_delta * 0.5)

        o.energy = max(0.0, o.energy + reward * 0.3 - penalty * 0.2)
        student.energy = max(0.0, student.energy + reward * 0.2 - penalty * 0.1)

        student.semantic["vecs"][word] = (student_vec * 0.78 + teacher_vec * 0.22)

        if random.random() < drive:
            try:
                o.language.learn_from(student.language)
                student.language.learn_from(o.language)
            except Exception:
                pass

        if random.random() < drive * 0.6:
            try:
                student.counting.learn_from(o.counting)
            except Exception:
                pass

        if reward > penalty:
            if hasattr(o, "state_event"):
                o.state_event("teaching_success")
            if hasattr(student, "state_event"):
                student.state_event("learning_success")
        else:
            if hasattr(o, "state_event"):
                o.state_event("teaching_failure")
            if hasattr(student, "state_event"):
                student.state_event("learning_failure")

        if hasattr(o, "api") and o.api:
            try:
                o.api.append_text(
                    "/notes.txt",
                    f"teach A{o.id}→A{student.id}: {word} "
                    f"sim={sim:.2f} Δ={reward - penalty:+.2f}\n",
                    scope="world",
                )
            except Exception:
                pass

        o._last_teaching_sim = sim
        o._last_teaching_reward = reward - penalty

        return reward, penalty, sim

    def evaluate_teaching(self, teacher_id, word, expected_vec):
        o = self.owner

        o._ensure_vec(word)
        student_vec = o.semantic["vecs"][word]

        sim = cos_sim(student_vec, expected_vec)

        if sim > 0.8:
            reward = +1.0
        elif sim > 0.5:
            reward = +0.3
        elif sim > 0.3:
            reward = -0.1
        else:
            reward = -0.5

        o.update_trust_channels(teacher_id, reward * 0.4)
        o.energy = max(0.0, o.energy + 0.2 * reward)
        o.learn_from_feedback(word, reward)

        if hasattr(o, "state_event"):
            o.state_event("learning_success" if reward > 0 else "learning_failure")

        o._last_teaching_sim = sim
        o._last_teaching_reward = reward
        return reward
