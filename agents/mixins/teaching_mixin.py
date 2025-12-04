# agents/mixins/teaching_mixin.py

import random
import numpy as np
from agents.semantics import cos_sim


class TeachingMixin:
    """
    Handles all teaching behaviour:
      • motivation + willingness
      • cooldown dynamics
      • energy cost
      • semantic alignment
      • trust/tie updates
      • language + counting co-learning
      • emotion hooks (state_event)
      • stable logging + diagnostics
    """

    # ============================================================
    #  INITIALISATION (invoked from Agent._post_init)
    # ============================================================
    def _init_teaching_system(self):
        self.last_taught = 0
        self.teach_cooldown = 3
        self._last_teaching_sim = 0.0
        self._last_teaching_reward = 0.0

    # ============================================================
    #  TEACHING WILLINGNESS
    # ============================================================
    def teaching_willingness(self, partner_id):
        """
        Scalar willingness to teach a partner:
          competence > reliability > affinity > collaboration
        """
        ch = self.trust_channels.get(partner_id)
        if not ch:
            return 0.0

        return (
            0.40 * ch["competence"] +
            0.30 * ch["reliability"] +
            0.20 * ch["affinity"] +
            0.10 * ch["collaboration"]
        )

    # ============================================================
    #  MAIN TEACHING EVENT
    # ============================================================
    def teach_student(self, student, word, current_gen=0):
        """
        Attempt to teach a concept to a student.

        Pipeline:
          0. Preconditions
          1. Motivation
          2. Cooldown
          3. Energy cost
          4. Student vec initialisation
          5. Compute similarity
          6. Assign reward/penalty
          7. Adjust trust
          8. Adjust energy
          9. Update student semantic vector
         10. Optional: language + counting mutual learning
         11. Emotional feedback + logging
        """

        # --------------------------------------------------------
        # 0. Preconditions
        # --------------------------------------------------------
        if not hasattr(self, "semantic"):
            return 0.0, 0.0, 0.0
        if word not in self.semantic["vecs"]:
            return 0.0, 0.0, 0.0

        teacher_vec = self.semantic["vecs"][word]

        # --------------------------------------------------------
        # 1. Motivation
        # --------------------------------------------------------
        drive = self.traits.get("teaching_drive", 0.5)
        if random.random() > drive:
            return 0.0, 0.0, 0.0

        # --------------------------------------------------------
        # 2. Cooldown
        # --------------------------------------------------------
        chattiness = self.traits.get("chattiness", 0.5)
        base_cd = max(1, int(5 - 4 * chattiness))

        if current_gen - self.last_taught < self.teach_cooldown:
            return 0.0, 0.0, 0.0

        self.last_taught = current_gen
        # natural variation (slightly noisy but stable)
        self.teach_cooldown = max(1, int(np.random.normal(base_cd, 0.4)))

        # --------------------------------------------------------
        # 3. Energy Cost
        # --------------------------------------------------------
        patience = self.traits.get("patience", 0.5)
        energy_cost = 0.03 + 0.07 * patience
        self.energy = max(0.0, self.energy - energy_cost)

        # --------------------------------------------------------
        # 4. Student prepares vector
        # --------------------------------------------------------
        student._ensure_vec(word)
        student_vec = student.semantic["vecs"][word]

        # --------------------------------------------------------
        # 5. Similarity evaluation
        # --------------------------------------------------------
        sim = cos_sim(teacher_vec, student_vec)

        # Reward schedule
        if sim > 0.75:
            reward, penalty = 0.06, 0.0
        elif sim > 0.45:
            reward, penalty = 0.03, 0.0
        else:
            reward, penalty = 0.0, 0.06

        # --------------------------------------------------------
        # 6. Trust adjustment
        # --------------------------------------------------------
        trust_delta = (reward - penalty)
        self.update_trust_channels(student.id, trust_delta)
        student.update_trust_channels(self.id, trust_delta * 0.5)

        # --------------------------------------------------------
        # 7. Energy reward/penalty
        # --------------------------------------------------------
        self.energy = max(0.0, self.energy + reward * 0.3 - penalty * 0.2)
        student.energy = max(0.0, student.energy + reward * 0.2 - penalty * 0.1)

        # --------------------------------------------------------
        # 8. Student semantic alignment
        # --------------------------------------------------------
        student.semantic["vecs"][word] = (
            student_vec * 0.78 + teacher_vec * 0.22
        )

        # --------------------------------------------------------
        # 9. Optional cultural language learning
        # --------------------------------------------------------
        if random.random() < drive:
            try:
                self.language.learn_from(student.language)
                student.language.learn_from(self.language)
            except Exception:
                pass

        # --------------------------------------------------------
        # 10. Optional counting-system drift exchange
        # --------------------------------------------------------
        if random.random() < drive * 0.6:
            try:
                student.counting.learn_from(self.counting)
            except Exception:
                pass

        # --------------------------------------------------------
        # 11. Emotional state + logging
        # --------------------------------------------------------
        if reward > penalty:
            if hasattr(self, "state_event"):
                self.state_event("teaching_success")
            if hasattr(student, "state_event"):
                student.state_event("learning_success")
        else:
            if hasattr(self, "state_event"):
                self.state_event("teaching_failure")
            if hasattr(student, "state_event"):
                student.state_event("learning_failure")

        # optional world logging
        if hasattr(self, "api") and self.api:
            try:
                self.api.append_text(
                    "/notes.txt",
                    f"teach A{self.id}→A{student.id}: {word} "
                    f"sim={sim:.2f} Δ={reward-penalty:+.2f}\n",
                    scope="world"
                )
            except Exception:
                pass

        # diagnostics
        self._last_teaching_sim = sim
        self._last_teaching_reward = reward - penalty

        return reward, penalty, sim

    # ============================================================
    #  STUDENT-SIDE EVALUATION (used rarely by coordinator)
    # ============================================================
    def evaluate_teaching(self, teacher_id, word, expected_vec):
        """
        Student evaluates how well they learned a concept.
        Used in higher-level challenge systems or meta-learning.
        """

        self._ensure_vec(word)
        student_vec = self.semantic["vecs"][word]

        sim = cos_sim(student_vec, expected_vec)

        # reward schedule
        if sim > 0.8:
            reward = +1.0
        elif sim > 0.5:
            reward = +0.3
        elif sim > 0.3:
            reward = -0.1
        else:
            reward = -0.5

        # trust update
        self.update_trust_channels(teacher_id, reward * 0.4)

        # energy update
        self.energy = max(0.0, self.energy + 0.2 * reward)

        # language association update
        self.learn_from_feedback(word, reward)

        # emotional feedback
        if reward > 0:
            if hasattr(self, "state_event"):
                self.state_event("learning_success")
        else:
            if hasattr(self, "state_event"):
                self.state_event("learning_failure")

        # diagnostics
        self._last_teaching_sim = sim
        self._last_teaching_reward = reward

        return reward