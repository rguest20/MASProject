# agents/mixins/social_mixin.py

import random
from collections import defaultdict

from agents.agent_utils import clamp01

class SocialMixin:
    """
    Handles:
      • social memory
      • trust channels (7-dimensional)
      • interaction logging
      • reputation export/import
      • social decay
      • partner willingness scoring

    This mixin is intentionally passive — the Coordinator or
    Orchestration layer decides when social events occur.
    """

    # ----------------------------------------------------
    # Initialization (called from Agent._post_init)
    # ----------------------------------------------------
    def _init_social_system(self):
        # partner_id -> record
        self.social_memory = {}

        # 7 trust channels per partner
        self.trust_channels = defaultdict(lambda: {
            "affinity": 0.0,
            "reliability": 0.0,
            "generosity": 0.0,
            "competence": 0.0,
            "consistency": 0.0,
            "collaboration": 0.0,
            "safety": 0.0,
        })

        self.interaction_memory = []  # list of (id, outcome)
        self._max_social_history = 50

    # ----------------------------------------------------
    # Internal helper — auto-create partner record
    # ----------------------------------------------------
    def _ensure_partner_record(self, pid):
        rec = self.social_memory.get(pid)
        if rec is None:
            rec = {
                "interactions":     0,
                "mean_outcome":     0.0,
                "last_outcome":     0.0,
                "trust_delta":      0.0,
                "offspring_success": 0.0,
                "last_seen_gen":    -1,
            }
            self.social_memory[pid] = rec
        return rec

    # ----------------------------------------------------
    # Interaction logging
    # ----------------------------------------------------
    def remember_interaction(
        self, partner_id, outcome,
        offspring_success=None, gen_index=0,
        alpha_outcome=0.3, alpha_offspring=0.25,
        trust_gain=0.5
    ):
        """
        outcome > 0 : positive  
        outcome < 0 : negative  
        offspring_success is optional (genetic lineage effect)
        """

        rec = self._ensure_partner_record(partner_id)

        rec["interactions"] += 1
        rec["last_outcome"] = float(outcome)

        # Update rolling averages
        rec["mean_outcome"] = (
            (1 - alpha_outcome) * rec["mean_outcome"]
            + alpha_outcome * float(outcome)
        )

        if offspring_success is not None:
            rec["offspring_success"] = (
                (1 - alpha_offspring) * rec["offspring_success"]
                + alpha_offspring * float(offspring_success)
            )

        # adjust trust_delta (scalar)
        rec["trust_delta"] = max(
            -1000.0, min(1000.0, rec["trust_delta"] + trust_gain * float(outcome))
        )

        rec["last_seen_gen"] = int(gen_index)

        # also append to short-term memory
        self.interaction_memory.append((partner_id, float(outcome)))
        if len(self.interaction_memory) > self._max_social_history:
            self.interaction_memory.pop(0)

        self.reinforce_identity(partner_id, outcome)

    # ----------------------------------------------------
    # Trust channel updates
    # ----------------------------------------------------
    def update_trust_channels(self, partner_id, reward):
        """
        reward ∈ [-1, 1]
        Adjust 7 sub-channels for this relationship.
        """
        reward = float(reward)
        ch = self.trust_channels[partner_id]
        lr = 0.08  # master learning rate

        # meaning-preserving multipliers
        ch["affinity"]      += lr * reward
        ch["reliability"]   += lr * reward * 1.2
        ch["generosity"]    += lr * reward * 0.6
        ch["competence"]    += lr * reward * 1.4
        ch["consistency"]   += lr * reward * 0.5
        ch["collaboration"] += lr * reward
        ch["safety"]        += lr * reward * 0.7

        # clamp
        for k in ch:
            ch[k] = max(-2.0, min(2.0, ch[k]))

    # ----------------------------------------------------
    # Trust adjustment via scalar channel index
    # ----------------------------------------------------
    def adjust_trust(self, target_id, amount, channel):
        """
        Allows Coordinator or Teaching subsystem to update specific trust aspects.

        channel legend:
            1: affinity
            2: reliability
            3: collaboration
            4: competence
            5: consistency
            6: generosity
            7: safety
        """

        amount = float(amount)

        # small semantic reinforcement proportional to trust adjustment
        if hasattr(self, "reinforce_identity"):
            self.reinforce_identity(target_id, amount)

        ch = self.trust_channels[target_id]

        # ----------------------------
        # Channel-specific update
        # ----------------------------
        if channel == 1: ch["affinity"]      += amount
        elif channel == 2: ch["reliability"]   += amount
        elif channel == 3: ch["collaboration"] += amount
        elif channel == 4: ch["competence"]    += amount
        elif channel == 5: ch["consistency"]   += amount
        elif channel == 6: ch["generosity"]    += amount
        elif channel == 7: ch["safety"]        += amount
        else:
            # unknown → spread softly across all
            ch["affinity"]      += 0.15 * amount
            ch["reliability"]   += 0.25 * amount
            ch["competence"]    += 0.25 * amount
            ch["consistency"]   += 0.15 * amount
            ch["collaboration"] += 0.10 * amount
            ch["safety"]        += 0.10 * amount

        # clamp channels
        for k in ch:
            ch[k] = max(-2.0, min(2.0, ch[k]))

        # ----------------------------
        # Update social_memory record
        # ----------------------------
        if not hasattr(self, "_trust_delta_cap"):
            self._trust_delta_cap = 0.05   # max change per gen per relation

        # cap per-call change
        capped = max(-self._trust_delta_cap,
                     min(self._trust_delta_cap, amount))

        rec = self._ensure_partner_record(target_id)
        rec["trust_delta"] = max(
            -1000.0,
            min(1000.0, rec["trust_delta"] + capped)
        )

    # ----------------------------------------------------
    # Social memory decay
    # ----------------------------------------------------
    def decay_social_memory(self, decay_rate, gen_index, stale_after=50, prune_threshold=0.02):
        """
        Gradually forgets old/weak relationships.
        Coordinator calls this each generation.
        """

        if not self.social_memory:
            return

        new_mem = {}

        for pid, rec in self.social_memory.items():

            # decay running averages
            rec["mean_outcome"] *= (1.0 - 0.5 * decay_rate)
            rec["trust_delta"]  *= (1.0 - decay_rate)

            age = gen_index - rec.get("last_seen_gen", gen_index)

            # measure “tiny” relationships
            tiny = (
                abs(rec["mean_outcome"])
                + abs(rec["trust_delta"])
                + abs(rec["offspring_success"])
            ) < prune_threshold

            # stale + tiny → prune
            if not (age > stale_after and tiny):
                new_mem[pid] = rec

        self.social_memory = new_mem

    # ----------------------------------------------------
    # Export reputation (for gossip)
    # ----------------------------------------------------
    def export_reputation(self, top_k=5):
        """
        Returns a sorted list of (pid, reputation_strength)
        based on this agent's perception.
        """
        items = []
        for pid, rec in self.social_memory.items():
            rep = (
                0.5 * rec["mean_outcome"]
                + 0.3 * rec["trust_delta"] * 0.01
                + 0.2 * rec["offspring_success"]
            )
            items.append((pid, rep))

        items.sort(key=lambda x: abs(x[1]), reverse=True)
        return items[:top_k]

    # ----------------------------------------------------
    # Import reputation (gossip ingestion)
    # ----------------------------------------------------
    def receive_reputation(self, sender_id, rep_list, weight_from_sender=0.2):
        """
        Blend sender's reported reputations into our own.
        """
        # trust-based weighting
        sender_rec = self.social_memory.get(sender_id)
        if sender_rec:
            trust_factor = min(2.0, max(-0.5, sender_rec["trust_delta"] * 0.005))
            weight = weight_from_sender * (1.0 + trust_factor)
        else:
            weight = weight_from_sender

        for pid, rep_val in rep_list:
            if pid == self.id:
                continue  # ignore gossip about self

            rec = self._ensure_partner_record(pid)

            rec["mean_outcome"] = 0.9 * rec["mean_outcome"] + 0.1 * weight * rep_val
            rec["trust_delta"]  = 0.9 * rec["trust_delta"]  + 0.1 * weight * rep_val * 10.0

    # ----------------------------------------------------
    # Reputation of *self*
    # (for coordinator if needed)
    # ----------------------------------------------------
    @property
    def reputation_strength(self):
        """
        Derived from how reliable/consistent this agent is.
        Used to determine how much others trust this agent's gossip.
        """
        ch = self.trust_channels.get(self.id, {})
        rel = float(ch.get("reliability", 0.0))
        cns = float(ch.get("consistency", 0.0))

        base = 0.012 + 0.004 * (rel + cns)
        return max(0.005, min(0.03, base))
