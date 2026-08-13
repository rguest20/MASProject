"""
agents/cognition/task_system_v2.py
Task solving system (v2).
"""

import random
import re

from agents.agent_constants import REL_GT, REL_LT, REL_EQ


class TaskSystemSignalMixin:
    def _solve_referential_signal(self, task):
        """Interpret a peer's invented signal using this agent's lexicon."""
        data = task.get("data", {}) or {}
        signal = data.get("signal", "")
        ledger = getattr(self, "community_lexicon", None)
        referent = ledger.referent_for_signal(signal) if ledger is not None else None
        if referent is None:
            for known_referent, known_signal in self.referent_lexicon.items():
                if known_signal == signal:
                    referent = known_referent
                    break

        return {
            "agent_id": f"A{self.id}",
            "signal": signal,
            "referent": referent,
            "confidence": 0.85 if referent is not None else 0.10,
        }

    def _solve_compositional_signal(self, task):
        """Parse a grounded two-slot message without a preinstalled order."""
        data = task.get("data", {}) or {}
        tokens = (data.get("signal") or "").split()
        if len(tokens) != 2:
            return self._task_fail(task.get("task_id"))

        ledger = getattr(self, "community_lexicon", None)
        referent = None
        value = None
        order = []
        for token in tokens:
            found_referent = ledger.referent_for_signal(token) if ledger is not None else None
            if found_referent is not None:
                referent = found_referent
                order.append("referent")
                continue
            digit = self.numeric_system.decode_token(token)
            if digit is not None:
                value = digit
                order.append("number")
            else:
                return self._task_fail(task.get("task_id"))

        return {
            "agent_id": f"A{self.id}",
            "referent": referent,
            "value": value,
            "order": tuple(order),
            "utterance": data.get("signal", ""),
            "confidence": 0.85 if referent is not None and value is not None else 0.10,
        }

    def _solve_action_signal(self, task):
        data = task.get("data", {}) or {}
        signal = data.get("signal", "")
        ledger = getattr(self, "community_lexicon", None)
        action = ledger.action_for_signal(signal) if ledger is not None else None
        if action is None:
            for known_action, known_signal in self.action_lexicon.items():
                if known_signal == signal:
                    action = known_action
                    break
        return {
            "agent_id": f"A{self.id}",
            "signal": signal,
            "action": action,
            "confidence": 0.85 if action is not None else 0.10,
        }

    def _solve_human_dictionary_link(self, task):
        """Use the supplied human dictionary to identify a semantic link."""
        data = task.get("data", {}) or {}
        word = data.get("word", "")
        relation = data.get("relation")
        options = data.get("options", []) or []
        dictionary = getattr(self, "human_dictionary", None)
        relations = dictionary.semantic_relations(word) if dictionary is not None else None
        related = [] if relations is None else relations.get(f"{relation}s", [])
        answer = next((option for option in options if option in related), None)
        return {
            "agent_id": f"A{self.id}",
            "word": word,
            "relation": relation,
            "related": answer,
            "confidence": 0.85 if answer is not None else 0.10,
        }

    def _solve_compositional_action_signal(self, task):
        """Parse a grounded referent-action-number message."""
        data = task.get("data", {}) or {}
        tokens = (data.get("signal") or "").split()
        if len(tokens) != 3:
            return self._task_fail(task.get("task_id"))

        ledger = getattr(self, "community_lexicon", None)
        referent = action = value = None
        order = []
        for token in tokens:
            known_referent = ledger.referent_for_signal(token) if ledger is not None else None
            if known_referent is not None:
                referent = known_referent
                order.append("referent")
                continue
            known_action = ledger.action_for_signal(token) if ledger is not None else None
            if known_action is not None:
                action = known_action
                order.append("action")
                continue
            digit = self.numeric_system.decode_token(token)
            if digit is None:
                return self._task_fail(task.get("task_id"))
            value = digit
            order.append("number")

        return {
            "agent_id": f"A{self.id}",
            "referent": referent,
            "action": action,
            "value": value,
            "order": tuple(order),
            "utterance": data.get("signal", ""),
            "confidence": 0.85 if all((referent, action, value is not None)) else 0.10,
        }

    def _solve_semantic_gap(self, task):
        """
        Agent responds to a semantic-gap task:
          - token: the word to inspect
          - community_vec: proposed centroid
          - community_confidence: 0..1
        """
        data = task.get("data", {})
        token = data.get("token")
        c_vec = data.get("community_vec")
        c_conf = float(data.get("community_confidence", 0.0))

        if not token or c_vec is None:
            return None

        sem = getattr(self, "semantic", None)
        if not sem:
            return None

        vecs = sem.get("vecs", {})
        my_vec = vecs.get(token)

        # If we don't know this token yet, lightly adopt community meaning
        if my_vec is None:
            try:
                my_vec = [float(x) for x in c_vec]
                vecs[token] = my_vec[:]
            except Exception:
                return None

        # distance estimate
        try:
            import math
            diff = [a - b for a, b in zip(my_vec, c_vec)]
            dist = math.sqrt(sum(d*d for d in diff))
        except Exception:
            dist = 0.0

        # simple heuristic: if we're not too far, we agree; else misaligned
        agree = dist < 2.0 or c_conf > 0.7

        # vector update
        if agree:
            eta_low = 0.03
            eta_high = 0.25
            eta = eta_low + (eta_high - eta_low) * (c_conf ** 2)
            eta = max(eta_low, min(eta_high, eta))
            new_vec = [
                (1.0 - eta) * a + eta * b
                for a, b in zip(my_vec, c_vec)
            ]
        else:
            eta = min(0.005, 0.02 * (1.0 - c_conf))
            new_vec = [
                a + eta * (a - b)
                for a, b in zip(my_vec, c_vec)
            ]

        vecs[token] = new_vec

        # reward shaping
        base_reward = 0.0
        if agree:
            base_reward = 0.5 * c_conf
        else:
            base_reward = 0.15 * (1.0 - c_conf)

        try:
            self.own_fitness += base_reward
        except Exception:
            pass

        if hasattr(self, "remember_interaction"):
            try:
                self.remember_interaction(
                    partner_id=-1,  # -1 = "community"
                    outcome=base_reward,
                    offspring_success=None,
                    gen_index=getattr(self, "current_generation", 0),
                )
            except Exception:
                pass

        return {
            "token": token,
            "agree": agree,
            "distance": dist,
            "confidence": c_conf,
            "reward": base_reward,
        }

    def _solve_agreement_dialogue(self, task):
        """
        Dialogue task:
            - Two agents negotiate over a topic (two emergent tokens).
            - Success comes from participation & token reuse pressure.
        """

        if not task:
            return None

        my_id = getattr(self, "id", None)
        assigned = task.get("assigned_agents", [])
        if my_id not in assigned:
            return None

        topic = task.get("topic", "")
        tid = task.get("task_id")

        # split topic
        try:
            a_tok, b_tok = [t.strip() for t in topic.split("||")]
        except Exception:
            a_tok, b_tok = "su", "tol"

        # 1) Build my proposal: small utterance linking them
        anchor = None
        if hasattr(self, "_ensure_concept_token"):
            anchors = ["agree", "align", "relate", "bind"]
            base_anchor = random.choice(anchors)
            try:
                anchor = self._ensure_concept_token(base_anchor, tag="rel")
            except Exception:
                anchor = base_anchor

        # semantic neighbors add flavour
        nbrs = []
        try:
            nb = self.semantic_system._semantic_neighbors(anchor, k=2)
            if nb:
                nbrs.append(random.choice(nb))
        except Exception:
            pass

        # free-language tail
        tail = []
        if hasattr(self, "produce_utterance") and random.random() < 0.6:
            try:
                tail = (self.produce_utterance() or "").split()[:2]
            except Exception:
                tail = []

        # choose which topic tokens to emphasise
        candidates = [a_tok, b_tok]
        chosen = []
        sem = getattr(self, "semantic", {})
        vecs = sem.get("vecs", {})

        for tok in candidates:
            if tok in vecs:
                chosen.append(tok)

        if not chosen:
            chosen = [random.choice(candidates)]

        if len(chosen) == 2 and random.random() < 0.5:
            chosen = [random.choice(chosen)]

        my_tokens = ["why"]
        if anchor:
            my_tokens.append(anchor)
        my_tokens.extend(chosen)
        my_tokens.extend(nbrs)
        my_tokens.extend(tail)
        my_tokens = [t for t in my_tokens if isinstance(t, str) and t.strip()]
        my_tokens = my_tokens[:8]
        proposal = " ".join(my_tokens) if my_tokens else "su tol"

        self.last_dialogue_proposal = my_tokens

        # participation reward
        try:
            self.own_fitness += 0.15
        except Exception:
            pass

        # semantic learning
        if hasattr(self, "_observe_tokens"):
            try:
                self._observe_tokens(my_tokens, gain=0.1)
            except Exception:
                pass

        return {
            "agent_id": f"A{self.id}",
            "proposal": proposal,
            "reused_partner_tokens": False,  # left for future use
        }

    def _solve_explain_partner_answer(self, task):
        """
        Explain-Partner task.
        Agent sees partner's answer (raw tokens) and must produce
        a paraphrase or interpretation using its own vocabulary.
        """

        if not task:
            return None

        my_id = getattr(self, "id", None)
        assigned = task.get("assigned_agents", [])
        if my_id not in assigned or len(assigned) != 2:
            return None

        partner_id = assigned[0] if assigned[1] == my_id else assigned[1]
        partner_answer = task.get("partner_answer", "")
        partner_tokens = partner_answer.split() if partner_answer else []

        # 1) Anchor: relation token
        anchor = None
        if hasattr(self, "_ensure_concept_token"):
            try:
                anchor = self._ensure_concept_token("rel", tag="rel") or "rel"
            except Exception:
                anchor = "rel"

        # 2) Try to reuse some partner tokens
        reused = []
        sem = getattr(self, "semantic", {})
        vecs = sem.get("vecs", {})

        for t in partner_tokens:
            if t in vecs and random.random() < 0.6:
                reused.append(t)

        # 3) Add my own flavour / structure
        structured = []
        if random.random() < 0.4 and anchor:
            structured.append(anchor)
        if hasattr(self, "produce_utterance") and random.random() < 0.3:
            try:
                structured.append((self.produce_utterance() or "").split()[0])
            except Exception:
                pass

        tokens = ["why"] + reused[:3] + structured[:3]
        tokens = [t for t in tokens if isinstance(t, str) and t.strip()]
        explanation = " ".join(tokens) if tokens else "why " + (reused[0] if reused else "su")

        # Fitness
        try:
            self.own_fitness += 0.1  # participation
            if reused:
                self.own_fitness += 0.1
        except Exception:
            pass

        if reused and hasattr(self, "adjust_trust"):
            try:
                self.adjust_trust(partner_id, +0.02, channel=2)
            except Exception:
                pass

        # Learning
        if hasattr(self, "_observe_tokens"):
            try:
                self._observe_tokens(tokens, gain=0.1)
            except Exception:
                pass

        return {
            "agent_id": f"A{self.id}",
            "explanation": explanation,
            "tokens_reused": reused,
        }
