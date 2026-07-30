"""
agents/cognition/task_system_v2.py
Task solving system (v2).
"""

import random
import re

from agents.agent_constants import REL_GT, REL_LT, REL_EQ


class TaskSystemNumericMixin:
    def _solve_compare_numbers(self, task):
        """
        Single-agent numeric comparison:
        - decode A and B with the reference numeric system if provided
        - otherwise fall back to this agent's own map
        - if either fails, request numeric help
        - otherwise choose the larger (or A if equal)
        - justification uses emergent rel* concept tokens
        """
        data = task.get("data", {}) or {}
        A = data.get("A", "") or ""
        B = data.get("B", "") or ""

        inst = task.get("instruction", {}) or {}
        override_map = inst.get("symbol_map", None)
        override_base = inst.get("base", None)

        # decode using override if present, else local
        valA = self._decode_number_phrase(A, override_map, override_base)
        valB = self._decode_number_phrase(B, override_map, override_base)

        # 1) request help if number not understood
        if valA is None or valB is None:
            ask = A if len(A.split()) >= len(B.split()) else B
            if ask and hasattr(self, "maybe_request_numeric_help"):
                try:
                    self.maybe_request_numeric_help(ask, None)
                except Exception:
                    pass
            return self._task_fail(task.get("task_id"))

        # 2) compare → relation
        if valA > valB:
            relation = REL_GT
            answer = A
        elif valB > valA:
            relation = REL_LT
            answer = B
        else:
            relation = REL_EQ
            answer = A

        just = self._build_relation_justification(relation)

        if hasattr(self, "_observe_tokens"):
            try:
                self._observe_tokens(just.split(), gain=0.15)
            except Exception:
                pass

        # 5) relation-truth teaching (optional, no vocab)
        self._maybe_emit_relation_teach(A, B, relation)

        # 6) confidence shaping
        if relation == REL_EQ:
            conf = 0.45 + 0.1 * random.random()
        else:
            conf = 0.55 + 0.2 * random.random()

        if hasattr(self, "adjust_trust"):
            try:
                self.adjust_trust(self.id, +0.015, channel=4)
            except Exception:
                pass

        return {
            "agent_id": f"A{self.id}",
            "answer": answer,
            "justification": just,
            "confidence": round(conf, 3),
        }

    def _solve_cooperative_compare_numbers(self, task):
        """
        Cooperative variant:
          - only agents in task["assigned_agents"] can see it
          - each agent decides whether to:
                * answer directly
                * request numeric help
          - both still generate justifications with a relation concept token
        """
        data = task.get("data", {})
        A = data.get("A", "")
        B = data.get("B", "")

        assigned = task.get("assigned_agents", [])
        my_id = getattr(self, "id", None)

        # identify partner (if any)
        partner_id = None
        if my_id is not None and len(assigned) >= 2:
            for aid in assigned:
                if aid != my_id:
                    partner_id = aid
                    break

        # Decode both sides (may fail)
        inst = task.get("instruction", {})
        override_map = inst.get("symbol_map")
        override_base = inst.get("base")

        valA = self._decode_number_phrase(A, override_map, override_base)
        valB = self._decode_number_phrase(B, override_map, override_base)

        # --------------------------------------------------
        # 1) If we can't decode, decide whether to ask partner
        # --------------------------------------------------
        if valA is None or valB is None:
            coop_w = getattr(self, "cooperation_weight", 0.3)

            # partner trust (channel 4 competence-ish), if available
            partner_trust = 0.0
            tc = getattr(self, "trust_channels", None)
            if partner_id is not None and isinstance(tc, dict):
                ch = tc.get(partner_id, {})
                if isinstance(ch, dict):
                    partner_trust = (
                        0.4 * ch.get("affinity", 0.0)
                        + 0.3 * ch.get("reliability", 0.0)
                        + 0.3 * ch.get("competence", 0.0)
                    )

            if (
                partner_id is not None
                and hasattr(self, "maybe_request_numeric_help")
                and random.random() < coop_w * (0.5 + 0.5 * partner_trust)
            ):
                ask = A if len(A.split()) >= len(B.split()) else B
                if ask:
                    try:
                        self.maybe_request_numeric_help(ask, None)
                    except Exception:
                        pass
                    self._log_coop_event(
                        task,
                        event="coop_request_help",
                        extra=f"partner=A{partner_id} phrase='{ask}'",
                    )
                return self._task_fail(task.get("task_id"))

            # otherwise fall back to normal failure + help request
            ask = A if len(A.split()) >= len(B.split()) else B
            if ask and hasattr(self, "maybe_request_numeric_help"):
                try:
                    self.maybe_request_numeric_help(ask, None)
                except Exception:
                    pass
            return self._task_fail(task.get("task_id"))

        # --------------------------------------------------
        # 2) Decode succeeded: compare as usual
        # --------------------------------------------------
        if valA > valB:
            relation = REL_GT
            answer = A
        elif valB > valA:
            relation = REL_LT
            answer = B
        else:
            relation = REL_EQ
            answer = A

        # relation justification with concept-token
        just = self._build_relation_justification(relation)

        if hasattr(self, "_observe_tokens"):
            try:
                self._observe_tokens(just.split(), gain=0.18)
            except Exception:
                pass

        # --------------------------------------------------
        # 3) Cooperative trust shaping
        # --------------------------------------------------
        if partner_id is not None and hasattr(self, "adjust_trust"):
            try:
                self.adjust_trust(partner_id, +0.03, channel=4)
                self.adjust_trust(partner_id, +0.02, channel=2)
            except Exception:
                pass

        # mutual self-trust bump
        if hasattr(self, "adjust_trust"):
            try:
                self.adjust_trust(self.id, +0.02, channel=4)
            except Exception:
                pass

        # occasionally log a co-op success event
        self._log_coop_event(
            task,
            event="coop_solved",
            extra=f"partner=A{partner_id}" if partner_id is not None else "partner=None",
        )

        # relation teaching as in single-agent mode
        self._maybe_emit_relation_teach(A, B, relation)

        # confidence slightly higher in co-op context
        if relation == REL_EQ:
            conf = 0.50 + 0.12 * random.random()
        else:
            conf = 0.60 + 0.20 * random.random()

        return {
            "agent_id": f"A{self.id}",
            "answer": answer,
            "justification": just,
            "confidence": round(conf, 3),
        }

    def _build_relation_justification(self, relation):
        """
        Build an emergent justification utterance for a relation ("gt"/"lt"/"eq"):

          - anchor on a rel* concept token
          - expand through rel_family and semantic neighbors
          - optionally pull in numeric tokens from the last comparison
          - optionally add a short free utterance tail
        """
        sem = getattr(self, "semantic", None)
        concept_tok = None

        # 1) relation → rel-concept anchor
        if hasattr(self, "_ensure_concept_token"):
            try:
                concept_tok = self._ensure_concept_token(relation, tag="rel")
            except Exception:
                concept_tok = None

        rel_chain = []
        if concept_tok and isinstance(concept_tok, str):
            rel_chain.append(concept_tok)

        # 2) rel-family variants (existing meta-family)
        rel_family_tokens = []
        if sem is not None:
            rel_family_tokens = list(sem.get("rel_family", {}).keys())

        # allow rel-family to replace/augment the raw concept token
        if rel_family_tokens:
            if random.random() < 0.7:
                rel_chain = [random.choice(rel_family_tokens)]
            if random.random() < 0.4 and len(rel_family_tokens) > 1:
                candidate = random.choice(rel_family_tokens)
                if candidate not in rel_chain:
                    rel_chain.append(candidate)

        # 3) semantic neighbors of the reasoning chain
        neighbor_tokens = []
        for t in rel_chain:
            try:
                nbrs = self.semantic_system._semantic_neighbors(t, k=3)
            except Exception:
                nbrs = []
            if nbrs:
                neighbor_tokens.append(random.choice(nbrs))

        # 4) numeric evidence from last numeric phrase (if any)
        numeric_tokens = []
        if hasattr(self, "_last_tokens") and hasattr(self, "symbol_map"):
            try:
                symvals = set(self.symbol_map.values())
                for t in (self._last_tokens or []):
                    if t in symvals:
                        numeric_tokens.append(t)
                numeric_tokens = numeric_tokens[:2]
            except Exception:
                numeric_tokens = []

        # 5) tiny emergent tail from general utterance generator
        tail_tokens = []
        if hasattr(self, "produce_utterance") and random.random() < 0.5:
            try:
                tail = self.produce_utterance() or ""
                tail_tokens = tail.split()[:2]
            except Exception:
                tail_tokens = []

        # 6) Assemble
        just_tokens = ["why"]
        just_tokens.extend(rel_chain)
        just_tokens.extend(neighbor_tokens)
        just_tokens.extend(tail_tokens)

        # filter & cap
        just_tokens = [
            t for t in just_tokens
            if isinstance(t, str) and t.strip()
        ]
        if not just_tokens:
            if hasattr(self, "produce_utterance"):
                try:
                    return self.produce_utterance() or "why"
                except Exception:
                    return "why"
            return "why"

        just_tokens = just_tokens[:8]
        return " ".join(just_tokens)

    def _maybe_emit_relation_teach(self, A, B, relation):
        if not hasattr(self, "api") or self.api is None:
            return
        if relation not in (REL_GT, REL_LT, REL_EQ):
            return

        if random.random() >= 0.25:
            return

        try:
            line = f"A{self.id} teach_relation {A} || {B} rel={relation}"
            if hasattr(self.api, "append_help_response"):
                self.api.append_help_response(line)
            else:
                self.api.append_text("/help_responses.txt", line + "\n", scope="world")
        except Exception:
            pass

    def _log_coop_event(self, task, event, extra=""):
        if not hasattr(self, "api") or self.api is None:
            return
        try:
            tid = task.get("task_id", "unknown")
            ttype = task.get("task_type", "unknown")
            line = (
                f"A{self.id} coop_event task={tid} type={ttype} "
                f"event={event} {extra}"
            )
            if hasattr(self.api, "append_text"):
                self.api.append_text("/coop_events.txt", line + "\n", scope="world")
        except Exception:
            pass

    def _decode_number_phrase(self, phrase, override_map=None, override_base=None):
        if not phrase:
            return None

        sym_map = override_map if override_map else self.symbol_map
        b = override_base if override_base else self.counting.base

        rev = {}
        if override_map is None:
            ledger = getattr(self, "community_lexicon", None)
            if ledger is not None and hasattr(ledger, "numeric_conventions"):
                try:
                    rev.update({token: digit for digit, token in ledger.numeric_conventions().items()})
                except Exception:
                    pass
        for digit, token in sym_map.items():
            # Public conventions take precedence over a stale private map.
            rev.setdefault(token, digit)

        digits = []
        for t in phrase.split():
            if t not in rev:
                return None
            d = rev[t]
            if not (0 <= d < b):
                return None
            digits.append(d)

        out = 0
        for d in digits:
            out = out * b + d

        if not isinstance(out, int):
            return None

        return out

    def _digits_for_tokens(self, tokens):
        if not hasattr(self, "symbol_map"):
            return None
        rev = {v: k for k, v in self.symbol_map.items()}
        try:
            return [rev[t] for t in tokens]
        except KeyError:
            return None

    def _solve_reconcile_counts(self, task):
        """Interpret another agent's numeral and report a normalised value."""
        views = task.get("views", {}) or {}
        phrase = views.get(self.id, views.get(str(self.id), ""))
        if not phrase:
            return self._task_fail(task.get("task_id"))

        ledger = getattr(self, "community_lexicon", None)
        shared_base = ledger.community_base() if ledger is not None else None
        decode_base = shared_base or self.numeric_system.base
        value = self._decode_number_phrase(phrase, override_base=decode_base)
        confidence = 0.85
        if value is None:
            confidence = 0.20
            value = 0
            for tok in phrase.split():
                digit = self.numeric_system.decode_token(tok)
                if digit is None:
                    return self._task_fail(task.get("task_id"))
                value = value * decode_base + digit

        return {
            "agent_id": f"A{self.id}",
            "normalized": value,
            "utterance": phrase,
            "confidence": confidence,
        }
